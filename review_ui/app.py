"""
ClinScribe review UI — FastAPI backend.
Uses FastAPI (not Flask) because Whisper inference takes 30-60s; Flask would block all requests.
"""

import io
import json
import logging
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from pipeline.extract import extract
from pipeline.extract_offline import extract_offline
from pipeline.fhir_write import (
    OpenMRSUnavailableError,
    write_encounter_and_conditions,
    verify_openmrs_connection,
)
from pipeline.map_codes import map_codes
from pipeline.transcribe import transcribe, translate_segments
from pipeline.translate_offline import translate_segments_offline

logger = logging.getLogger(__name__)

app = FastAPI(title="ClinScribe", version="1.0.0")

# Static files (index.html)
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_ERROR_LOG = Path(__file__).parent.parent / "ERROR_LOG.md"


def _log_error(step: str, action: str, error_type: str, message: str) -> None:
    """
    What it does: Appends an error entry to ERROR_LOG.md.
    Inputs: step — pipeline step name; action — what was happening; error_type/message — str
    Outputs: None
    Dependencies: ERROR_LOG.md must be writable
    Side effects: Appends to ERROR_LOG.md
    Failure modes: Non-fatal — logs warning if file not writable
    """
    entry = (
        f"\n## ERROR — {datetime.now(timezone.utc).isoformat()}\n"
        f"**Step:** {step}\n"
        f"**What I was doing:** {action}\n"
        f"**Error type:** {error_type}\n"
        f"**Full error message:**\n```\n{message}\n```\n"
        f"**Resolution:** BLOCKED — needs investigation\n---\n"
    )
    try:
        with open(_ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(entry)
    except OSError:
        logger.warning("Could not write to ERROR_LOG.md")


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    What it does: Checks OpenMRS connectivity and Whisper load status.
    Never fails — always returns 200 with status fields.
    """
    openmrs_ok = False
    try:
        sess = _openmrs_session()
        r = sess.get(f"{config.OPENMRS_BASE_URL}/ws/rest/v1/session", timeout=5)
        openmrs_ok = r.json().get("authenticated", False)
    except Exception:
        pass

    # Whisper is loaded at transcribe module import time
    try:
        from pipeline.transcribe import _whisper_model
        whisper_loaded = _whisper_model is not None
    except Exception:
        whisper_loaded = False

    return {
        "status": "ok",
        "openmrs_reachable": openmrs_ok,
        "whisper_loaded": whisper_loaded,
        "openmrs_spa_url": config.OPENMRS_SPA_URL,
    }


# ── OpenMRS session (OpenMRS 3 disables Basic Auth; use session cookie instead) ──

_session_store: requests.Session | None = None
_session_lock = threading.Lock()

def _openmrs_session() -> requests.Session:
    """
    Returns an authenticated requests.Session for OpenMRS.
    OpenMRS 3 disables Basic Auth on the REST API by default, so we authenticate
    via POST to /session with JSON credentials to get a JSESSIONID cookie, then
    reuse that session for all subsequent calls.
    """
    global _session_store
    with _session_lock:
        # Reuse existing session if still authenticated
        if _session_store is not None:
            try:
                r = _session_store.get(
                    f"{config.OPENMRS_BASE_URL}/ws/rest/v1/session", timeout=3
                )
                if r.json().get("authenticated"):
                    return _session_store
            except Exception:
                pass

        s = requests.Session()

        # Try 1: POST to /session with Basic Auth — standard OpenMRS REST auth
        s.auth = (config.OPENMRS_USER, config.OPENMRS_PASSWORD)
        try:
            r = s.post(f"{config.OPENMRS_BASE_URL}/ws/rest/v1/session", timeout=5)
            if r.json().get("authenticated"):
                _session_store = s
                return s
        except Exception:
            pass

        # Try 2: GET with Basic Auth (older OpenMRS configs)
        try:
            r = s.get(f"{config.OPENMRS_BASE_URL}/ws/rest/v1/session", timeout=5)
            if r.json().get("authenticated"):
                _session_store = s
                return s
        except Exception:
            pass

        # Try 3: POST JSON credentials
        s2 = requests.Session()
        try:
            r = s2.post(
                f"{config.OPENMRS_BASE_URL}/ws/rest/v1/session",
                json={"username": config.OPENMRS_USER, "password": config.OPENMRS_PASSWORD},
                timeout=5,
            )
            if r.json().get("authenticated"):
                _session_store = s2
                return s2
        except Exception:
            pass

        # Try 4: legacy login servlet (form POST)
        s3 = requests.Session()
        try:
            r3 = s3.post(
                f"{config.OPENMRS_BASE_URL}/loginServlet",
                data={
                    "uname": config.OPENMRS_USER,
                    "pw": config.OPENMRS_PASSWORD,
                    "redirect": "/openmrs",
                    "refererURL": "",
                },
                timeout=5,
                allow_redirects=True,
            )
            # Verify the session cookie actually works
            check = s3.get(f"{config.OPENMRS_BASE_URL}/ws/rest/v1/session", timeout=5)
            if check.json().get("authenticated"):
                _session_store = s3
                return s3
        except Exception:
            pass

        _session_store = s
        return s


# ── Patient search ─────────────────────────────────────────────────────────────

@app.get("/patients")
async def search_patients(q: str = Query(default="")):
    """
    What it does: Proxies a patient name search to OpenMRS REST v1 API.
    Uses session-based auth because OpenMRS 3 disables Basic Auth on REST endpoints.
    Returns empty list (not error) if OpenMRS unreachable — UI shows a warning.
    """
    if not q.strip():
        return {"patients": []}

    url = f"{config.OPENMRS_BASE_URL}/ws/rest/v1/patient"
    try:
        sess = _openmrs_session()
        resp = sess.get(url, params={"q": q, "v": "default", "limit": 10}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        patients = []
        for p in data.get("results", []):
            uuid = p.get("uuid", "")
            display = p.get("display", "") or p.get("person", {}).get("display", "")
            identifiers = p.get("identifiers", [])
            identifier = identifiers[0].get("identifier", "") if identifiers else ""
            patients.append({"uuid": uuid, "display": display, "identifier": identifier})
        return {"patients": patients}
    except requests.ConnectionError:
        return {"patients": [], "warning": "OpenMRS not reachable"}
    except Exception as e:
        logger.error(f"Patient search error: {e}")
        return {"patients": [], "warning": str(e)}


# ── Transcribe ────────────────────────────────────────────────────────────────

_ALLOWED_AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp3", "audio/ogg"}
_ALLOWED_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a"}


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    mode: str = Form("online"),
):
    """
    What it does: Runs the full pipeline (transcribe → translate → extract → map_codes).
    mode='online'  — uses Claude API for translate + extract (requires internet + API key)
    mode='offline' — uses IndicTrans2 + vLLM for translate + extract (fully local)
    Accepts MP3/WAV. Returns all pipeline results with per-stage timing.
    On any pipeline error: returns 500 with stage name and error detail.
    """
    if mode == "online" and not config.ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="Online mode requires ANTHROPIC_API_KEY in .env"
        )
    ext = Path(audio.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Accepted: {_ALLOWED_EXTENSIONS}"
        )

    # Write to temp WAV file — torchaudio on Windows cannot decode MP3/M4A/OGG
    # directly. Convert everything to WAV via pydub (uses ffmpeg under the hood).
    import tempfile, os
    from pydub import AudioSegment

    raw_bytes = await audio.read()
    fmt = ext.lstrip(".")
    if fmt == "mp3":
        fmt = "mp3"
    elif fmt in ("m4a",):
        fmt = "mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name

    try:
        seg = AudioSegment.from_file(io.BytesIO(raw_bytes), format=fmt)
        seg.export(tmp_path, format="wav")
    except Exception as conv_err:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {conv_err}")

    timings = {}
    try:
        # Stage 1: transcribe
        t0 = time.monotonic()
        transcript_result = transcribe(tmp_path)
        timings["transcribe"] = round(time.monotonic() - t0, 2)

        # Stage 2: translate
        t0 = time.monotonic()
        if mode == "offline":
            translated_segments = translate_segments_offline(list(transcript_result["segments"]))
        else:
            translated_segments = translate_segments(list(transcript_result["segments"]))
        timings["translate"] = round(time.monotonic() - t0, 2)

        # Stage 3: extract — run on English translation so all entities are in English.
        # Diagnoses/symptoms extracted from Hindi would be in Devanagari and unusable
        # for OpenMRS concept matching and doctor review.
        t0 = time.monotonic()
        english_full_text = " ".join(
            seg.get("english_translation", seg.get("text", ""))
            for seg in translated_segments
        )
        if mode == "offline":
            entities = extract_offline(english_full_text)
        else:
            entities = extract(english_full_text)
        timings["extract"] = round(time.monotonic() - t0, 2)

        # Stage 4: map codes (on diagnosis list)
        t0 = time.monotonic()
        diagnoses = [
            d.get("text", d) if isinstance(d, dict) else d
            for d in entities.get("diagnosis", [])
        ]
        coded = map_codes(diagnoses)
        timings["map_codes"] = round(time.monotonic() - t0, 2)

        timings["total"] = round(sum(timings.values()), 2)

        return {
            "segments": translated_segments,
            "entities": entities,
            "mapped_codes": coded,
            "processing_time_seconds": timings,
            "mode": mode,
        }

    except Exception as e:
        stage = max(timings, key=timings.get) if timings else "transcribe"
        _log_error("Step 12 — /transcribe", f"Running pipeline stage: {stage}", type(e).__name__, traceback.format_exc())
        raise HTTPException(status_code=500, detail={"error": stage, "detail": str(e)})
    finally:
        os.unlink(tmp_path)


# ── Approve ───────────────────────────────────────────────────────────────────

class ApproveRequest(BaseModel):
    patient_uuid: str
    edited_entities: dict
    mapped_codes: list
    vitals: dict = {}
    clinical_notes: str = ""


@app.post("/approve")
async def approve_and_write(req: ApproveRequest):
    """
    What it does: Writes approved transcription to OpenMRS via FHIR R4 after doctor review.

    # HUMAN APPROVAL GATE — only write to OpenMRS after explicit doctor approval.
    # Removing this gate means unreviewed AI data enters patient records — patient safety risk.
    """
    if not req.patient_uuid or not req.patient_uuid.strip():
        raise HTTPException(status_code=400, detail="patient_uuid is required")

    logger.warning(f"[approve] diagnoses received: {req.edited_entities.get('diagnosis', [])}")

    try:
        result = write_encounter_and_conditions(
            patient_uuid=req.patient_uuid,
            entities=req.edited_entities,
            mapped_codes=req.mapped_codes,
            session=_openmrs_session(),
            vitals=req.vitals,
            clinical_notes=req.clinical_notes,
        )
        return result
    except OpenMRSUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        _log_error("Step 12 — /approve", "Writing to OpenMRS FHIR", type(e).__name__, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve index.html at root ──────────────────────────────────────────────────

from fastapi.responses import FileResponse

@app.get("/")
async def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))
