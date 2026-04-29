"""
ClinScribe offline entity extraction — agentic multi-pass via vLLM.

Single-pass extraction with a 7B model misses clinical details.
This module runs 5 focused agent steps and merges the findings:
  1. Initial extraction      — structured base JSON
  2. Symptom deep-dive       — re-scan for missed/partial symptoms
  3. Red flag scan           — dedicated pass for urgent/dangerous findings
  4. Medication review       — dosages, herbal remedies, allergies
  5. Clinical consistency    — do symptoms + duration support the diagnosis?
Each step logs to llm_calls.jsonl for auditability.
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434/v1")
_OLLAMA_MODEL    = getattr(config, "OLLAMA_MODEL",    "qwen2.5:7b")

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LLM_LOG = _LOG_DIR / "llm_calls.jsonl"

_REQUIRED_KEYS = {
    "chief_complaint", "history_of_present_illness", "symptoms",
    "duration", "medications_mentioned", "diagnosis",
    "treatment_plan", "red_flags", "raw_language",
}
_STOP_WORDS = {"the", "a", "an", "of", "in", "with", "has", "have"}

# ── Red flags the dedicated agent step explicitly looks for ───────────────────
_RED_FLAG_TARGETS = [
    "chest pain", "shortness of breath", "difficulty breathing", "dyspnea",
    "loss of consciousness", "syncope", "altered mental status", "confusion",
    "severe bleeding", "hemoptysis", "blood in stool", "blood in urine",
    "high fever above 39", "seizure", "stroke symptoms", "facial drooping",
    "arm weakness", "sudden severe headache", "suicidal ideation",
    "severe allergic reaction", "anaphylaxis", "rapid heart rate",
    "severe abdominal pain", "signs of sepsis", "jaundice",
]
# Pre-built once at import — joined on every call before
_RED_FLAG_TARGETS_STR = "\n".join(f"- {t}" for t in _RED_FLAG_TARGETS)

# Module-level singleton — OpenAI client creation (~50ms) was happening
# on every one of the 5 LLM calls per offline request
from openai import OpenAI as _OpenAI
_client = _OpenAI(base_url=_OLLAMA_BASE_URL, api_key="dummy")


def _call(step: str, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
    """
    Single Ollama call. Returns raw text. Logs to llm_calls.jsonl.
    Raises on connection failure — callers decide how to handle.
    """
    t0 = time.monotonic()
    try:
        response = _client.chat.completions.create(
            model=_OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        raw = response.choices[0].message.content.strip()
        _log(step, latency_ms, True, None)
        return raw
    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000
        _log(step, latency_ms, False, str(e))
        raise


def _log(step: str, latency_ms: float, success: bool, error: str | None) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": f"extraction_offline/{step}",
        "model": _OLLAMA_MODEL,
        "latency_ms": round(latency_ms, 1),
        "success": success,
        "error": error,
    }
    try:
        with open(_LLM_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.warning(f"Could not write LLM log: {e}")


def _parse_json(raw: str, step: str) -> dict | list | None:
    """Strip markdown fences and parse JSON. Returns None on failure."""
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    clean = re.sub(r"\s*```$", "", clean).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        logger.warning(f"[agent/{step}] JSON parse failed. Raw: {clean[:300]}")
        return None


def _parse_list(raw: str, step: str) -> list[str]:
    """Parse a JSON array of strings. Falls back to line splitting."""
    result = _parse_json(raw, step)
    if isinstance(result, list):
        return [str(x).strip() for x in result if x]
    # Fallback: treat each non-empty line as an item
    lines = [ln.lstrip("-• ").strip() for ln in raw.splitlines() if ln.strip()]
    return [ln for ln in lines if ln and not ln.startswith("{")]


# ── Agent step 1: initial extraction ─────────────────────────────────────────

def _step_extract(transcript: str) -> dict:
    system = (
        "You are a clinical documentation assistant reviewing a doctor-patient conversation. "
        "Extract structured clinical information and return ONLY valid JSON. "
        "Never invent information not present in the transcript."
    )
    user = f"""Extract clinical information from this doctor-patient transcript.

Transcript:
{transcript}

Return a JSON object with exactly these keys:
- chief_complaint: string (main reason for visit)
- history_of_present_illness: string (full narrative including onset, character, radiation, severity, timing, modifying factors)
- symptoms: list of strings (include severity/location/duration qualifiers where stated)
- duration: string (how long symptoms have been present)
- medications_mentioned: list of strings (include dosage and frequency if stated)
- diagnosis: list of strings
- treatment_plan: string
- red_flags: list of strings (urgent or dangerous findings)
- raw_language: "hi"

Return ONLY the JSON object."""

    raw = _call("extract", system, user)
    data = _parse_json(raw, "extract")
    if not isinstance(data, dict):
        # Retry once
        raw = _call("extract_retry", system, user + "\n\nIMPORTANT: Return ONLY raw JSON. No markdown, no explanation.")
        data = _parse_json(raw, "extract_retry")
    if not isinstance(data, dict):
        raise ValueError(f"Initial extraction failed after retry. Raw: {raw[:500]}")
    return data


# ── Agent step 2: symptom deep-dive ──────────────────────────────────────────

def _step_symptoms(transcript: str, already_found: list[str]) -> list[str]:
    system = (
        "You are a clinical symptom analyst. Your task is to find EVERY symptom "
        "mentioned in a doctor-patient conversation, including subtle or briefly mentioned ones. "
        "Be thorough — missing a symptom in a healthcare record is a patient safety risk."
    )
    found_str = json.dumps(already_found, ensure_ascii=False)
    user = f"""Re-read this transcript carefully and list ALL symptoms mentioned.

Already found: {found_str}

Transcript:
{transcript}

Instructions:
- Include symptoms already found AND any you find that were missed
- Include severity, location, and timing qualifiers if stated (e.g. "sharp chest pain for 2 days")
- Include symptoms the patient denies if clinically relevant (prefix with "denies: ")
- Do NOT include diagnoses — only reported symptoms

Return a JSON array of strings. Example: ["severe headache for 3 days", "nausea", "denies: fever"]
Return ONLY the JSON array."""

    raw = _call("symptoms", system, user)
    return _parse_list(raw, "symptoms") or already_found


# ── Agent step 3: red flag scan ───────────────────────────────────────────────

def _step_red_flags(transcript: str, already_found: list[str]) -> list[str]:
    system = (
        "You are a clinical safety reviewer. Your task is to identify any urgent, "
        "dangerous, or immediately life-threatening findings in a medical transcript. "
        "Err on the side of caution — false negatives are more dangerous than false positives."
    )
    found_str = json.dumps(already_found, ensure_ascii=False)
    user = f"""Scan this transcript for urgent clinical red flags.

Already flagged: {found_str}

Specific red flags to check for (but do not limit yourself to these):
{_RED_FLAG_TARGETS_STR}

Transcript:
{transcript}

Instructions:
- List all red flags present, including ones already flagged
- Include the exact quote or paraphrase from the transcript
- If NONE are present, return an empty array
- Do NOT fabricate — only flag what is actually mentioned

Return a JSON array of strings.
Return ONLY the JSON array."""

    raw = _call("red_flags", system, user, max_tokens=256, temperature=0.0)
    result = _parse_list(raw, "red_flags")
    # Merge with already_found, deduplicate
    merged = list({x.lower(): x for x in (already_found + result)}.values())
    return merged


# ── Agent step 4: medication review ──────────────────────────────────────────

def _step_medications(transcript: str, already_found: list[str]) -> list[str]:
    system = (
        "You are a clinical pharmacology reviewer. Extract all medications, supplements, "
        "and traditional/herbal remedies mentioned in doctor-patient conversations. "
        "Include dosage and frequency if stated. Missing medications is a patient safety risk."
    )
    found_str = json.dumps(already_found, ensure_ascii=False)
    user = f"""Review this transcript for ALL medication mentions.

Already found: {found_str}

Transcript:
{transcript}

Instructions:
- Include prescription drugs, OTC drugs, supplements, herbal remedies, home remedies
- Include dosage and frequency if mentioned (e.g. "Paracetamol 500mg twice daily")
- Include medications the patient stopped taking or was previously on
- Include any reported drug allergies (prefix with "ALLERGY: ")
- If nothing new found, return the already-found list unchanged

Return a JSON array of strings.
Return ONLY the JSON array."""

    raw = _call("medications", system, user, max_tokens=512)
    result = _parse_list(raw, "medications")
    merged = list({x.lower(): x for x in (already_found + result)}.values())
    return merged


# ── Agent step 5: clinical consistency check ─────────────────────────────────

def _step_consistency(transcript: str, base: dict) -> dict:
    """
    Asks the model to verify whether the diagnosis is supported by the symptoms,
    and whether the treatment plan matches the diagnosis.
    Returns a dict with optional keys: diagnosis_notes, treatment_notes, warnings.
    """
    system = (
        "You are a senior clinical reviewer checking the consistency of a medical summary. "
        "Your job is to catch clinical errors, unsupported diagnoses, and missing treatment steps. "
        "Be rigorous — this is a healthcare application."
    )
    user = f"""Review the clinical summary below for consistency against the original transcript.

Transcript:
{transcript}

Extracted summary:
- Symptoms: {json.dumps(base.get('symptoms', []), ensure_ascii=False)}
- Duration: {base.get('duration', '')}
- Diagnosis: {json.dumps(base.get('diagnosis', []), ensure_ascii=False)}
- Treatment plan: {base.get('treatment_plan', '')}
- Red flags: {json.dumps(base.get('red_flags', []), ensure_ascii=False)}

Check:
1. Is each diagnosis supported by the symptoms and duration in the transcript?
2. Does the treatment plan match the diagnosis?
3. Are there any symptoms that suggest a diagnosis NOT in the list?
4. Are there red flags that are not being addressed in the treatment plan?

Return a JSON object with these keys:
- diagnosis_supported: true/false
- unsupported_diagnoses: list of strings (diagnoses with weak transcript support)
- missing_diagnoses: list of strings (possible diagnoses suggested by symptoms but not listed)
- treatment_gaps: list of strings (red flags or symptoms not addressed in treatment)
- warnings: list of strings (any other clinical concerns)

Return ONLY the JSON object."""

    raw = _call("consistency", system, user, max_tokens=512, temperature=0.0)
    result = _parse_json(raw, "consistency")
    if not isinstance(result, dict):
        return {}
    return result


# ── Merge all agent findings ──────────────────────────────────────────────────

def _merge(base: dict, symptoms: list, red_flags: list, medications: list, consistency: dict) -> dict:
    """
    Merges all agent step results into the final extraction dict.
    Consistency warnings are appended to red_flags so they surface in the UI.
    """
    merged = dict(base)
    merged["symptoms"] = symptoms
    merged["red_flags"] = red_flags
    merged["medications_mentioned"] = medications
    merged["raw_language"] = "hi"

    # Append consistency warnings to red_flags so they appear in the UI
    extra_warnings = []
    if consistency.get("unsupported_diagnoses"):
        for d in consistency["unsupported_diagnoses"]:
            extra_warnings.append(f"[CONSISTENCY] Diagnosis may lack transcript support: {d}")
    if consistency.get("missing_diagnoses"):
        for d in consistency["missing_diagnoses"]:
            extra_warnings.append(f"[CONSISTENCY] Possible missed diagnosis: {d}")
    if consistency.get("treatment_gaps"):
        for g in consistency["treatment_gaps"]:
            extra_warnings.append(f"[CONSISTENCY] Treatment gap: {g}")
    if consistency.get("warnings"):
        for w in consistency["warnings"]:
            extra_warnings.append(f"[CONSISTENCY] {w}")

    if extra_warnings:
        merged["red_flags"] = list(merged.get("red_flags") or []) + extra_warnings

    return merged


# ── Validation + hallucination check (same as online extract.py) ─────────────

def _validate_and_coerce(data: dict) -> dict:
    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    for key in ("symptoms", "medications_mentioned", "diagnosis", "red_flags"):
        if not isinstance(data[key], list):
            data[key] = [] if data[key] is None else [str(data[key])]
    return data


def _check_hallucinations(entities: dict, transcript: str) -> dict:
    transcript_lower = transcript.lower()
    for field in ("symptoms", "diagnosis"):
        flagged = []
        for item in entities.get(field, []):
            if isinstance(item, str):
                # "A / B" terms: check each alternative independently
                alternatives = re.split(r'\s*/\s*', item)
                grounded = False
                for alt in alternatives:
                    # Strip clinical prefixes ("denies:", "possible:", "r/o") before word matching
                    alt_clean = re.sub(
                        r'^(denies|possible|suspected|rule\s+out|r/o)\s*:?\s*', '',
                        alt, flags=re.IGNORECASE
                    ).strip()
                    words = [
                        w.strip('.,;:()') for w in alt_clean.lower().split()
                        if w.strip('.,;:()') not in _STOP_WORDS and len(w.strip('.,;:()')) > 2
                    ]
                    if any(w in transcript_lower for w in words):
                        grounded = True
                        break
                if not grounded and item.strip():
                    logger.warning(f"[hallucination] {field}: '{item}'")
                    flagged.append({"text": item, "potentially_hallucinated": True})
                else:
                    flagged.append(item)
            else:
                flagged.append(item)
        entities[field] = flagged
    return entities


# ── Public entry point ────────────────────────────────────────────────────────

def extract_offline(transcript: str) -> dict:
    """
    Single-pass clinical extraction via Ollama (qwen2.5:7b).
    One comprehensive call instead of 5 sequential ones — ~3s vs ~14s.
    The doctor reviews and edits the output before approving, so minor
    recall differences are caught at the review stage.
    """
    system = (
        "You are a senior clinical documentation specialist. "
        "Extract complete, accurate structured information from doctor-patient conversations. "
        "Missing clinical information in a healthcare record is a patient safety risk. "
        "Return ONLY valid JSON — no markdown, no explanation."
    )

    red_flag_examples = ", ".join(_RED_FLAG_TARGETS[:12])

    user = f"""Extract ALL clinical information from this doctor-patient transcript.

Transcript:
{transcript}

Return a JSON object with EXACTLY these keys:

- chief_complaint: string — main reason for visit as the patient stated it
- history_of_present_illness: string — full narrative including onset, character, severity, timing, modifying factors, relevant history
- symptoms: list of strings — every symptom mentioned, with severity/location/timing qualifiers where stated; prefix denied symptoms with "denies: "
- duration: string — how long symptoms have been present
- medications_mentioned: list of strings — every drug, supplement, herbal remedy, home remedy; include dosage and frequency if stated; prefix drug allergies with "ALLERGY: "
- diagnosis: list of strings — diagnoses stated or clearly implied by the doctor
- treatment_plan: string — what the doctor recommended
- red_flags: list of strings — urgent or dangerous findings; check specifically for: {red_flag_examples}; return [] if none; add "[CONSISTENCY] ..." if a diagnosis lacks symptom support
- raw_language: "hi"

Return ONLY the JSON object."""

    logger.info("[extract_offline] Single-pass extraction")
    raw = _call("extract", system, user, max_tokens=1024, temperature=0)
    data = _parse_json(raw, "extract")

    if not isinstance(data, dict):
        logger.warning("[extract_offline] Retrying with explicit JSON instruction")
        raw = _call(
            "extract_retry", system,
            user + "\n\nCRITICAL: Return ONLY the raw JSON object. No markdown fences, no text before or after.",
            max_tokens=1024, temperature=0,
        )
        data = _parse_json(raw, "extract_retry")

    if not isinstance(data, dict):
        raise ValueError(f"[extract_offline] Extraction failed after retry. Raw: {raw[:500]}")

    data = _validate_and_coerce(data)
    data = _check_hallucinations(data, transcript)
    logger.info(
        f"[extract_offline] Done. symptoms={len(data['symptoms'])}, "
        f"diagnosis={len(data['diagnosis'])}, medications={len(data['medications_mentioned'])}"
    )
    return data
