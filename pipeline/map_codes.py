"""
ClinScribe terminology mapping.

ICD-10: local lookup via simple-icd-10-cm + rapidfuzz (works offline, no API needed).
SNOMED-CT: public browser API (online only — no free offline dataset without UMLS registration).

For SNOMED offline, register free at https://www.nlm.nih.gov/research/umls/ to get
an API key, then replace _fetch_snomed with a UMLS API call.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

# Result cache — same diagnosis string never hits ICD-10/SNOMED twice per process lifetime
_map_cache: dict[str, dict] = {}
_map_cache_lock = threading.Lock()

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_UNMAPPED_LOG = _LOG_DIR / "unmapped_terms.txt"

logger = logging.getLogger(__name__)

_SNOMED_BASE = (
    "https://browser.ihtsdotools.org/snowstorm/snomed-ct/browser/MAIN/descriptions"
)
_TIMEOUT = 10

# ── Local ICD-10 index (built once, lazily, on first use) ─────────────────────

_icd10_index: dict[str, tuple[str, str]] = {}  # lower_desc → (code, display_desc)
_icd10_ready = False
_icd10_build_lock = threading.Lock()
_log_lock = threading.Lock()


def _ensure_icd10_index() -> None:
    global _icd10_ready, _icd10_index
    if _icd10_ready:
        return
    with _icd10_build_lock:
        if _icd10_ready:
            return
        try:
            import simple_icd_10_cm as cm
            for code in cm.get_all_codes(with_dots=True):
                desc = cm.get_description(code)
                if desc:
                    _icd10_index[desc.lower()] = (code, desc)
            logger.info(f"[map_codes] ICD-10 local index ready: {len(_icd10_index)} codes")
        except ImportError:
            logger.warning(
                "[map_codes] simple-icd-10-cm not installed — falling back to ICD-10 API. "
                "Run: pip install simple-icd-10-cm rapidfuzz"
            )
        except Exception as e:
            logger.error(f"[map_codes] ICD-10 index build failed: {e} — falling back to API")
        finally:
            _icd10_ready = True


# Pre-warm in a background thread at import time so the first real request
# doesn't block for 2-3s building the index
threading.Thread(target=_ensure_icd10_index, daemon=True, name="icd10-warmup").start()


def _fetch_icd10_local(term: str) -> dict | None:
    """Fuzzy-matches term against local ICD-10-CM index. Returns None if no confident match."""
    _ensure_icd10_index()
    if not _icd10_index:
        return None
    try:
        from rapidfuzz import fuzz, process
        match = process.extractOne(
            term.lower(),
            list(_icd10_index.keys()),
            scorer=fuzz.WRatio,
            score_cutoff=70,
        )
        if match:
            key, score, _ = match
            code, desc = _icd10_index[key]
            logger.info(f"[map_codes] ICD-10 local: '{term}' → {code} ({desc}) [score={score:.0f}]")
            return {"code": code, "description": desc}
    except ImportError:
        logger.warning("[map_codes] rapidfuzz not installed — pip install rapidfuzz")
    return None


# ── SNOMED-CT (API only) ──────────────────────────────────────────────────────

def _fetch_snomed(term: str) -> dict | None:
    """
    Queries NLM SNOMED CT Browser API. Online only.
    Returns {"conceptId": str, "term": str} or None.
    """
    params = {"term": term, "active": "true", "limit": 3}
    for attempt in range(2):
        try:
            resp = requests.get(_SNOMED_BASE, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if items:
                first = items[0]
                return {"conceptId": first["concept"]["conceptId"], "term": first["term"]}
            return None
        except requests.Timeout:
            if attempt == 0:
                time.sleep(1)
                continue
            return None
        except (requests.RequestException, ValueError, KeyError):
            return None
    return None


# ── ICD-10 API fallback (used only when local index unavailable) ──────────────

_ICD10_API_BASE = "https://icd10api.com/"


def _fetch_icd10_api(term: str) -> dict | None:
    """Fallback ICD-10 lookup via icd10api.com when local index is not installed."""
    params = {"s": term, "desc": "short", "r": "json"}
    for attempt in range(2):
        try:
            resp = requests.get(_ICD10_API_BASE, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if data.get("Response") == "True" and data.get("Notes"):
                first = data["Notes"][0]
                return {"code": first.get("ICD10Code", ""), "description": first.get("Description", "")}
            return None
        except requests.Timeout:
            if attempt == 0:
                time.sleep(1)
                continue
            return None
        except (requests.RequestException, ValueError, KeyError):
            return None
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _confidence(term: str, icd10: dict | None, snomed: dict | None) -> str:
    if icd10 is None and snomed is None:
        return "unmapped"
    term_lower = term.lower()
    for result in (icd10, snomed):
        if result:
            desc = (result.get("description") or result.get("term") or "").lower()
            if term_lower in desc or desc in term_lower:
                return "high"
    return "low"


def _log_unmapped(term: str, icd10_error: str, snomed_error: str) -> None:
    entry = (
        f"{datetime.now(timezone.utc).isoformat()} | {term} "
        f"| icd10: {icd10_error} | snomed: {snomed_error}\n"
    )
    try:
        with _log_lock, open(_UNMAPPED_LOG, "a", encoding="utf-8") as f:
            f.write(entry)
    except OSError as e:
        logger.warning(f"Could not write unmapped log: {e}")


def _map_one(clean_term: str) -> dict:
    with _map_cache_lock:
        if clean_term in _map_cache:
            return _map_cache[clean_term]

    # ICD-10: local index first, API only if index unavailable
    icd10 = _fetch_icd10_local(clean_term)
    if icd10 is None:
        icd10 = _fetch_icd10_api(clean_term)

    # SNOMED: API only (runs in parallel with ICD-10 API fallback if needed)
    snomed = _fetch_snomed(clean_term)

    confidence = _confidence(clean_term, icd10, snomed)

    notes_parts = []
    if icd10:
        notes_parts.append(f"ICD-10: {icd10['code']} ({icd10['description']})")
    else:
        notes_parts.append("ICD-10: no result")
        _log_unmapped(clean_term, "no result", "no result" if snomed is None else "ok")

    if snomed:
        notes_parts.append(f"SNOMED: {snomed['conceptId']} ({snomed['term']})")
    else:
        notes_parts.append("SNOMED: no result")
        if icd10 is not None:
            _log_unmapped(clean_term, "ok", "no result")

    result = {
        "term": clean_term,
        "icd10": icd10,
        "snomed": snomed,
        "confidence": confidence,
        "notes": "; ".join(notes_parts),
    }
    with _map_cache_lock:
        _map_cache[clean_term] = result
    return result


# ── Public entry point ────────────────────────────────────────────────────────

def map_codes(diagnoses: list) -> list:
    """
    Maps clinical terms to ICD-10 (local) and SNOMED-CT (API) codes.
    All terms are processed concurrently; output order matches input order.
    """
    clean_terms = [t.strip() for t in diagnoses if isinstance(t, str) and t.strip()]
    if not clean_terms:
        return []

    with ThreadPoolExecutor(max_workers=min(len(clean_terms), 6)) as ex:
        futures = [ex.submit(_map_one, term) for term in clean_terms]
        return [f.result() for f in futures]
