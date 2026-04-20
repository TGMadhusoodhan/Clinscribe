"""
ClinScribe terminology mapping.
Fetches SNOMED-CT and ICD-10 codes from real public APIs at runtime.
No codes are hardcoded — hardcoded codes in patient records are a clinical error risk.
"""

import urllib.parse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_UNMAPPED_LOG = _LOG_DIR / "unmapped_terms.txt"

logger = logging.getLogger(__name__)

_ICD10_BASE = "https://icd10api.com/"
_SNOMED_BASE = (
    "https://browser.ihtsdotools.org/snowstorm/snomed-ct/browser/MAIN/descriptions"
)
_TIMEOUT = 10  # seconds before retry


def _fetch_icd10(term: str) -> dict | None:
    """
    What it does: Queries icd10api.com for the best-matching ICD-10 code.
    Inputs: term — str clinical term in English
    Outputs: dict {"code": str, "description": str} or None on failure/empty
    Dependencies: icd10api.com (public, no auth)
    Side effects: HTTP GET; one retry on timeout
    Failure modes: returns None on empty results, HTTP error, or two timeouts
    """
    params = {"s": term, "desc": "short", "r": "json"}
    for attempt in range(2):
        try:
            resp = requests.get(_ICD10_BASE, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            # API returns {"Response": "True", "Notes": [...]} or {"Response": "False"}
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


def _fetch_snomed(term: str) -> dict | None:
    """
    What it does: Queries NLM SNOMED CT Browser API for the best-matching concept.
    Inputs: term — str clinical term in English
    Outputs: dict {"conceptId": str, "term": str} or None on failure/empty
    Dependencies: browser.ihtsdotools.org (public, no auth)
    Side effects: HTTP GET; one retry on timeout
    Failure modes: returns None on empty results, HTTP error, or two timeouts
    """
    params = {
        "term": term,
        "active": "true",
        "limit": 3,
    }
    for attempt in range(2):
        try:
            resp = requests.get(_SNOMED_BASE, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            if items:
                first = items[0]
                return {
                    "conceptId": first["concept"]["conceptId"],
                    "term": first["term"],
                }
            return None
        except requests.Timeout:
            if attempt == 0:
                time.sleep(1)
                continue
            return None
        except (requests.RequestException, ValueError, KeyError):
            return None
    return None


def _confidence(term: str, icd10: dict | None, snomed: dict | None) -> str:
    """
    What it does: Scores mapping quality based on text overlap between term and returned descriptions.
    Inputs: term — original search term; icd10/snomed — mapping results or None
    Outputs: "high" | "low" | "unmapped"
    Dependencies: None
    Side effects: None
    Failure modes: None
    """
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
    """
    What it does: Appends an unmapped term entry to logs/unmapped_terms.txt.
    Inputs: term — str; icd10_error/snomed_error — str error descriptions
    Outputs: None
    Dependencies: logs/ directory exists
    Side effects: Appends one line to logs/unmapped_terms.txt
    Failure modes: OSError if not writable (logs warning, non-fatal)
    """
    entry = f"{datetime.now(timezone.utc).isoformat()} | {term} | icd10: {icd10_error} | snomed: {snomed_error}\n"
    try:
        with open(_UNMAPPED_LOG, "a", encoding="utf-8") as f:
            f.write(entry)
    except OSError as e:
        logger.warning(f"Could not write unmapped log: {e}")


def map_codes(diagnoses: list) -> list:
    """
    What it does: Maps a list of clinical terms to SNOMED-CT and ICD-10 codes via public APIs.
    Inputs: diagnoses — list of str clinical terms (English or transliterated Hindi)
    Outputs: list of dicts each with: term, icd10, snomed, confidence, notes
    Dependencies: icd10api.com; browser.ihtsdotools.org; logs/ directory
    Side effects: Writes unmapped terms to logs/unmapped_terms.txt
    Failure modes: Individual term failures return {"confidence": "unmapped"}; never raises
    """
    results = []
    for term in diagnoses:
        if not isinstance(term, str) or not term.strip():
            continue

        clean_term = term.strip()
        icd10 = _fetch_icd10(clean_term)
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
                # Only log unmapped if not already logged above
                _log_unmapped(clean_term, "ok", "no result")

        results.append({
            "term": clean_term,
            "icd10": icd10,
            "snomed": snomed,
            "confidence": confidence,
            "notes": "; ".join(notes_parts),
        })

    return results
