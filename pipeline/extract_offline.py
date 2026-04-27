"""
ClinScribe offline entity extraction via vLLM (Qwen2.5-7B-Instruct).
Replaces Claude extract() for offline/rural deployments.
Requires vLLM server running at VLLM_BASE_URL (default: http://localhost:8001/v1).
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_VLLM_BASE_URL = getattr(config, "VLLM_BASE_URL", "http://localhost:8001/v1")
_VLLM_MODEL = getattr(config, "VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LLM_LOG = _LOG_DIR / "llm_calls.jsonl"

_SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. You will be given an English-language "
    "doctor-patient conversation transcript (translated from Hindi). Extract structured "
    "clinical information and return ONLY valid JSON with no explanation, no markdown, "
    "no code fences. Never invent information not present in the transcript."
)

_REQUIRED_KEYS = {
    "chief_complaint", "history_of_present_illness", "symptoms",
    "duration", "medications_mentioned", "diagnosis",
    "treatment_plan", "red_flags", "raw_language",
}

_STOP_WORDS = {"the", "a", "an", "of", "in", "with", "has", "have"}


def _build_user_prompt(transcript: str) -> str:
    return f"""Extract clinical information from this doctor-patient transcript.

Transcript:
{transcript}

Return a JSON object with exactly these keys:
- chief_complaint: string (main reason for visit)
- history_of_present_illness: string (narrative of the complaint)
- symptoms: list of strings
- duration: string (how long symptoms present)
- medications_mentioned: list of strings
- diagnosis: list of strings
- treatment_plan: string
- red_flags: list of strings (urgent/dangerous findings)
- raw_language: "hi" (always this value for Hindi-origin input)

Return ONLY the JSON object. No explanation."""


def _get_client():
    from openai import OpenAI
    return OpenAI(base_url=_VLLM_BASE_URL, api_key="dummy")


def _log_llm_call(model: str, latency_ms: float, success: bool, error: str | None) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "extraction_offline",
        "model": model,
        "latency_ms": round(latency_ms, 1),
        "success": success,
        "error": error,
    }
    try:
        with open(_LLM_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.warning(f"Could not write LLM log: {e}")


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
                words = [w for w in item.lower().split() if w not in _STOP_WORDS]
                grounded = any(w in transcript_lower for w in words if len(w) > 2)
                if not grounded and words:
                    logger.warning(f"Possible hallucination in {field}: '{item}'")
                    flagged.append({"text": item, "potentially_hallucinated": True})
                else:
                    flagged.append(item)
            else:
                flagged.append(item)
        entities[field] = flagged
    return entities


def _call_vllm(transcript: str, retry_msg: str | None = None) -> dict:
    client = _get_client()
    user_content = _build_user_prompt(transcript)
    if retry_msg:
        user_content += f"\n\nIMPORTANT: {retry_msg}"

    t0 = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=_VLLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000
        _log_llm_call(_VLLM_MODEL, latency_ms, False, str(e))
        raise

    raw = re.sub(r"^```(?:json)?", "", raw).rstrip("```").strip()

    try:
        data = json.loads(raw)
        _log_llm_call(_VLLM_MODEL, latency_ms, True, None)
        return data
    except json.JSONDecodeError as e:
        _log_llm_call(_VLLM_MODEL, latency_ms, False, str(e))
        raise ValueError(f"JSON parse failed. Raw response:\n{raw}") from e


def extract_offline(transcript: str) -> dict:
    """
    Extracts structured clinical entities using a locally-running vLLM server.
    Drop-in replacement for extract.extract() in offline mode.
    Requires: vLLM running at VLLM_BASE_URL serving VLLM_MODEL.
    """
    try:
        data = _call_vllm(transcript)
    except ValueError:
        data = _call_vllm(transcript, retry_msg="Return only raw JSON, no markdown fences, no explanation.")

    data = _validate_and_coerce(data)
    data = _check_hallucinations(data, transcript)
    return data
