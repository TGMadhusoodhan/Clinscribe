"""
ClinScribe clinical entity extraction.
Uses Claude API to extract structured clinical data from Hindi transcripts.
"""

import json
import time
import re
import logging
from datetime import datetime, timezone
from pathlib import Path

import anthropic

import config

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

# Log directory must exist before any LLM call
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LLM_LOG = _LOG_DIR / "llm_calls.jsonl"

# Fixed system prompt — defines the schema and output contract.
# Not a variable because the schema does not change per-request.
_SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. You will be given a Hindi-language "
    "doctor-patient conversation transcript. Extract structured clinical information "
    "and return ONLY valid JSON with no explanation, no markdown, no code fences. "
    "Never invent information not present in the transcript."
)

_REQUIRED_KEYS = {
    "chief_complaint", "history_of_present_illness", "symptoms",
    "duration", "medications_mentioned", "diagnosis",
    "treatment_plan", "red_flags", "raw_language",
}

# Stopwords excluded from hallucination grounding check
_STOP_WORDS = {"the", "a", "an", "of", "in", "with", "has", "have"}


def _build_user_prompt(transcript: str) -> str:
    """
    What it does: Builds the extraction prompt with the actual transcript inserted.
    Inputs: transcript — str Hindi transcript text
    Outputs: str formatted prompt
    Dependencies: None
    Side effects: None
    Failure modes: None
    """
    return f"""Extract clinical information from this Hindi doctor-patient transcript.

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
- raw_language: "hi" (always this value for Hindi input)

Return ONLY the JSON object. No explanation."""


def _log_llm_call(phase: str, model: str, input_tokens: int, output_tokens: int,
                  latency_ms: float, success: bool, error: str | None) -> None:
    """
    What it does: Appends one JSON line to logs/llm_calls.jsonl for cost auditing.
    Inputs: phase — pipeline stage name; model — model ID; input_tokens/output_tokens — ints;
            latency_ms — float ms; success — bool; error — str or None
    Outputs: None
    Dependencies: logs/ directory must exist
    Side effects: Writes one line to logs/llm_calls.jsonl
    Failure modes: OSError if log directory not writable (non-fatal, logs warning)
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
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
    """
    What it does: Verifies all 9 required keys present and list fields are lists (not null).
    Inputs: data — dict parsed from Claude JSON response
    Outputs: coerced dict with all 9 keys guaranteed present
    Dependencies: None
    Side effects: None
    Failure modes: ValueError if required keys missing after coercion attempt
    """
    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    # List fields must be lists — null → empty list
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
                # "A / B" alternatives: check each independently
                alternatives = re.split(r'\s*/\s*', item)
                grounded = False
                for alt in alternatives:
                    # Strip clinical prefixes before word-matching
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
                    logger.warning(f"Possible hallucination in {field}: '{item}'")
                    flagged.append({"text": item, "potentially_hallucinated": True})
                else:
                    flagged.append(item)
            else:
                flagged.append(item)
        entities[field] = flagged
    return entities


def _call_claude(transcript: str, retry_msg: str | None = None) -> tuple[dict, object]:
    """
    What it does: Calls Claude API once, returns (parsed_dict, response_object).
    Inputs: transcript — Hindi transcript; retry_msg — extra instruction appended on retry
    Outputs: (dict of extracted entities, anthropic response object)
    Dependencies: _client (Anthropic SDK); config.ANTHROPIC_MODEL
    Side effects: One API call; logs to llm_calls.jsonl
    Failure modes: anthropic.APIError on network failure; ValueError on bad JSON
    """
    user_content = _build_user_prompt(transcript)
    if retry_msg:
        user_content += f"\n\nIMPORTANT: {retry_msg}"

    t0 = time.monotonic()
    response = _client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1024,
        # Ephemeral cache on the fixed system prompt — saves input tokens and
        # ~200ms on every call after the first within a 5-minute window.
        system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_content}],
    )
    latency_ms = (time.monotonic() - t0) * 1000

    raw = response.content[0].text.strip()
    # Strip markdown fences if model includes them despite instructions
    raw = re.sub(r"^```(?:json)?", "", raw).rstrip("```").strip()

    try:
        data = json.loads(raw)
        _log_llm_call("extraction", config.ANTHROPIC_MODEL,
                      response.usage.input_tokens, response.usage.output_tokens,
                      latency_ms, True, None)
        return data, response
    except json.JSONDecodeError as e:
        _log_llm_call("extraction", config.ANTHROPIC_MODEL,
                      response.usage.input_tokens, response.usage.output_tokens,
                      latency_ms, False, str(e))
        raise ValueError(f"JSON parse failed. Raw response:\n{raw}") from e


def extract(transcript: str) -> dict:
    """
    What it does: Extracts structured clinical entities from a Hindi transcript using Claude.
    Inputs: transcript — str Hindi doctor-patient conversation text
    Outputs: dict with keys: chief_complaint, history_of_present_illness, symptoms,
             duration, medications_mentioned, diagnosis, treatment_plan, red_flags, raw_language
    Dependencies: Anthropic API; config.py; logs/ directory
    Side effects: Appends one line to logs/llm_calls.jsonl
    Failure modes: ValueError if Claude returns malformed JSON twice in a row;
                   anthropic.APIError on network/auth failure
    """
    try:
        data, _ = _call_claude(transcript)
    except ValueError:
        # Retry once with explicit raw-JSON instruction
        try:
            data, _ = _call_claude(transcript, retry_msg="Return only raw JSON, no markdown fences, no explanation.")
        except ValueError as e:
            # Log full raw response in ERROR_LOG — do not silently swallow
            error_entry = (
                f"\n## ERROR — {datetime.now(timezone.utc).isoformat()}\n"
                f"**Step:** Step 7 — extract.py\n"
                f"**What I was doing:** Extracting entities from transcript\n"
                f"**Error type:** ValueError (JSON parse failure after 2 attempts)\n"
                f"**Full error message:**\n```\n{e}\n```\n"
                f"**Resolution:** BLOCKED — needs investigation\n"
                f"**Impact on project:** extract() raises ValueError to caller\n---\n"
            )
            error_log = Path(__file__).parent.parent / "ERROR_LOG.md"
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(error_entry)
            raise

    data = _validate_and_coerce(data)
    data = _check_hallucinations(data, transcript)
    return data
