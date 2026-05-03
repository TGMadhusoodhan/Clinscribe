"""
ClinScribe offline translation via Ollama (qwen2.5:7b).

Replaces NLLB-200 which produced literal, context-free translations.
qwen2.5:7b already runs for entity extraction — no extra memory cost.
All segments are batched into one call so the model has full context
(e.g. "कल" = yesterday vs tomorrow, speaker turns, medical intent).

Glossary pre-substitution (data/hindi_medical_glossary.csv) runs before
the Ollama call to handle Devanagari-transliterated English that even a 7B
model doesn't recognise (e.g. "ओरस" → "ORS", "फूट पॉइस्निंग" → "food poisoning").
"""

import csv
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL  = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434/v1")
_OLLAMA_MODEL     = getattr(config, "OLLAMA_MODEL",    "qwen2.5:7b")
_GLOSSARY_PATH    = Path(__file__).parent.parent / "data" / "hindi_medical_glossary.csv"

# Module-level singleton — avoids re-creating connection pool on every translation call
from openai import OpenAI as _OpenAI
_ollama_client = _OpenAI(base_url=_OLLAMA_BASE_URL, api_key="ollama")


@lru_cache(maxsize=1)
def _load_glossary() -> list[tuple[str, str]]:
    """Loads hindi_medical_glossary.csv sorted longest-first for greedy matching."""
    if not _GLOSSARY_PATH.exists():
        logger.warning(f"[translate_offline] Glossary not found at {_GLOSSARY_PATH}")
        return []
    pairs: list[tuple[str, str]] = []
    with open(_GLOSSARY_PATH, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            h = row.get("hindi", "").strip()
            e = row.get("english", "").strip()
            if h and e:
                pairs.append((h, e))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def _apply_glossary(text: str) -> str:
    """Replaces Devanagari-transliterated English and known mis-translated terms
    with their English equivalents so Ollama gets unambiguous input."""
    for hindi, english in _load_glossary():
        text = text.replace(hindi, english)
    return text


def _parse_json_array(raw: str) -> list[str]:
    """Extract a JSON array of strings from model output, tolerating markdown fences."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [str(x) for x in result]
    except json.JSONDecodeError:
        pass
    # Fallback: find first [...] block in the response
    m = re.search(r"\[.*?\]", raw, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass
    return []


def translate_segments_offline(segments: list) -> list:
    """
    Translates Hindi segments to English using Ollama (qwen2.5:7b).
    All segments are batched into one call — the model sees the full
    conversation context so it resolves ambiguous words correctly
    (e.g. "कल" = yesterday, speaker identity, clinical intent).
    Falls back to original segment text if the model call fails.
    """
    if not segments:
        return segments

    # Apply glossary first — converts Devanagari-transliterated English ("ओरस" → "ORS")
    # that even a 7B model won't recognise from context alone.
    numbered = "\n".join(
        f"[{i}] {_apply_glossary(seg['text'])}" for i, seg in enumerate(segments)
    )

    prompt = (
        "You are a medical interpreter. Translate each numbered Hindi segment into "
        "natural, fluent English. Use clinical terminology where appropriate. "
        "Preserve the original meaning and speaker intent — do not paraphrase or "
        "add information. Return ONLY a JSON array of strings, one translation per "
        "segment, in the same order as the input. No explanation, no markdown.\n\n"
        f"{numbered}"
    )

    try:
        response = _ollama_client.chat.completions.create(
            model=_OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096,  # raised from 1024 — long recordings with many segments need headroom
        )
        raw = response.choices[0].message.content or ""
        translations = _parse_json_array(raw)

        for i, seg in enumerate(segments):
            if i < len(translations) and translations[i].strip():
                seg["english_translation"] = translations[i].strip()
            else:
                seg["english_translation"] = seg["text"]

    except Exception as e:
        logger.warning(f"[translate_offline] Ollama call failed: {e} — using original text")
        for seg in segments:
            seg.setdefault("english_translation", seg["text"])

    return segments
