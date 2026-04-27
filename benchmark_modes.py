"""
ClinScribe — Online vs Offline Mode Benchmark (Hindi)

Compares the full pipeline on the same 6 Hindi clinical audio clips:
  Online:  Whisper → Claude translation → Claude extraction
  Offline: Whisper → Helsinki-NLP/opus-mt-hi-en → Qwen via vLLM

Metrics:
  - Translation latency
  - Extraction latency
  - Entity completeness (fields filled / total fields)
  - Entity count per field (symptoms, diagnoses, medications)
  - Side-by-side translation output

Requirements:
  - Online:  ANTHROPIC_API_KEY in .env
  - Offline: vLLM server running at VLLM_BASE_URL (VLLM_MODEL loaded)
  - Audio clips in benchmark_audio/ (run Benchmarking.py first if missing)

Output:
  - benchmark_modes_report.txt   human-readable comparison
  - benchmark_modes_results.json full data
"""

import json
import time
import os
import sys
from pathlib import Path

# ── Load env before any pipeline imports ─────────────────────────────────────
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import config

# ── Corpus (same clips as Benchmarking.py) ───────────────────────────────────

CORPUS = [
    {
        "id": 1,
        "scenario": "Fever and headache",
        "audio": "benchmark_audio/hindi_1.mp3",
    },
    {
        "id": 2,
        "scenario": "Chest pain and breathlessness",
        "audio": "benchmark_audio/hindi_2.mp3",
    },
    {
        "id": 3,
        "scenario": "Diabetes follow-up",
        "audio": "benchmark_audio/hindi_3.mp3",
    },
    {
        "id": 4,
        "scenario": "Cough and TB concern",
        "audio": "benchmark_audio/hindi_4.mp3",
    },
    {
        "id": 5,
        "scenario": "Stomach pain and vomiting",
        "audio": "benchmark_audio/hindi_5.mp3",
    },
    {
        "id": 6,
        "scenario": "Joint pain",
        "audio": "benchmark_audio/hindi_6.mp3",
    },
]

# Fields that must be non-empty for an extraction to be "complete"
_REQUIRED_FIELDS = [
    "chief_complaint",
    "history_of_present_illness",
    "symptoms",
    "duration",
    "medications_mentioned",
    "diagnosis",
    "treatment_plan",
    "red_flags",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _field_filled(value) -> bool:
    """Returns True if a field has meaningful content."""
    if value is None:
        return False
    if isinstance(value, list):
        return len([v for v in value if v and str(v).strip()]) > 0
    return bool(str(value).strip())


def _completeness(entities: dict) -> float:
    """Returns fraction of required fields that are filled (0.0–1.0)."""
    filled = sum(1 for f in _REQUIRED_FIELDS if _field_filled(entities.get(f)))
    return round(filled / len(_REQUIRED_FIELDS), 2)


def _entity_counts(entities: dict) -> dict:
    """Returns counts for list fields."""
    return {
        "symptoms":     len([s for s in entities.get("symptoms", []) if s]),
        "diagnoses":    len([d for d in entities.get("diagnosis", []) if d]),
        "medications":  len([m for m in entities.get("medications_mentioned", []) if m]),
        "red_flags":    len([r for r in entities.get("red_flags", []) if r]),
    }


def _convert_to_wav(mp3_path: str) -> str:
    """Convert MP3 to WAV for Whisper (pydub + ffmpeg)."""
    import tempfile
    from pydub import AudioSegment

    with open(mp3_path, "rb") as f:
        raw = f.read()

    seg = AudioSegment.from_file(__import__("io").BytesIO(raw), format="mp3")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    seg.export(tmp.name, format="wav")
    return tmp.name


# ── Transcription (shared — same for both modes) ──────────────────────────────

def run_transcription(audio_path: str) -> dict:
    """Run Whisper transcription. Returns transcript_result dict."""
    from pipeline.transcribe import transcribe
    wav_path = _convert_to_wav(audio_path)
    try:
        t0 = time.monotonic()
        result = transcribe(wav_path)
        result["latency_seconds"] = round(time.monotonic() - t0, 2)
        return result
    finally:
        os.unlink(wav_path)


# ── Online pipeline ───────────────────────────────────────────────────────────

def run_online(transcript_result: dict) -> dict:
    """
    Online mode: Claude translation + Claude extraction.
    Returns timing, translation text, entities, completeness.
    """
    from pipeline.transcribe import translate_segments
    from pipeline.extract import extract

    segments = list(transcript_result["segments"])

    t0 = time.monotonic()
    translated = translate_segments(segments)
    translate_latency = round(time.monotonic() - t0, 2)

    english_text = " ".join(
        s.get("english_translation", s.get("text", "")) for s in translated
    )

    t0 = time.monotonic()
    entities = extract(english_text)
    extract_latency = round(time.monotonic() - t0, 2)

    return {
        "mode": "online",
        "english_text": english_text,
        "translate_latency_s": translate_latency,
        "extract_latency_s": extract_latency,
        "total_latency_s": round(translate_latency + extract_latency, 2),
        "entities": entities,
        "completeness": _completeness(entities),
        "entity_counts": _entity_counts(entities),
    }


# ── Offline pipeline ──────────────────────────────────────────────────────────

def run_offline(transcript_result: dict) -> dict:
    """
    Offline mode: Helsinki-NLP/opus-mt-hi-en translation + Qwen via vLLM extraction.
    Returns timing, translation text, entities, completeness.
    """
    from pipeline.translate_offline import translate_segments_offline
    from pipeline.extract_offline import extract_offline

    segments = list(transcript_result["segments"])

    t0 = time.monotonic()
    translated = translate_segments_offline(segments)
    translate_latency = round(time.monotonic() - t0, 2)

    english_text = " ".join(
        s.get("english_translation", s.get("text", "")) for s in translated
    )

    t0 = time.monotonic()
    entities = extract_offline(english_text)
    extract_latency = round(time.monotonic() - t0, 2)

    return {
        "mode": "offline",
        "english_text": english_text,
        "translate_latency_s": translate_latency,
        "extract_latency_s": extract_latency,
        "total_latency_s": round(translate_latency + extract_latency, 2),
        "entities": entities,
        "completeness": _completeness(entities),
        "entity_counts": _entity_counts(entities),
    }


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(run_online_mode: bool = True, run_offline_mode: bool = True):
    print("=" * 65)
    print("ClinScribe — Online vs Offline Mode Benchmark (Hindi)")
    print("=" * 65)

    if run_online_mode and not config.ANTHROPIC_API_KEY:
        print("WARNING: ANTHROPIC_API_KEY not set — skipping online mode.")
        run_online_mode = False

    # Load Whisper once
    print("\nLoading Whisper large-v3...")
    from pipeline.transcribe import _whisper_model
    print("Whisper loaded.\n")

    all_results = []

    online_completeness = []
    offline_completeness = []
    online_translate_lat = []
    offline_translate_lat = []
    online_extract_lat = []
    offline_extract_lat = []

    for item in CORPUS:
        audio_path = item["audio"]
        if not Path(audio_path).exists():
            print(f"SKIP: {audio_path} not found. Run Benchmarking.py first.")
            continue

        print(f"\n{'─'*65}")
        print(f"Clip {item['id']}: {item['scenario']}")
        print(f"{'─'*65}")

        # Transcription (shared)
        print("  [Whisper] Transcribing...", end=" ", flush=True)
        t0 = time.monotonic()
        transcript = run_transcription(audio_path)
        whisper_lat = round(time.monotonic() - t0, 2)
        print(f"{whisper_lat}s")
        print(f"  Hindi: {transcript['full_text'][:100]}...")

        clip_result = {
            "id": item["id"],
            "scenario": item["scenario"],
            "whisper_latency_s": whisper_lat,
            "hindi_text": transcript["full_text"],
            "online": None,
            "offline": None,
        }

        # Online mode
        if run_online_mode:
            print("  [Online]  Translating (Claude)...", end=" ", flush=True)
            try:
                online = run_online(transcript)
                print(f"translate={online['translate_latency_s']}s  |  "
                      f"extract={online['extract_latency_s']}s  |  "
                      f"completeness={int(online['completeness']*100)}%")
                clip_result["online"] = online
                online_completeness.append(online["completeness"])
                online_translate_lat.append(online["translate_latency_s"])
                online_extract_lat.append(online["extract_latency_s"])
            except Exception as e:
                print(f"FAILED: {e}")
                clip_result["online"] = {"error": str(e)}

        # Offline mode
        if run_offline_mode:
            print("  [Offline] Translating (MarianMT)...", end=" ", flush=True)
            try:
                offline = run_offline(transcript)
                print(f"translate={offline['translate_latency_s']}s  |  "
                      f"extract={offline['extract_latency_s']}s  |  "
                      f"completeness={int(offline['completeness']*100)}%")
                clip_result["offline"] = offline
                offline_completeness.append(offline["completeness"])
                offline_translate_lat.append(offline["translate_latency_s"])
                offline_extract_lat.append(offline["extract_latency_s"])
            except Exception as e:
                print(f"FAILED: {e}")
                clip_result["offline"] = {"error": str(e)}

        all_results.append(clip_result)

    # ── Build report ──────────────────────────────────────────────────────────

    def avg(lst): return round(sum(lst) / len(lst), 2) if lst else 0

    summary = {
        "clips": len(all_results),
        "online": {
            "avg_completeness_pct": round(avg(online_completeness) * 100, 1),
            "avg_translate_latency_s": avg(online_translate_lat),
            "avg_extract_latency_s": avg(online_extract_lat),
            "avg_total_latency_s": round(avg(online_translate_lat) + avg(online_extract_lat), 2),
            "translation_model": config.ANTHROPIC_MODEL,
            "extraction_model": config.ANTHROPIC_MODEL,
        } if run_online_mode else None,
        "offline": {
            "avg_completeness_pct": round(avg(offline_completeness) * 100, 1),
            "avg_translate_latency_s": avg(offline_translate_lat),
            "avg_extract_latency_s": avg(offline_extract_lat),
            "avg_total_latency_s": round(avg(offline_translate_lat) + avg(offline_extract_lat), 2),
            "translation_model": "Helsinki-NLP/opus-mt-hi-en",
            "extraction_model": config.VLLM_MODEL,
        } if run_offline_mode else None,
        "clips_detail": all_results,
    }

    # ── Print report ──────────────────────────────────────────────────────────

    lines = []
    lines.append("\n" + "=" * 65)
    lines.append("RESULTS: Online (Claude) vs Offline (MarianMT + Qwen)")
    lines.append("=" * 65)

    # Per-clip table
    header = f"{'Scenario':<32} {'Online':>14} {'Offline':>14}"
    lines.append(f"\n{'Entity Completeness (% of fields filled)'}")
    lines.append(header)
    lines.append("-" * 62)

    for r in all_results:
        o_val = f"{int(r['online']['completeness']*100)}%" if r.get("online") and "completeness" in r["online"] else "ERROR"
        f_val = f"{int(r['offline']['completeness']*100)}%" if r.get("offline") and "completeness" in r["offline"] else "ERROR"
        lines.append(f"{r['scenario']:<32} {o_val:>14} {f_val:>14}")

    if online_completeness or offline_completeness:
        lines.append("-" * 62)
        o_avg = f"{round(avg(online_completeness)*100,1)}%" if online_completeness else "N/A"
        f_avg = f"{round(avg(offline_completeness)*100,1)}%" if offline_completeness else "N/A"
        lines.append(f"{'AVERAGE':<32} {o_avg:>14} {f_avg:>14}")

    # Latency table
    lines.append(f"\n{'Latency (seconds)'}")
    lines.append(f"{'Stage':<32} {'Online':>14} {'Offline':>14}")
    lines.append("-" * 62)
    lines.append(f"{'Translation':<32} {avg(online_translate_lat):>13.1f}s {avg(offline_translate_lat):>13.1f}s")
    lines.append(f"{'Extraction':<32} {avg(online_extract_lat):>13.1f}s {avg(offline_extract_lat):>13.1f}s")
    lines.append(f"{'Total (translate + extract)':<32} {avg(online_translate_lat)+avg(online_extract_lat):>13.1f}s {avg(offline_translate_lat)+avg(offline_extract_lat):>13.1f}s")

    # Models used
    lines.append(f"\n{'Models'}")
    lines.append("-" * 62)
    lines.append(f"  Online  translation : {config.ANTHROPIC_MODEL}")
    lines.append(f"  Online  extraction  : {config.ANTHROPIC_MODEL}")
    lines.append(f"  Offline translation : Helsinki-NLP/opus-mt-hi-en")
    lines.append(f"  Offline extraction  : {config.VLLM_MODEL}")

    # Side-by-side translations
    lines.append(f"\n{'=' * 65}")
    lines.append("TRANSLATION COMPARISON (Hindi → English)")
    lines.append("=" * 65)

    for r in all_results:
        lines.append(f"\nClip {r['id']}: {r['scenario']}")
        lines.append(f"  Hindi   : {r['hindi_text'][:120]}")
        if r.get("online") and "english_text" in r["online"]:
            lines.append(f"  Online  : {r['online']['english_text'][:120]}")
        if r.get("offline") and "english_text" in r["offline"]:
            lines.append(f"  Offline : {r['offline']['english_text'][:120]}")

    # Entity comparison
    lines.append(f"\n{'=' * 65}")
    lines.append("ENTITY EXTRACTION COMPARISON")
    lines.append("=" * 65)

    for r in all_results:
        lines.append(f"\nClip {r['id']}: {r['scenario']}")
        for mode_key, label in [("online", "Online "), ("offline", "Offline")]:
            m = r.get(mode_key)
            if not m or "error" in m:
                lines.append(f"  {label}: ERROR")
                continue
            e = m["entities"]
            lines.append(f"  {label} | complaint  : {str(e.get('chief_complaint',''))[:80]}")
            lines.append(f"  {label} | symptoms   : {e.get('symptoms', [])}")
            lines.append(f"  {label} | diagnosis  : {e.get('diagnosis', [])}")
            lines.append(f"  {label} | medications: {e.get('medications_mentioned', [])}")
            lines.append(f"  {label} | duration   : {e.get('duration','')}")
            lines.append(f"  {label} | treatment  : {str(e.get('treatment_plan',''))[:80]}")
            lines.append(f"  {label} | red_flags  : {e.get('red_flags', [])}")

    lines.append(f"\n{'=' * 65}")
    lines.append("NOTES")
    lines.append("=" * 65)
    lines.append("Completeness = fraction of 8 required fields that are non-empty.")
    lines.append("Latency does not include Whisper (identical for both modes).")
    lines.append("Offline mode requires vLLM server running with VLLM_MODEL loaded.")
    lines.append("Online mode quality reflects Claude's Hindi medical knowledge.")

    report = "\n".join(lines)
    print(report)

    with open("benchmark_modes_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    with open("benchmark_modes_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nSaved:")
    print(f"  benchmark_modes_report.txt")
    print(f"  benchmark_modes_results.json")

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pass --online-only or --offline-only to skip one mode
    online  = "--offline-only" not in sys.argv
    offline = "--online-only"  not in sys.argv
    run_benchmark(run_online_mode=online, run_offline_mode=offline)
