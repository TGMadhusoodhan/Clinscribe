"""
Benchmark: Hindi→English translation quality and speed.

Compares three translators on the same clinical Hindi segments:
  1. MarianMT (Helsinki-NLP/opus-mt-hi-en) — "LangTrans" / previous offline approach
  2. Qwen2.5-7b via Ollama                  — current offline approach
  3. Claude API (claude-sonnet-4-6)          — current online approach

Requires:
  - benchmark_audio/ directory with the 6 Hindi MP3 clips
  - Ollama running locally  (for Qwen)
  - ANTHROPIC_API_KEY in .env  (for Claude)
  - pip install transformers sentencepiece  (for MarianMT)

Usage:
  python benchmark_translation_compare.py

Output: benchmark_translation_compare_report.txt
"""

import json
import os
import re
import sys
import time
import tempfile
import warnings
from pathlib import Path
from functools import lru_cache

warnings.filterwarnings("ignore")

# ── Load .env ──────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

import config

# ── Clips (same set as whisper benchmark) ─────────────────────────────────────
CLIPS = [
    ("hindi_1.mp3", "Fever and headache"),
    ("hindi_2.mp3", "Chest pain and breathlessness"),
    ("hindi_3.mp3", "Diabetes follow-up"),
    ("hindi_4.mp3", "Cough and TB concern"),
    ("hindi_5.mp3", "Stomach pain and vomiting"),
    ("hindi_6.mp3", "Joint pain"),
]
AUDIO_DIR  = Path("benchmark_audio")
REPORT_OUT = Path("benchmark_translation_compare_report.txt")

# ── Step 0: Transcribe all clips once via faster-whisper ──────────────────────

def _load_whisper():
    from faster_whisper import WhisperModel
    print(f"[whisper] Loading {config.WHISPER_MODEL} on CPU (int8)...", flush=True)
    t0 = time.monotonic()
    model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    print(f"[whisper] Loaded in {time.monotonic()-t0:.1f}s", flush=True)
    return model


def _mp3_to_wav(path: str) -> str:
    from pydub import AudioSegment
    seg = AudioSegment.from_file(path)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    seg.export(tmp.name, format="wav")
    return tmp.name


def transcribe_clip(whisper_model, audio_path: str) -> list[dict]:
    """Returns list of segment dicts with 'text' key (Hindi)."""
    wav = _mp3_to_wav(audio_path)
    PROMPT = (
        "यह एक डॉक्टर और मरीज़ के बीच की बातचीत है। "
        "मरीज़ के लक्षण, दवाइयाँ, और निदान का उल्लेख हो सकता है।"
    )
    try:
        raw_segs, _ = whisper_model.transcribe(
            wav, language="hi", initial_prompt=PROMPT,
            beam_size=5, word_timestamps=False,
        )
        segments = [{"id": i, "text": s.text.strip()} for i, s in enumerate(raw_segs)]
    finally:
        os.unlink(wav)
    return segments


# ── Translator 1: MarianMT (Helsinki-NLP/opus-mt-hi-en) — "LangTrans" ─────────

_MARIAN_MODEL_ID = "Helsinki-NLP/opus-mt-hi-en"

@lru_cache(maxsize=1)
def _load_marian():
    from transformers import MarianMTModel, MarianTokenizer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MarianMT] Loading {_MARIAN_MODEL_ID} on {device}...", flush=True)
    t0 = time.monotonic()
    tokenizer = MarianTokenizer.from_pretrained(_MARIAN_MODEL_ID)
    model = MarianMTModel.from_pretrained(_MARIAN_MODEL_ID).to(device)
    model.eval()
    print(f"[MarianMT] Loaded in {time.monotonic()-t0:.1f}s", flush=True)
    return tokenizer, model, device


def translate_marian(segments: list[dict]) -> tuple[list[dict], float]:
    """Translate segments with MarianMT. Returns (segments_with_translation, latency_s)."""
    import torch
    segs = [dict(s) for s in segments]
    try:
        tokenizer, model, device = _load_marian()
    except Exception as e:
        print(f"  [MarianMT] Load failed: {e}")
        for s in segs:
            s["english_translation"] = "[MarianMT unavailable]"
        return segs, 0.0

    texts = [s["text"] for s in segs]
    t0 = time.monotonic()
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=512).to(device)
    with torch.no_grad():
        translated = model.generate(**inputs, num_beams=4, max_length=512)
    translations = tokenizer.batch_decode(translated, skip_special_tokens=True)
    latency = time.monotonic() - t0

    for i, s in enumerate(segs):
        s["english_translation"] = translations[i] if i < len(translations) else ""
    return segs, round(latency, 2)


# ── Translator 2: Qwen2.5-7b via Ollama ───────────────────────────────────────

_OLLAMA_BASE = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434/v1")
_OLLAMA_MODEL = getattr(config, "OLLAMA_MODEL", "qwen2.5:7b")


def translate_qwen(segments: list[dict]) -> tuple[list[dict], float]:
    """Translate segments with Qwen via Ollama. Returns (segments_with_translation, latency_s)."""
    from openai import OpenAI
    segs = [dict(s) for s in segments]
    client = OpenAI(base_url=_OLLAMA_BASE, api_key="ollama")

    numbered = "\n".join(f"[{i}] {s['text']}" for i, s in enumerate(segs))
    prompt = (
        "You are a medical interpreter. Translate each numbered Hindi segment into "
        "natural, fluent English. Use clinical terminology where appropriate. "
        "Return ONLY a JSON array of strings, one translation per segment, in the same order. "
        "No explanation, no markdown.\n\n" + numbered
    )

    t0 = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=_OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content or ""
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw).strip()
        translations = json.loads(raw) if raw.startswith("[") else []
        if not translations:
            m = re.search(r"\[.*?\]", raw, re.DOTALL)
            translations = json.loads(m.group(0)) if m else []
    except Exception as e:
        print(f"  [Qwen] Error: {e}")
        translations = []
    latency = time.monotonic() - t0

    for i, s in enumerate(segs):
        s["english_translation"] = translations[i] if i < len(translations) else "[Qwen error]"
    return segs, round(latency, 2)


# ── Translator 3: Claude API ───────────────────────────────────────────────────

def translate_claude(segments: list[dict]) -> tuple[list[dict], float]:
    """Translate segments with Claude API. Returns (segments_with_translation, latency_s)."""
    import anthropic
    segs = [dict(s) for s in segments]
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    numbered = "\n".join(f"[{i}] {s['text']}" for i, s in enumerate(segs))
    prompt = (
        "Translate each numbered Hindi segment to English. "
        "Return ONLY a JSON array of strings, one per segment, in the same order. "
        "Example: [\"translation 0\", \"translation 1\"]\n\n" + numbered
    )

    t0 = time.monotonic()
    try:
        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        translations = json.loads(raw)
    except Exception as e:
        print(f"  [Claude] Error: {e}")
        translations = []
    latency = time.monotonic() - t0

    for i, s in enumerate(segs):
        s["english_translation"] = translations[i] if i < len(translations) else "[Claude error]"
    return segs, round(latency, 2)


# ── Quality heuristics ────────────────────────────────────────────────────────
# No ground truth needed — measure word count and clinical term retention.
_CLINICAL_KEYWORDS = [
    "fever", "pain", "cough", "blood", "pressure", "sugar", "diabetes",
    "chest", "breath", "vomit", "joint", "tb", "tuberculosis", "medicine",
    "tablet", "treatment", "test", "doctor", "patient", "day", "week", "month",
]

def _quality_metrics(hindi_texts: list[str], translations: list[str]) -> dict:
    en_words    = sum(len(t.split()) for t in translations)
    hi_words    = sum(len(h.split()) for h in hindi_texts)
    word_ratio  = round(en_words / hi_words, 2) if hi_words else 0
    all_en      = " ".join(translations).lower()
    kw_hits     = sum(1 for kw in _CLINICAL_KEYWORDS if kw in all_en)
    return {
        "word_ratio":  word_ratio,   # en/hi word count ratio — ~1.0-1.5 is typical
        "clinical_kw": kw_hits,      # how many of 20 clinical keywords appear
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("Translation Benchmark: MarianMT (LangTrans) vs Qwen2.5-7b vs Claude")
    print("=" * 72)

    # ── Check Claude key ──
    if not config.ANTHROPIC_API_KEY:
        print("WARNING: ANTHROPIC_API_KEY not set — Claude results will be skipped.")

    # ── Transcribe all clips first (shared step) ──
    whisper_model = _load_whisper()
    print()

    clip_data: list[dict] = []
    for fname, scenario in CLIPS:
        path = AUDIO_DIR / fname
        if not path.exists():
            print(f"SKIP: {fname} not found in {AUDIO_DIR}/")
            continue
        print(f"[whisper] Transcribing: {scenario} ({fname})...", end=" ", flush=True)
        t0 = time.monotonic()
        segments = transcribe_clip(whisper_model, str(path))
        print(f"{time.monotonic()-t0:.1f}s  ({len(segments)} segments)")
        clip_data.append({"fname": fname, "scenario": scenario,
                          "segments": segments})

    if not clip_data:
        print(f"\nNo audio clips found in {AUDIO_DIR}/. Aborting.")
        sys.exit(1)

    print()

    # ── Run all three translators ──
    results = []
    for clip in clip_data:
        scenario = clip["scenario"]
        segs     = clip["segments"]
        hindi_texts = [s["text"] for s in segs]

        print(f"{'─'*72}")
        print(f"Clip: {scenario} ({clip['fname']})")
        print(f"  Hindi segments: {len(segs)}")
        print()

        # MarianMT (LangTrans)
        print("  [MarianMT / LangTrans]  translating...", end=" ", flush=True)
        marian_segs, marian_t = translate_marian(segs)
        marian_translations = [s["english_translation"] for s in marian_segs]
        marian_q = _quality_metrics(hindi_texts, marian_translations)
        print(f"{marian_t}s  |  word-ratio={marian_q['word_ratio']}  clinical-kw={marian_q['clinical_kw']}/{len(_CLINICAL_KEYWORDS)}")

        # Qwen
        print("  [Qwen2.5-7b / Ollama]   translating...", end=" ", flush=True)
        qwen_segs, qwen_t = translate_qwen(segs)
        qwen_translations = [s["english_translation"] for s in qwen_segs]
        qwen_q = _quality_metrics(hindi_texts, qwen_translations)
        print(f"{qwen_t}s  |  word-ratio={qwen_q['word_ratio']}  clinical-kw={qwen_q['clinical_kw']}/{len(_CLINICAL_KEYWORDS)}")

        # Claude
        if config.ANTHROPIC_API_KEY:
            print("  [Claude API]            translating...", end=" ", flush=True)
            claude_segs, claude_t = translate_claude(segs)
            claude_translations = [s["english_translation"] for s in claude_segs]
            claude_q = _quality_metrics(hindi_texts, claude_translations)
            print(f"{claude_t}s  |  word-ratio={claude_q['word_ratio']}  clinical-kw={claude_q['clinical_kw']}/{len(_CLINICAL_KEYWORDS)}")
        else:
            claude_translations = ["[skipped — no API key]"] * len(segs)
            claude_t = 0.0
            claude_q = {"word_ratio": 0, "clinical_kw": 0}

        results.append({
            "scenario":        scenario,
            "fname":           clip["fname"],
            "hindi_texts":     hindi_texts,
            "marian_t":        marian_t,
            "qwen_t":          qwen_t,
            "claude_t":        claude_t,
            "marian_q":        marian_q,
            "qwen_q":          qwen_q,
            "claude_q":        claude_q,
            "marian_out":      marian_translations,
            "qwen_out":        qwen_translations,
            "claude_out":      claude_translations,
        })
        print()

    # ── Console summary ───────────────────────────────────────────────────────
    W = 30
    print("=" * 72)
    print("TIMING SUMMARY (seconds per clip)")
    print("=" * 72)
    print(f"\n{'Scenario':<{W}} {'MarianMT':>10} {'Qwen':>10} {'Claude':>10}")
    print("─" * (W + 33))

    tot_m = tot_q = tot_c = 0.0
    for r in results:
        print(f"{r['scenario']:<{W}} {r['marian_t']:>9.1f}s {r['qwen_t']:>9.1f}s {r['claude_t']:>9.1f}s")
        tot_m += r["marian_t"]
        tot_q += r["qwen_t"]
        tot_c += r["claude_t"]

    print("─" * (W + 33))
    print(f"{'TOTAL':<{W}} {tot_m:>9.1f}s {tot_q:>9.1f}s {tot_c:>9.1f}s")

    print()
    print("=" * 72)
    print("QUALITY SUMMARY (automated heuristics — see report for full translations)")
    print("=" * 72)
    print(f"\n{'Scenario':<{W}} {'MarianMT kw':>12} {'Qwen kw':>10} {'Claude kw':>11}")
    print("─" * (W + 36))
    for r in results:
        mk = r["marian_q"]["clinical_kw"]
        qk = r["qwen_q"]["clinical_kw"]
        ck = r["claude_q"]["clinical_kw"]
        print(f"{r['scenario']:<{W}} {mk:>11}/{len(_CLINICAL_KEYWORDS)} {qk:>9}/{len(_CLINICAL_KEYWORDS)} {ck:>10}/{len(_CLINICAL_KEYWORDS)}")

    # ── Write full report ─────────────────────────────────────────────────────
    with open(REPORT_OUT, "w", encoding="utf-8") as f:

        f.write("Translation Benchmark: MarianMT (LangTrans) vs Qwen2.5-7b vs Claude\n")
        f.write("=" * 72 + "\n\n")
        f.write("Models tested:\n")
        f.write("  [MarianMT]  Helsinki-NLP/opus-mt-hi-en  (previous offline / LangTrans)\n")
        f.write(f"  [Qwen]      {_OLLAMA_MODEL} via Ollama  (current offline)\n")
        f.write(f"  [Claude]    {config.ANTHROPIC_MODEL}  (current online)\n\n")

        # Timing table
        f.write("TIMING SUMMARY\n")
        f.write("─" * 72 + "\n")
        f.write(f"{'Scenario':<{W}} {'MarianMT':>10} {'Qwen':>10} {'Claude':>10}\n")
        f.write("─" * (W + 33) + "\n")
        for r in results:
            f.write(f"{r['scenario']:<{W}} {r['marian_t']:>9.1f}s {r['qwen_t']:>9.1f}s {r['claude_t']:>9.1f}s\n")
        f.write("─" * (W + 33) + "\n")
        f.write(f"{'TOTAL':<{W}} {tot_m:>9.1f}s {tot_q:>9.1f}s {tot_c:>9.1f}s\n\n")

        # Quality heuristics table
        f.write("QUALITY HEURISTICS  (clinical_kw = keywords from 20-word list found in output)\n")
        f.write("─" * 72 + "\n")
        f.write(f"{'Scenario':<{W}} {'MarianMT':>10} {'Qwen':>10} {'Claude':>10}  word-ratio\n")
        f.write("─" * (W + 48) + "\n")
        for r in results:
            mk = r["marian_q"]["clinical_kw"]
            qk = r["qwen_q"]["clinical_kw"]
            ck = r["claude_q"]["clinical_kw"]
            mr = r["marian_q"]["word_ratio"]
            qr = r["qwen_q"]["word_ratio"]
            cr = r["claude_q"]["word_ratio"]
            f.write(
                f"{r['scenario']:<{W}} "
                f"{mk:>6}/{len(_CLINICAL_KEYWORDS)}  "
                f"{qk:>6}/{len(_CLINICAL_KEYWORDS)}  "
                f"{ck:>6}/{len(_CLINICAL_KEYWORDS)}  "
                f"M={mr}  Q={qr}  C={cr}\n"
            )
        f.write("\n\n")

        # Full translation comparison per clip
        f.write("FULL TRANSLATION COMPARISON\n")
        f.write("=" * 72 + "\n")
        for r in results:
            f.write(f"\n{'─'*72}\n")
            f.write(f"Clip: {r['scenario']} ({r['fname']})\n")
            f.write(f"{'─'*72}\n\n")
            for i, hindi in enumerate(r["hindi_texts"]):
                f.write(f"  [{i}] Hindi    : {hindi}\n")
                mout = r["marian_out"][i] if i < len(r["marian_out"]) else ""
                qout = r["qwen_out"][i]   if i < len(r["qwen_out"])   else ""
                cout = r["claude_out"][i] if i < len(r["claude_out"]) else ""
                f.write(f"      MarianMT  : {mout}\n")
                f.write(f"      Qwen      : {qout}\n")
                f.write(f"      Claude    : {cout}\n\n")

    print(f"\nFull report written to: {REPORT_OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
