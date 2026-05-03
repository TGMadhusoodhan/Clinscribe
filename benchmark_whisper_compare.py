"""
Benchmark: openai-whisper vs faster-whisper (large-v3, INT8) on Hindi clinical audio.
Metrics: latency per clip, total time, transcript side-by-side for quality check.
No ground truth needed — visual transcript comparison is sufficient.
"""

import time
import os
import io
import tempfile
from pathlib import Path
from pydub import AudioSegment

CLIPS = [
    ("hindi_1.mp3", "Fever and headache"),
    ("hindi_2.mp3", "Chest pain and breathlessness"),
    ("hindi_3.mp3", "Diabetes follow-up"),
    ("hindi_4.mp3", "Cough and TB concern"),
    ("hindi_5.mp3", "Stomach pain and vomiting"),
    ("hindi_6.mp3", "Joint pain"),
]
AUDIO_DIR = Path("benchmark_audio")
LANGUAGE = "hi"
INITIAL_PROMPT = (
    "यह एक डॉक्टर और मरीज़ के बीच की बातचीत है। "
    "मरीज़ के लक्षण, दवाइयाँ, और निदान का उल्लेख हो सकता है।"
)


def mp3_to_wav(mp3_path: str) -> str:
    seg = AudioSegment.from_file(mp3_path, format="mp3")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    seg.export(tmp.name, format="wav")
    return tmp.name


# ── openai-whisper ─────────────────────────────────────────────────────────────

def load_openai_whisper():
    import warnings
    warnings.filterwarnings("ignore")
    import whisper
    print("[openai-whisper] Loading large-v3 on CPU...")
    t0 = time.monotonic()
    model = whisper.load_model("large-v3", device="cpu")
    print(f"[openai-whisper] Loaded in {time.monotonic()-t0:.1f}s")
    return model


def run_openai_whisper(model, wav_path: str) -> tuple[str, float]:
    import whisper
    t0 = time.monotonic()
    result = model.transcribe(
        wav_path,
        language=LANGUAGE,
        initial_prompt=INITIAL_PROMPT,
        fp16=False,
    )
    elapsed = time.monotonic() - t0
    return result["text"].strip(), round(elapsed, 2)


# ── faster-whisper ─────────────────────────────────────────────────────────────

def load_faster_whisper():
    from faster_whisper import WhisperModel
    print("[faster-whisper] Loading large-v3 on CPU (int8)...")
    t0 = time.monotonic()
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    print(f"[faster-whisper] Loaded in {time.monotonic()-t0:.1f}s")
    return model


def run_faster_whisper(model, wav_path: str) -> tuple[str, float]:
    t0 = time.monotonic()
    segments, info = model.transcribe(
        wav_path,
        language=LANGUAGE,
        initial_prompt=INITIAL_PROMPT,
        beam_size=5,
        word_timestamps=False,
    )
    text = " ".join(seg.text.strip() for seg in segments)
    elapsed = time.monotonic() - t0
    return text.strip(), round(elapsed, 2)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Whisper Benchmark: openai-whisper vs faster-whisper (large-v3, INT8)")
    print("=" * 70)

    ow_model = load_openai_whisper()
    fw_model = load_faster_whisper()

    results = []

    for fname, scenario in CLIPS:
        path = AUDIO_DIR / fname
        if not path.exists():
            print(f"\nSKIP: {fname} not found")
            continue

        print(f"\n{'-'*70}")
        print(f"Clip: {scenario} ({fname})")
        print(f"{'-'*70}")

        wav = mp3_to_wav(str(path))
        try:
            print("  [openai-whisper]  transcribing...", end=" ", flush=True)
            ow_text, ow_time = run_openai_whisper(ow_model, wav)
            print(f"{ow_time}s")

            print("  [faster-whisper]  transcribing...", end=" ", flush=True)
            fw_text, fw_time = run_faster_whisper(fw_model, wav)
            print(f"{fw_time}s  ({ow_time/fw_time:.1f}x faster)")

            # Hindi text can't print to Windows cp1252 console — written to file below
            print(f"  (transcripts saved to benchmark_whisper_compare_report.txt)")

            results.append({
                "scenario": scenario,
                "ow_time": ow_time,
                "fw_time": fw_time,
                "speedup": round(ow_time / fw_time, 2),
                "ow_text": ow_text,
                "fw_text": fw_text,
            })
        finally:
            os.unlink(wav)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Scenario':<35} {'openai':>8} {'faster':>8} {'speedup':>8}")
    print("-" * 62)

    total_ow = total_fw = 0
    for r in results:
        print(f"{r['scenario']:<35} {r['ow_time']:>7.1f}s {r['fw_time']:>7.1f}s {r['speedup']:>7.1f}x")
        total_ow += r["ow_time"]
        total_fw += r["fw_time"]

    if results:
        avg_speedup = round(total_ow / total_fw, 2)
        print("-" * 62)
        print(f"{'TOTAL':<35} {total_ow:>7.1f}s {total_fw:>7.1f}s {avg_speedup:>7.1f}x")

    # Write full transcript comparison to UTF-8 file
    report_path = "benchmark_whisper_compare_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Whisper Benchmark: openai-whisper vs faster-whisper (large-v3, INT8)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Scenario':<35} {'openai':>8} {'faster':>8} {'speedup':>8}\n")
        f.write("-" * 62 + "\n")
        for r in results:
            f.write(f"{r['scenario']:<35} {r['ow_time']:>7.1f}s {r['fw_time']:>7.1f}s {r['speedup']:>7.1f}x\n")
        if results:
            f.write("-" * 62 + "\n")
            f.write(f"{'TOTAL':<35} {total_ow:>7.1f}s {total_fw:>7.1f}s {avg_speedup:>7.1f}x\n")
        f.write("\n\nTRANSCRIPT QUALITY COMPARISON\n")
        f.write("=" * 70 + "\n")
        for r in results:
            f.write(f"\n{r['scenario']}\n")
            f.write(f"  openai : {r['ow_text']}\n")
            f.write(f"  faster : {r['fw_text']}\n")

    print(f"\nFull transcript comparison saved to: {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
