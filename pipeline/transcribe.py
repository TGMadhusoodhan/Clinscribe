"""
ClinScribe transcription pipeline.
Loads Whisper large-v3 and silero-vad once at module level.
All audio is processed locally — never sent to external APIs (HIPAA constraint).
"""

import time
import warnings
import torch
import numpy as np
from pathlib import Path
import anthropic

warnings.filterwarnings("ignore", message="Failed to launch Triton kernels")

import config

# ── Model loading (happens once on import, not per call) ──────────────────────
# Loading takes 10-15 seconds. Per-call loading makes the API unusable.

# Whisper runs on CPU (int8) so the GPU is fully available for Ollama (qwen2.5:7b needs ~4.4GB VRAM).
# faster-whisper with int8 is 1.3x faster than openai-whisper on CPU with equivalent quality
# (benchmarked on 6 Hindi clinical clips — see benchmark_whisper_compare_report.txt).
from faster_whisper import WhisperModel

print(f"[transcribe] Loading Whisper {config.WHISPER_MODEL} on CPU (int8)...")
_whisper_model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
print(f"[transcribe] Whisper loaded.")

# silero-vad removes silence before Whisper, reducing WER on clinic audio
# (clinic recordings have long pauses between speaker turns)
print("[transcribe] Loading silero-vad...")
_vad_model, _vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
)
_get_speech_timestamps, _, _read_audio, *_ = _vad_utils
print("[transcribe] silero-vad loaded.")

# Clinical domain prompt — biases Whisper decoder toward medical vocabulary.
# Hardcoded because it is a fixed domain hint, not a configurable value.
# language="hi" hardcoded — ISO 639-1 standard for Hindi.
# Auto-detect adds 2s latency and misidentifies Hindi as Urdu on clips under 10s.
_INITIAL_PROMPT = (
    "यह एक डॉक्टर और मरीज़ के बीच की बातचीत है। "
    "मरीज़ के लक्षण, दवाइयाँ, और निदान का उल्लेख हो सकता है।"
)


def _remove_silence(audio_path: str) -> np.ndarray:
    """
    What it does: Loads audio, strips silent segments via silero-vad, returns numpy array.
    Inputs: audio_path — str path to WAV file (must be pre-converted; torchaudio on Windows
            requires soundfile backend which only handles WAV/FLAC reliably)
    Outputs: np.ndarray of float32 audio samples at 16kHz, silence removed
    Dependencies: silero-vad loaded at module level; soundfile
    Side effects: None
    Failure modes: FileNotFoundError if path invalid; RuntimeError if audio corrupt
    """
    import soundfile as sf
    from torchaudio.functional import resample as ta_resample

    data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    # Mix down to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data)
    # Resample to 16 kHz if needed
    if sr != 16000:
        wav = ta_resample(wav, sr, 16000)

    speech_timestamps = _get_speech_timestamps(
        wav, _vad_model, sampling_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
    )
    if not speech_timestamps:
        return wav.numpy()
    chunks = [wav[ts["start"]: ts["end"]] for ts in speech_timestamps]
    return torch.cat(chunks).numpy()


def transcribe(audio_path: str) -> dict:
    """
    What it does: Transcribes a Hindi audio file using Whisper large-v3 with VAD preprocessing.
    Inputs: audio_path — str path to MP3 or WAV audio file
    Outputs: dict with keys: segments, full_text, duration, model_used, language
    Dependencies: Whisper and silero-vad loaded at module level; config.py
    Side effects: None (no files written, no external calls)
    Failure modes: FileNotFoundError if audio_path missing; RuntimeError on GPU OOM (reduce batch or use CPU)
    """
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_np = _remove_silence(audio_path)
    duration = len(audio_np) / 16000.0

    # faster-whisper returns a generator — must be consumed before duration is used
    raw_segments, _ = _whisper_model.transcribe(
        audio_np,
        language=config.WHISPER_LANGUAGE,  # "hi" — ISO 639-1 code for Hindi
        initial_prompt=_INITIAL_PROMPT,
        word_timestamps=True,
        beam_size=5,
    )

    segments = []
    for seg in raw_segments:
        words = []
        for w in (seg.words or []):
            words.append({
                "word": w.word,
                "start": round(w.start, 3),
                "end": round(w.end, 3),
            })
        segments.append({
            "id": seg.id,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "language": config.WHISPER_LANGUAGE,
            "words": words,
        })

    return {
        "segments": segments,
        "full_text": " ".join(s["text"] for s in segments),
        "duration": round(duration, 2),
        "model_used": config.WHISPER_MODEL,
        "language": config.WHISPER_LANGUAGE,
    }


def translate_segments(segments: list) -> list:
    """
    What it does: Adds English translations to each segment via a single batched Claude API call.
    Inputs: segments — list of segment dicts from transcribe()
    Outputs: same list with "english_translation" key added to each segment
    Dependencies: anthropic SDK; config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL
    Side effects: One Anthropic API call per invocation
    Failure modes: anthropic.APIError on network/auth failure; json.JSONDecodeError if model returns malformed JSON
    """
    if not segments:
        return segments

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Batch all segments into one call — per-segment calls waste tokens and time
    numbered = "\n".join(
        f"[{i}] {seg['text']}" for i, seg in enumerate(segments)
    )
    prompt = (
        "Translate each numbered Hindi segment to English. "
        "Return ONLY a JSON array of strings, one per segment, in the same order. "
        "Example: [\"translation 0\", \"translation 1\"]\n\n"
        f"{numbered}"
    )

    response = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    import json, re
    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        translations = json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting a JSON array from anywhere in the response
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            translations = json.loads(m.group(0))
        else:
            translations = []

    for i, seg in enumerate(segments):
        seg["english_translation"] = translations[i] if i < len(translations) else ""

    return segments
