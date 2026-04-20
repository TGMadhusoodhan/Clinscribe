"""
Tests for pipeline/transcribe.py.

What is tested:
- Shape of the return dict matches the documented contract
- Segment fields have correct types
- Hindi Unicode characters appear in output

What is NOT tested:
- Whisper inference accuracy (too slow for CI; benchmarked separately in corpus/benchmark_whisper.py)
- silero-vad silence removal (requires real audio with silent sections)
- translate_segments (requires live Anthropic API; tested manually)

Mock strategy:
- No mocks for transcribe() — it runs real inference on a real audio file.
- Tests use corpus/clips/clip_01_fever_headache.mp3 if available, else skip.
- If no clips exist, tests requiring audio are skipped with pytest.skip().
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _find_test_clip():
    """Return path to first available test audio clip, or None."""
    candidates = [
        "corpus/clips/clip_01_fever_headache.mp3",
        "benchmark_audio/hindi_1.mp3",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


@pytest.mark.slow
def test_transcribe_returns_correct_shape():
    """
    What it does: Verifies transcribe() returns a dict with all documented top-level keys.
    Inputs: real audio clip from corpus/clips/ or benchmark_audio/
    Outputs: asserts dict shape
    """
    clip = _find_test_clip()
    if clip is None:
        pytest.skip("No test audio clip available")

    from pipeline.transcribe import transcribe
    result = transcribe(clip)

    assert isinstance(result, dict)
    assert "segments" in result
    assert "full_text" in result
    assert "duration" in result
    assert "model_used" in result
    assert "language" in result

    assert isinstance(result["segments"], list)
    assert isinstance(result["full_text"], str)
    assert isinstance(result["duration"], float)
    assert result["language"] == "hi"


@pytest.mark.slow
def test_segment_has_required_fields():
    """
    What it does: Verifies each segment has start/end floats and end > start.
    Inputs: real audio clip
    Outputs: asserts field types and ordering constraint
    """
    clip = _find_test_clip()
    if clip is None:
        pytest.skip("No test audio clip available")

    from pipeline.transcribe import transcribe
    result = transcribe(clip)

    for seg in result["segments"]:
        assert "start" in seg and "end" in seg and "text" in seg
        assert isinstance(seg["start"], float)
        assert isinstance(seg["end"], float)
        assert seg["end"] > seg["start"], "Segment end must be after start"
        assert isinstance(seg["text"], str)


@pytest.mark.slow
def test_hindi_characters_in_output():
    """
    What it does: Verifies output contains Devanagari Unicode characters (U+0900–U+097F).
    Inputs: real Hindi audio clip
    Outputs: asserts at least one Devanagari character in full_text
    Why: If Whisper returns empty or ASCII-only output, something is wrong with language config.
    """
    clip = _find_test_clip()
    if clip is None:
        pytest.skip("No test audio clip available")

    from pipeline.transcribe import transcribe
    result = transcribe(clip)

    has_devanagari = any('\u0900' <= c <= '\u097F' for c in result["full_text"])
    assert has_devanagari, (
        f"Expected Devanagari characters in output but got: {result['full_text'][:100]}"
    )
