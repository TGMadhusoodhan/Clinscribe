"""
ClinScribe offline translation via Helsinki-NLP/opus-mt-hi-en (MarianMT).
Replaces Claude translate_segments() for offline/rural deployments.
Models are lazy-loaded on first call — import is fast.
"""

import logging
import torch
from functools import lru_cache

logger = logging.getLogger(__name__)

_MODEL_ID = "Helsinki-NLP/opus-mt-hi-en"


@lru_cache(maxsize=1)
def _load_models():
    """Load MarianMT Hindi→English model once, cache forever."""
    from transformers import MarianMTModel, MarianTokenizer

    logger.info(f"[translate_offline] Loading {_MODEL_ID}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MarianTokenizer.from_pretrained(_MODEL_ID)
    model = MarianMTModel.from_pretrained(_MODEL_ID).to(device)
    model.eval()
    logger.info(f"[translate_offline] Model loaded on {device}.")
    return tokenizer, model, device


def translate_segments_offline(segments: list) -> list:
    """
    Translates Hindi segment texts to English using MarianMT (Helsinki-NLP/opus-mt-hi-en).
    Drop-in replacement for transcribe.translate_segments() in offline mode.
    Inputs: segments — list of dicts from transcribe() with 'text' key
    Outputs: same list with 'english_translation' added to each segment
    """
    if not segments:
        return segments

    tokenizer, model, device = _load_models()

    texts = [seg["text"] for seg in segments]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        translated = model.generate(**inputs, num_beams=4, max_length=512)

    translations = tokenizer.batch_decode(translated, skip_special_tokens=True)

    for i, seg in enumerate(segments):
        seg["english_translation"] = translations[i] if i < len(translations) else ""

    return segments
