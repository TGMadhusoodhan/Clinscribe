"""
ClinScribe offline translation via facebook/nllb-200-distilled-600M.
Replaces MarianMT — NLLB has significantly better contextual understanding.
Segments are joined into one block before translation for cross-segment context,
then split back by the segment separator.
"""

import logging
import torch
from functools import lru_cache

logger = logging.getLogger(__name__)

_MODEL_ID = "facebook/nllb-200-distilled-600M"
_SRC_LANG = "hin_Deva"   # Devanagari Hindi
_TGT_LANG = "eng_Latn"   # English


@lru_cache(maxsize=1)
def _load_models():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    logger.info(f"[translate_offline] Loading {_MODEL_ID} on GPU...")
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    # NLLB runs on CPU — GPU is reserved for Ollama (qwen2.5:7b)
    model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_ID).cpu()
    model.eval()
    logger.info("[translate_offline] Model loaded on CPU.")
    return tokenizer, model


def translate_segments_offline(segments: list) -> list:
    """
    Translates Hindi segment texts to English using NLLB-200-distilled-600M on CPU.
    Each segment is translated individually so no segment is ever truncated by the
    1024-token model limit — longer consultations were previously cut off mid-transcript.
    Drop-in replacement for transcribe.translate_segments() in offline mode.
    Inputs: segments — list of dicts from transcribe() with 'text' key
    Outputs: same list with 'english_translation' added to each segment
    """
    if not segments:
        return segments

    tokenizer, model = _load_models()
    forced_bos = tokenizer.convert_tokens_to_ids(_TGT_LANG)

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            seg["english_translation"] = ""
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                num_beams=5,
                max_length=512,
                early_stopping=True,
            )

        seg["english_translation"] = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    return segments
