# src/utils.py
"""
Utility functions for language detection and text translation.
Optimized for performance with improved language detection, error handling, and device management.
"""

import torch
import logging
import re
from typing import Dict, Tuple
from transformers import PreTrainedTokenizer, PreTrainedModel

# Optional: Use langdetect for more robust language detection
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_language(text: str) -> str:
    """
    Detect the language of the input text (Persian or English).

    Args:
        text (str): Input text to analyze.

    Returns:
        str: 'fa' for Persian, 'en' for English.

    Raises:
        ValueError: If input is empty or invalid.
    """
    if not text or not isinstance(text, str):
        logger.error("Invalid or empty input provided for language detection.")
        raise ValueError("Input must be a non-empty string.")

    try:
        # Use langdetect if available and text is long enough
        if LANGDETECT_AVAILABLE and len(text) > 10:
            lang = detect(text)
            if lang in ["fa", "en"]:
                logger.debug(f"Detected language (langdetect): {lang}")
                return lang

        # Fallback to heuristic-based detection
        fa_chars = len(re.findall(r"[\u0600-\u06FF]", text))
        threshold = len(text) / 4
        detected_lang = "fa" if fa_chars > threshold else "en"
        logger.debug(f"Detected language (heuristic): {detected_lang}")
        return detected_lang

    except Exception as e:
        logger.error(f"Language detection failed: {str(e)}")
        # Fallback to English if detection fails
        return "en"

@torch.no_grad()
def translate_text(
    text: str,
    direction: str,
    models: Dict[str, Tuple[PreTrainedTokenizer, PreTrainedModel]],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_length: int = 128,
    num_beams: int = 4
) -> str:
    """
    Translate text between Persian and English using the specified model.

    Args:
        text (str): Input text to translate.
        direction (str): Translation direction ('fa_en' or 'en_fa').
        models (dict): Dictionary containing (tokenizer, model) pairs for translation.
        device (str): Device to run the model on ('cuda' or 'cpu').
        max_length (int): Maximum length for generated translation.
        num_beams (int): Number of beams for beam search.

    Returns:
        str: Translated text.

    Raises:
        ValueError: If input is empty or direction is invalid.
        RuntimeError: If translation fails.
    """
    if not text or not isinstance(text, str):
        logger.error("Invalid or empty input provided for translation.")
        raise ValueError("Input must be a non-empty string.")

    if direction not in ["fa_en", "en_fa"]:
        logger.error(f"Invalid translation direction: {direction}")
        raise ValueError("Direction must be 'fa_en' or 'en_fa'.")

    try:
        tokenizer, model = models[direction]
        # Move input to the specified device
        input_ids = tokenizer.encode(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(device)

        # Generate translation
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=2,  # Prevent repetitive phrases
            early_stopping=True      # Stop early to save computation
        )
        translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.debug(f"Translated text ({direction}): {translated_text}")
        return translated_text

    except Exception as e:
        logger.error(f"Translation failed for direction {direction}: {str(e)}")
        raise RuntimeError(f"Translation failed: {str(e)}")
