# src/model_loader.py
"""
Model Loader Module
-------------------
Efficiently loads and caches the conversation and translation models.
Supports device management (CPU/GPU), error handling, and optional model optimization.
"""

import torch
import logging
from functools import lru_cache
from transformers import (
    BlenderbotSmallTokenizer,
    BlenderbotSmallForConditionalGeneration,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def load_blenderbot_model(device="cuda" if torch.cuda.is_available() else "cpu", use_fp16=False):
    """
    Load and cache BlenderBot small model for dialogue.

    Args:
        device (str): Device to load the model on ('cuda' or 'cpu').
        use_fp16 (bool): Use mixed precision (float16) for GPU to save memory.

    Returns:
        tuple: (tokenizer, model) for BlenderBotSmall.

    Raises:
        RuntimeError: If model loading fails.
    """
    try:
        logger.info(f"Loading BlenderBotSmall model on {device}...")
        model_name = "facebook/blenderbot_small-90M"
        tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_name)
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(model_name)

        # Move model to specified device
        model = model.to(device)
        
        # Use mixed precision if enabled and on GPU
        if use_fp16 and device == "cuda":
            model = model.half()
            logger.info("Applied FP16 precision to BlenderBot model.")

        # Set model to evaluation mode
        model.eval()
        logger.info("BlenderBotSmall model loaded successfully.")
        return tokenizer, model

    except Exception as e:
        logger.error(f"Failed to load BlenderBot model: {str(e)}")
        raise RuntimeError(f"BlenderBot model loading failed: {str(e)}")

@lru_cache(maxsize=1)
def load_translation_models(device="cuda" if torch.cuda.is_available() else "cpu", use_fp16=False):
    """
    Load and cache Persian <-> English translation models.

    Args:
        device (str): Device to load the models on ('cuda' or 'cpu').
        use_fp16 (bool): Use mixed precision (float16) for GPU to save memory.

    Returns:
        dict: Dictionary with keys 'fa_en' and 'en_fa', each containing (tokenizer, model).

    Raises:
        RuntimeError: If model loading fails.
    """
    try:
        logger.info(f"Loading translation models on {device}...")
        fa_en_model_name = "persiannlp/mt5-small-parsinlu-opus-translation_fa_en"
        en_fa_model_name = "persiannlp/mt5-small-parsinlu-translation_en_fa"

        # Load Persian to English model
        fa_en_tokenizer = MT5Tokenizer.from_pretrained(fa_en_model_name)
        fa_en_model = MT5ForConditionalGeneration.from_pretrained(fa_en_model_name)
        fa_en_model = fa_en_model.to(device)
        if use_fp16 and device == "cuda":
            fa_en_model = fa_en_model.half()

        # Load English to Persian model
        en_fa_tokenizer = MT5Tokenizer.from_pretrained(en_fa_model_name)
        en_fa_model = MT5ForConditionalGeneration.from_pretrained(en_fa_model_name)
        en_fa_model = en_fa_model.to(device)
        if use_fp16 and device == "cuda":
            en_fa_model = en_fa_model.half()

        # Set models to evaluation mode
        fa_en_model.eval()
        en_fa_model.eval()

        logger.info("Translation models loaded successfully.")
        return {
            "fa_en": (fa_en_tokenizer, fa_en_model),
            "en_fa": (en_fa_tokenizer, en_fa_model),
        }

    except Exception as e:
        logger.error(f"Failed to load translation models: {str(e)}")
        raise RuntimeError(f"Translation models loading failed: {str(e)}")
