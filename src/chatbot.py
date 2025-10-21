# src/chatbot.py
"""
Chatbot Core Logic
------------------
Implements a bilingual chatbot with support for Persian and English.
Uses BlenderBot for conversation and MT5-based models for translation.
Optimized for performance with lazy loading, caching, and error handling.
"""

import torch
import logging
from functools import lru_cache
from src.model_loader import load_blenderbot_model, load_translation_models
from src.utils import detect_language, translate_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Chatbot:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the bilingual chatbot with lazy-loaded models.

        Args:
            device (str): Device to run models on ('cuda' or 'cpu').
        """
        self.device = device
        self.bb_tokenizer = None
        self.bb_model = None
        self.translation_models = None
        self._is_initialized = False
        logger.info(f"Chatbot initialized with device: {self.device}")

    def initialize_models(self):
        """Lazily load models only when needed."""
        if not self._is_initialized:
            try:
                self.bb_tokenizer, self.bb_model = load_blenderbot_model(device=self.device)
                self.translation_models = load_translation_models(device=self.device)
                self._is_initialized = True
                logger.info("Models loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load models: {str(e)}")
                raise RuntimeError(f"Model loading failed: {str(e)}")

    @lru_cache(maxsize=100)
    def _cached_translate(self, text, direction):
        """Cache translation results to avoid redundant computations."""
        return translate_text(text, direction, self.translation_models)

    @torch.no_grad()
    def generate_response(self, user_input, max_length=128, num_beams=4):
        """
        Generate a bilingual response based on user input.

        Args:
            user_input (str): Input text from the user (Persian or English).
            max_length (int): Maximum length of the generated response.
            num_beams (int): Number of beams for beam search in BlenderBot.

        Returns:
            str: Response in the user's input language.

        Raises:
            ValueError: If input is empty or invalid.
            RuntimeError: If models are not loaded or inference fails.
        """
        if not user_input or not isinstance(user_input, str):
            logger.error("Invalid or empty input provided.")
            raise ValueError("Input must be a non-empty string.")

        # Ensure models are loaded
        if not self._is_initialized:
            self.initialize_models()

        try:
            # Detect input language
            user_lang = detect_language(user_input)
            logger.debug(f"Detected language: {user_lang}")

            # Translate to English if Persian
            if user_lang == "fa":
                input_for_bot = self._cached_translate(user_input, "fa_en")
            else:
                input_for_bot = user_input

            # Generate response using BlenderBot
            inputs = self.bb_tokenizer([input_for_bot], return_tensors="pt").to(self.device)
            reply_ids = self.bb_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=2,  # Prevent repetitive phrases
                early_stopping=True      # Stop early to save computation
            )
            response_en = self.bb_tokenizer.decode(reply_ids[0], skip_special_tokens=True)

            # Translate response back to Persian if needed
            if user_lang == "fa":
                response_final = self._cached_translate(response_en, "en_fa")
            else:
                response_final = response_en

            logger.debug(f"Generated response: {response_final}")
            return response_final

        except Exception as e:
            logger.error(f"Error during response generation: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")

    def clear_cache(self):
        """Clear the translation cache to free memory."""
        self._cached_translate.cache_clear()
        logger.info("Translation cache cleared.")
