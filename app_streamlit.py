# src/app_streamlit.py
"""
Streamlit application for the bilingual AI-1900 Chatbot.
Optimized for performance, user experience, and error handling.
Supports Persian and English with a modern chat interface.
"""

import streamlit as st
import logging
from src.chatbot import Chatbot
from src.utils import detect_language

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI-1900 Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- CSS STYLE ---
st.markdown("""
    <style>
        body {
            background-color: #0a0a0f;
        }
        .chat-container {
            max-height: 70vh;
            overflow-y: auto;
            padding: 10px;
        }
        .chat-bubble-user {
            background-color: #1f1f2e;
            color: #00ffff;
            padding: 10px 14px;
            border-radius: 18px;
            margin: 8px 0;
            text-align: right;
            max-width: 80%;
            align-self: flex-end;
            box-shadow: 0 0 8px rgba(0, 255, 255, 0.4);
            animation: fadeIn 0.3s ease-in;
        }
        .chat-bubble-bot {
            background-color: #101020;
            color: #ffffff;
            padding: 10px 14px;
            border-radius: 18px;
            margin: 8px 0;
            text-align: left;
            max-width: 80%;
            align-self: flex-start;
            box-shadow: 0 0 10px rgba(0, 200, 255, 0.3);
            animation: fadeIn 0.3s ease-in;
        }
        .typing {
            color: #00ffff;
            font-style: italic;
            animation: blink 1s infinite;
        }
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(10px);}
            to {opacity: 1; transform: translateY(0);}
        }
        @keyframes blink {
            50% {opacity: 0;}
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 style='text-align:center; color:#00ffff;'>Welcome to the AI-1900 Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888;'>Bilingual Persian-English Chatbot Prototype</p>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #00ffff;'>", unsafe_allow_html=True)

# --- INITIALIZATION ---
def initialize_chatbot():
    """Initialize chatbot with error handling."""
    try:
        if "chatbot" not in st.session_state:
            st.session_state.chatbot = Chatbot()
            logger.info("Chatbot initialized successfully.")
        if "history" not in st.session_state:
            st.session_state.history = []
            logger.info("Chat history initialized.")
        if "history_limit" not in st.session_state:
            st.session_state.history_limit = 50  # Limit history to prevent memory issues
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        st.error("Failed to initialize chatbot. Please try again.")
        st.stop()

initialize_chatbot()

# --- CHAT HISTORY DISPLAY ---
def display_chat_history():
    """Display chat history efficiently."""
    with st.container():
        chat_container = st.container()
        chat_container.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for sender, message in st.session_state.history:
            if sender == "user":
                chat_container.markdown(f"<div class='chat-bubble-user'>{message}</div>", unsafe_allow_html=True)
            else:
                chat_container.markdown(f"<div class='chat-bubble-bot'>{message}</div>", unsafe_allow_html=True)
        chat_container.markdown('</div>', unsafe_allow_html=True)

# --- USER INPUT ---
user_input = st.chat_input("Type your message (in Persian or English)...", key="user_input")

if user_input:
    try:
        # Add user message to history
        st.session_state.history.append(("user", user_input))
        logger.debug(f"User input: {user_input}")

        # Limit history size
        if len(st.session_state.history) > st.session_state.history_limit:
            st.session_state.history = st.session_state.history[-st.session_state.history_limit:]
            logger.info("Chat history trimmed to maintain limit.")

        # Display updated history
        display_chat_history()

        # Detect language
        lang = detect_language(user_input)
        logger.debug(f"Detected language: {lang}")

        # Show typing animation using a spinner
        with st.spinner("🤖 Generating response..."):
            # Generate bot response
            bot_response = st.session_state.chatbot.generate_response(
                user_input,
                max_length=128,
                num_beams=4
            )

        # Add bot response to history
        st.session_state.history.append(("bot", bot_response))
        logger.debug(f"Bot response: {bot_response}")

        # Update display with new response
        display_chat_history()

        # Clear translation cache periodically to manage memory
        if len(st.session_state.history) % 10 == 0:
            st.session_state.chatbot.clear_cache()
            logger.info("Translation cache cleared to optimize memory.")

    except ValueError as ve:
        logger.error(f"Invalid input error: {str(ve)}")
        st.error("Invalid input. Please enter a valid message.")
    except RuntimeError as re:
        logger.error(f"Response generation failed: {str(re)}")
        st.error("Failed to generate response. Please try again.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        st.error("An unexpected error occurred. Please try again.")

# Initial display of chat history
if st.session_state.history:
    display_chat_history()
