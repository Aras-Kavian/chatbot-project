import streamlit as st

st.set_page_config(page_title="🧠 Chatbot", page_icon="💬", layout="centered")

st.title("💬 Free Local Chatbot")
st.caption("Powered by 🤗 HuggingFace & Streamlit — No paid APIs")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input box
user_input = st.chat_input("Type your message...")
if user_input:
    # Append user message immediately
    st.session_state.chat_history.append(("user", user_input))

    # Temporary bot reply (can replace with model later)
    bot_reply = f"You said: {user_input}"
    st.session_state.chat_history.append(("assistant", bot_reply))

# Display chat history as bubbles
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)