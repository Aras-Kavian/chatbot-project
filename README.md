# 🤖 Bilingual Chatbot (Persian-English)

A **bilingual chatbot** built with PyTorch and Transformers to support conversations in **Persian** and **English**.  
This project demonstrates a complete workflow: loading pre-trained models (BlenderBotSmall for dialogue and MT5 for translation), handling bilingual inputs, and deploying an interactive chat interface with Streamlit.

---

## 🚀 Demo

👉 [Live Streamlit App](https://huggingface.co/spaces/AI-1900/AI1900-Bilingual-Chatbot)  
- (Chat in Persian or English and get responses in the same language in real-time.)

---

## ✨ Features

- Supports **bilingual conversations** (Persian and English) with automatic language detection  
- Uses **BlenderBotSmall** for dialogue generation and **MT5-based models** for translation  
- Optimized for performance with lazy model loading, translation caching, and efficient memory management  
- **Streamlit UI** for an interactive and user-friendly chat experience  

---

## 🧱 Project Structure

bilingual-chatbot-project/
- notebooks/
- └── chatbot_notebook.ipynb
- src/
- ├── __init__.py
- ├── chatbot.py
- ├── model_loader.py
- └── utils.py
- app_streamlit.py
- requirements.txt
- README.md

---

## 🛠️ Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Aras-Kavian/bilingual-chatbot-project.git
cd bilingual-chatbot-project
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run src/app_streamlit.py
```

- Then open your browser at `http://localhost:8501`.

---

## 🧠 Technologies Used

- PyTorch
- Transformers (Hugging Face)
- Streamlit
- Langdetect (optional, for enhanced language detection)
- Python 3.10+

---

## 🌍 Author & Links

#### 👤 Aras Kavyani / AI 1900
- 🔗 [GitHub](#www.github.com/Aras-Kavian)
- 🔗 [LinkedIn](#www.linkedin.com/in/aras-kavyani)
- 🔗 [LaborX Profile](#www.laborx.com/customers/users/id409982?ref=409982)
- 🔗 [CryptoTask Profile](#www.cryptotask.org/en/freelancers/aras-kavyan/46480)
- 🔗 [Twitter](#www.x.com/ai_1900?s=21)
