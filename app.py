import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import faiss
import time
from dotenv import load_dotenv
import os
load_dotenv()

# --------------------------
# Embedding Function (Ollama)
# --------------------------
def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]

# --------------------------
# Load DB + Build FAISS Index
# --------------------------
@st.cache_resource
def load_embeddings():
    df = pd.read_pickle("embeddings2.pkl")
    embeddings = np.vstack(df["embedding"].to_numpy()).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity
    index.add(embeddings)
    return df, index

df, index = load_embeddings()

# --------------------------
# Streamlit Config
# --------------------------
st.set_page_config(page_title="SamarthBot", layout="wide")

# --------------------------
# Dark / Light Theme Toggle
# --------------------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

def toggle_theme():
    st.session_state["theme"] = "dark" if st.session_state["theme"] == "light" else "light"

theme_css = {
    "light": """
        body { background-color: #f9fafb; color: #111827; }
        .stChatMessage { border-radius: 14px; padding: 14px; margin-bottom: 12px; font-size: 16px; }
        .user-msg { background: linear-gradient(135deg, #4f46e5, #4338ca); color: white; text-align: right; }
        .bot-msg { background: white; border: 1px solid #e5e7eb; color: black; }
        .skeleton { background: #e5e7eb; height: 60px; border-radius: 10px; margin: 10px 0; animation: pulse 1.5s infinite; }
    """,
    "dark": """
        body { background-color: #1f2937; color: #f9fafb; }
        .stChatMessage { border-radius: 14px; padding: 14px; margin-bottom: 12px; font-size: 16px; }
        .user-msg { background: linear-gradient(135deg, #6366f1, #4f46e5); color: white; text-align: right; }
        .bot-msg { background: #374151; border: 1px solid #4b5563; color: #f9fafb; }
        .skeleton { background: #4b5563; height: 60px; border-radius: 10px; margin: 10px 0; animation: pulse 1.5s infinite; }
    """
}
st.markdown(f"<style>{theme_css[st.session_state['theme']]}</style>", unsafe_allow_html=True)
st.sidebar.button("🌗 Toggle Theme", on_click=toggle_theme)
if st.sidebar.button("🔄 Clear Chat"):
    st.session_state["messages"] = []
    st.rerun()


st.sidebar.button("### 📌Chat History")

st.markdown("""
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2561eb, #1e40af); 
    color: white;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: white !important;
    font-weight: 500;
}

/* Sidebar buttons */
.stButton > button {
    background-color: #3b82f6;
    color: white;
    border-radius: 8px;
    padding: 6px 12px;
    border: none;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #1e40af;
    color: #f9fafb;
}

/* Sidebar headers (like Chat History) */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #e0f2fe !important; /* Light blue text */
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.user-msg {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    text-align: right;
    margin-left: auto;
    border-radius: 18px 18px 0 18px;
    padding: 12px 16px;
    max-width: 70%;
}
.bot-msg {
    background: #f3f4f6;
    color: #111827;
    text-align: left;
    margin-right: auto;
    border-radius: 18px 18px 18px 0;
    padding: 12px 16px;
    max-width: 70%;
}
.typing {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 8px 0;
  font-size: 14px;
  color: #6b7280; /* grey text */
}
.dots {
  display: flex;
  gap: 4px;
}
.dot {
  width: 6px;
  height: 6px;
  background: #6b7280;
  border-radius: 50%;
  animation: bounce 1.3s infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}

</style>
""", unsafe_allow_html=True)

# --------------------------
# Title
# --------------------------
st.markdown("""
    <style>
        .header-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .app-logo {
            width: 45px;
            height: 45px;
        }
        .app-title {
            font-weight: 700;
            margin-bottom: 0;
            text-align: left;
            font-size: 2.2rem; /* Desktop size */
        }
        .app-subtitle {
            font-size: 1rem;
            color: #6b7280; /* grey text */
            margin-top: 0.2rem;
        }
        @media (max-width: 600px) {
            .app-title { font-size: 1.4rem !important; }
            .app-subtitle { font-size: 0.85rem; }
        }
    </style>
    <div class="header-container">
        <img src="https://cdn-icons-png.flaticon.com/512/2966/2966488.png" class="app-logo" />
        <div>
            <h1 class="app-title">SamarthBot</h1>
            <p class="app-subtitle">Ask questions about agriculture and climate using official government data</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# --------------------------
# Store conversation
# --------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "embed_cache" not in st.session_state:
    st.session_state["embed_cache"] = {}

# --------------------------
# Chat UI
# --------------------------
for msg in st.session_state.messages:
    role = msg["role"]
    text = msg["text"]
    css_class = "user-msg" if role == "user" else "bot-msg"
    st.markdown(f"<div class='stChatMessage {css_class}'>{text}</div>", unsafe_allow_html=True)

    if "images" in msg and msg["images"]:
        st.image(msg["images"], width=250, caption=["Related Image"] * len(msg["images"]))

# --------------------------
# User Input
# --------------------------
query = st.chat_input("Ask a medical question...")

if query:
    st.session_state.messages.append({"role": "user", "text": query})
    st.rerun()

# --------------------------
# Process Last User Message
# --------------------------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    skeleton_placeholder = st.empty()
    skeleton_placeholder.markdown("""
<div class="typing">
  <div class="dots">
    <span class="dot"></span>
    <span class="dot"></span>
    <span class="dot"></span>
  </div>
  <span>Bot is typing...</span>
</div>
""", unsafe_allow_html=True)


    query = st.session_state.messages[-1]["text"]

    # Cached Embedding
    if query not in st.session_state.embed_cache:
        st.session_state.embed_cache[query] = create_embedding([query])[0]
    question_embedding = np.array([st.session_state.embed_cache[query]], dtype="float32")
    faiss.normalize_L2(question_embedding)

    # FAISS Search
    D, I = index.search(question_embedding, 20)
    retrieved_df = df.iloc[I[0]]
    # print("retrieved_df",retrieved_df)
    context = "\n".join(
        f"[{row['disease']} - {row['topic']}] {row['title']}"
        for _, row in retrieved_df.iterrows()
    )

#     prompt = f"""You are a helpful medical assistant.
# Answer the following question using only the information from the context.
# If the answer is not in the context, say "I don’t know based on available data."

# Context:
# {context}

# Question: {query}
# Answer:"""
    prompt = f"""You are SamarthBot, a helpful assistant for Indian government data on agriculture and climate. 
Answer the following question using only the information from the official datasets provided. 
If the answer is not in the data, say "I don’t know based on available data." 
Always provide the source of the data you are using.

Context:
{context}

Question: {query}
Answer:"""



    # --------------------------
    # Perplexity API Call
    # --------------------------
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('PPLX_API_KEY')}",  # replace with real key
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar",  # or "sonar reasoning", "sonar deep research"
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(url, headers=headers, json=payload)

    bot_placeholder = st.empty()
    skeleton_placeholder.empty()

    if response.status_code == 200:
        try:
            final_answer = response.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            final_answer = "⚠️ Could not parse Perplexity response."
    else:
        final_answer = f"⚠️ API Error {response.status_code}: {response.text}"

    # --------------------------
    # Typing Effect Simulation
    # --------------------------
    typed_text = ""
    for char in final_answer:
        typed_text += char
        bot_placeholder.markdown(
            f"<div class='stChatMessage bot-msg'>{typed_text}▋</div>",
            unsafe_allow_html=True
        )
        time.sleep(0.02)  # Adjust typing speed (seconds per char)

    # remove cursor at the end
    bot_placeholder.markdown(
        f"<div class='stChatMessage bot-msg'>{final_answer}</div>",
        unsafe_allow_html=True
    )

    # Save bot response
    st.session_state.messages.append({"role": "bot", "text": final_answer})
    st.rerun()