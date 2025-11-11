import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
from pathlib import Path

# =========================
# Streamlit & Page Setup
# =========================
st.set_page_config(
    page_title="💼 Alfred — Pavan Kumar’s Personal AI Assistant",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("💼 Alfred — Pavan Kumar’s Personal AI Assistant")

# =========================
# Load API Key
# =========================
api_key = st.secrets.get("GROQ_API_KEY", None)
if not api_key:
    st.error("GROQ_API_KEY not found in Streamlit secrets. Please set it under App Settings → Secrets.")
    st.stop()

# Initialize Groq client
try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

# =========================
# Vector DB Initialization
# =========================
@st.cache_resource
def get_chroma_client():
    return chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="chroma_db"))

chroma_client = get_chroma_client()

@st.cache_resource
def get_embeddings_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = get_embeddings_model()

# =========================
# Load Resume File
# =========================
resume_path = Path("resume_knowledge_base.md")
if not resume_path.exists():
    st.error("Resume file (resume_knowledge_base.md) not found. Please place it in the project root.")
    st.stop()

resume_text = resume_path.read_text(encoding="utf-8")

# =========================
# Vectorize Resume
# =========================
collection = chroma_client.get_or_create_collection("resume_knowledge_base")

if collection.count() == 0:
    # Split resume into chunks (basic)
    from textwrap import wrap
    chunks = wrap(resume_text, width=1000)
    embeddings = embedding_model.encode(chunks).tolist()
    for i, chunk in enumerate(chunks):
        collection.add(ids=[f"chunk_{i}"], embeddings=[embeddings[i]], documents=[chunk])

# =========================
# Helper Functions
# =========================
def search_resume(query, top_k=3):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["documents"][0] if results and "documents" in results else []

def summarize_history(messages, max_len=3000):
    full_text = " ".join([msg["content"] for msg in messages])
    if len(full_text) < max_len:
        return full_text
    else:
        return full_text[-max_len:]  # Keep last N chars

# =========================
# Conversation Handling
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Good day. I am Alfred, Mr. Pavan Kumar’s personal AI assistant. How may I assist you?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# Chat Input
# =========================
if user_input := st.chat_input("Your message to Alfred..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # RAG retrieval
    relevant_chunks = search_resume(user_input)
    resume_context = "\n\n".join(relevant_chunks)

    # Conversation summarization
    history_summary = summarize_history(st.session_state.messages[-8:])

    # Build system prompt
    system_prompt = f"""
You are Alfred Pennyworth, a refined and intelligent assistant representing Pavan Kumar.
Answer the user's query professionally, using Pavan's resume context provided below.

Only use facts from the résumé context. If unsure, respond gracefully that the detail isn’t specified.

Résumé Context:
{resume_context}

Recent Conversation Summary:
{history_summary}
"""

    messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages[-8:] + [
        {"role": "user", "content": user_input}
    ]

    # Stream response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response_text = ""

        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.8,
                max_tokens=512,
                top_p=0.9,
                stream=True,
            )
            for chunk in completion:
                delta = chunk.choices[0].delta.content or ""
                response_text += delta
                message_placeholder.markdown(response_text + "▌")
            message_placeholder.markdown(response_text)
        except Exception as e:
            response_text = f"Apologies, something went wrong: {e}"
            message_placeholder.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
