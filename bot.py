import os
import re
import pathlib
import textwrap
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from groq import Groq

# ---------------------------
# CONFIGURATION
# ---------------------------
RESUME_PATH = "resume_knowledge_base.md"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 4
MAX_HISTORY = 8
SUMMARIZE_AFTER = 14

# ---------------------------
# INITIALIZE
# ---------------------------
st.set_page_config(page_title="Pavan's Assistant - Alfred", layout="centered")

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_chroma_client():
    return chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="chroma_db"))

def get_groq_client():
    key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not key:
        st.error("GROQ_API_KEY missing. Please add it to .streamlit/secrets.toml")
        st.stop()
    return Groq(api_key=key)

# ---------------------------
# RESUME LOADING
# ---------------------------
def load_resume(path=RESUME_PATH):
    if not os.path.exists(path):
        st.error(f"Resume file '{path}' not found.")
        st.stop()
    return pathlib.Path(path).read_text(encoding="utf-8")

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# ---------------------------
# VECTOR DB SETUP
# ---------------------------
def ensure_index(client, model, resume_text):
    try:
        coll = client.get_collection("resume_index")
    except Exception:
        coll = client.create_collection("resume_index")

    if coll.count() == 0:
        st.info("Indexing resume into Chroma (first time only)...")
        chunks = chunk_text(resume_text)
        embeddings = model.encode(chunks, convert_to_numpy=True).tolist()
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        coll.add(ids=ids, embeddings=embeddings, documents=chunks)
        try:
            client.persist()
        except Exception:
            pass

def query_resume(client, model, query, top_k=TOP_K):
    coll = client.get_collection("resume_index")
    q_emb = model.encode([query], convert_to_numpy=True).tolist()
    results = coll.query(query_embeddings=q_emb, n_results=top_k, include=["documents"])
    docs = results["documents"][0] if results["documents"] else []
    return docs

# ---------------------------
# HELPERS
# ---------------------------
def extract_name(text):
    patterns = [r"my name is\s+(\w+)", r"i'?m\s+(\w+)", r"i am\s+(\w+)", r"call me\s+(\w+)", r"this is\s+(\w+)"]
    for p in patterns:
        match = re.search(p, text, flags=re.I)
        if match:
            return match.group(1).capitalize()
    return None

def summarize(client, messages):
    convo = "\n".join([f"{m['role']}: {m['content']}" for m in messages[-20:]])
    msgs = [
        {"role": "system", "content": "You summarize chats in one short paragraph."},
        {"role": "user", "content": convo}
    ]
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=msgs,
        max_tokens=150,
        temperature=0.3
    )
    return completion.choices[0].message["content"].strip()

def make_prompt(context, summary, question):
    context_text = "\n\n".join(textwrap.shorten(c, 480) for c in context)
    system = f"""
You are Alfred Pennyworth, the intelligent and composed assistant to Pavan Kumar.

Knowledge Base:
{context_text}

Conversation Summary:
{summary}

Answer the user briefly (2-4 sentences) with authority and professionalism.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]

# ---------------------------
# LLM CALL
# ---------------------------
def query_groq(client, messages):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.8,
        max_tokens=512,
        top_p=0.9,
        stream=True
    )
    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    return response.strip()

# ---------------------------
# MAIN
# ---------------------------
def main():
    st.title("💼 Alfred — Pavan Kumar’s Personal AI Assistant")

    # Session
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Good day! I’m Alfred, Mr. Pavan Kumar’s personal assistant. May I know your name?"}]
    if "summary" not in st.session_state:
        st.session_state.summary = "No previous context yet."
    if "user_name" not in st.session_state:
        st.session_state.user_name = None

    # Init
    resume_text = load_resume()
    embed_model = get_embedding_model()
    chroma_client = get_chroma_client()
    ensure_index(chroma_client, embed_model, resume_text)
    groq_client = get_groq_client()

    # Sidebar
    with st.sidebar:
        st.header("Session Control")
        if st.button("🔄 Reset Chat"):
            st.session_state.clear()
            st.rerun()
        st.write("User:", st.session_state.user_name or "Unknown")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if user_input := st.chat_input("Your message to Alfred..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Extract name
        name = extract_name(user_input)
        if name:
            st.session_state.user_name = name
            st.session_state.messages.append({"role": "assistant", "content": f"My apologies, {name}. I shall address you properly from now on."})

        # Summarize if long
        if len(st.session_state.messages) > SUMMARIZE_AFTER:
            st.session_state.summary = summarize(groq_client, st.session_state.messages)
            st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

        # Retrieve context
        context = query_resume(chroma_client, embed_model, user_input)
        if not context:
            context = ["No relevant résumé context found."]

        # Build and send
        messages = make_prompt(context, st.session_state.summary, user_input)
        response = query_groq(groq_client, messages)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
