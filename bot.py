import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
import sys

# =========================
# Streamlit & Page Setup
# =========================
st.set_page_config(
    page_title="💼 Alfred — Pavan Kumar's Personal AI Assistant",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("💼 Alfred — Pavan Kumar's Personal AI Assistant")

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
    """Initialize ChromaDB client with proper configuration"""
    try:
        # Use updated ChromaDB configuration
        return chromadb.PersistentClient(path="./chroma_db")
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {e}")
        st.stop()

chroma_client = get_chroma_client()

@st.cache_resource
def get_embeddings_model():
    """Load sentence transformer model"""
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Failed to load embeddings model: {e}")
        st.stop()

embedding_model = get_embeddings_model()

# =========================
# Load Resume File
# =========================
resume_path = Path("resume_knowledge_base.md")
if not resume_path.exists():
    st.error("Resume file (resume_knowledge_base.md) not found. Please place it in the project root.")
    st.stop()

try:
    resume_text = resume_path.read_text(encoding="utf-8")
except Exception as e:
    st.error(f"Failed to read resume file: {e}")
    st.stop()

# =========================
# Vectorize Resume
# =========================
@st.cache_resource
def initialize_knowledge_base(_chroma_client, _embedding_model, resume_content):
    """Initialize or load the resume knowledge base"""
    try:
        collection = _chroma_client.get_or_create_collection(
            name="resume_knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        
        if collection.count() == 0:
            # Split resume into meaningful chunks
            chunks = []
            current_chunk = ""
            lines = resume_content.split('\n')
            
            for line in lines:
                # Start new chunk on headers or after reaching size limit
                if line.startswith('#') and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = line + '\n'
                elif len(current_chunk) > 800:
                    chunks.append(current_chunk.strip())
                    current_chunk = line + '\n'
                else:
                    current_chunk += line + '\n'
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Remove empty chunks
            chunks = [c for c in chunks if c.strip()]
            
            # Generate embeddings and add to collection
            embeddings = _embedding_model.encode(chunks, show_progress_bar=False).tolist()
            
            collection.add(
                ids=[f"chunk_{i}" for i in range(len(chunks))],
                embeddings=embeddings,
                documents=chunks,
                metadatas=[{"chunk_id": i} for i in range(len(chunks))]
            )
            
            st.success(f"✅ Initialized knowledge base with {len(chunks)} chunks")
        
        return collection
    
    except Exception as e:
        st.error(f"Failed to initialize knowledge base: {e}")
        st.stop()

collection = initialize_knowledge_base(chroma_client, embedding_model, resume_text)

# =========================
# Helper Functions
# =========================
def search_resume(query, top_k=3):
    """Search resume using semantic similarity"""
    try:
        query_embedding = embedding_model.encode([query], show_progress_bar=False).tolist()
        results = collection.query(
            query_embeddings=query_embedding, 
            n_results=top_k
        )
        
        if results and "documents" in results and results["documents"]:
            return results["documents"][0]
        return []
    
    except Exception as e:
        st.warning(f"Search error: {e}")
        return []

def summarize_history(messages, max_len=2000):
    """Summarize conversation history to fit context window"""
    full_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        full_text += f"{role}: {content}\n"
    
    if len(full_text) <= max_len:
        return full_text
    else:
        # Keep the last max_len characters
        return "...(earlier conversation)...\n" + full_text[-max_len:]

# =========================
# Initialize Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Good day. I am Alfred, Mr. Pavan Kumar's personal AI assistant. How may I assist you today?"
        }
    ]

# =========================
# Display Chat History
# =========================
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
    relevant_chunks = search_resume(user_input, top_k=3)
    
    if relevant_chunks:
        resume_context = "\n\n---\n\n".join(relevant_chunks)
    else:
        resume_context = "No specific resume context found for this query."

    # Conversation history (keep last 6 messages for context)
    recent_messages = st.session_state.messages[-7:-1] if len(st.session_state.messages) > 1 else []
    history_summary = summarize_history(recent_messages, max_len=1500)

    # Build system prompt
    system_prompt = f"""You are Alfred Pennyworth, the distinguished and articulate personal assistant to Mr. Pavan Kumar.

Your demeanor is professional, refined, and helpful. You speak with clarity and precision, always maintaining a respectful tone.

When answering questions about Mr. Pavan Kumar:
- Use ONLY information from the résumé context provided below
- Be precise and factual
- If information is not in the context, politely say "I don't have that specific information in Mr. Kumar's records"
- Speak in first person when representing Mr. Kumar (e.g., "Mr. Kumar has expertise in..." or "He specializes in...")

Recent Conversation Context:
{history_summary}

Résumé Context:
{resume_context}

Respond professionally and concisely."""

    # Prepare messages for API
    api_messages = [{"role": "system", "content": system_prompt}]
    
    # Add recent conversation context
    for msg in recent_messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current user message
    api_messages.append({"role": "user", "content": user_input})

    # Stream response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response_text = ""

        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=api_messages,
                temperature=0.7,
                max_tokens=600,
                top_p=0.9,
                stream=True,
            )
            
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    response_text += delta
                    message_placeholder.markdown(response_text + "▌")
            
            message_placeholder.markdown(response_text)
        
        except Exception as e:
            response_text = f"My apologies, I encountered a difficulty: {str(e)}"
            message_placeholder.markdown(response_text)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Keep conversation history manageable (last 20 messages)
    if len(st.session_state.messages) > 20:
        # Keep first message (greeting) and last 19
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-19:]
