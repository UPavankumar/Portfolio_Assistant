import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
import re

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
    """Initialize ChromaDB client"""
    try:
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
# Intent Classification
# =========================
def classify_intent(query: str) -> str:
    """Classify user intent using keyword matching"""
    query_lower = query.lower()
    
    intent_patterns = {
        "experience": ["work", "job", "company", "companies", "employer", "position", "role", "worked", "career", "employment"],
        "projects": ["project", "built", "created", "developed", "model", "application", "churn", "discord", "bot", "prediction"],
        "skills": ["skill", "technology", "tech stack", "programming", "language", "tool", "knows", "proficient", "expertise", "can he"],
        "education": ["study", "studied", "degree", "college", "university", "education", "graduated", "bachelor"],
        "contact": ["contact", "email", "phone", "reach", "linkedin", "portfolio", "connect", "availability"],
        "summary": ["who is", "about", "overview", "summary", "tell me about", "introduction", "background"],
        "certifications": ["certification", "certified", "certificate", "course", "training"],
        "achievements": ["achievement", "promotion", "award", "recognition", "accomplishment"],
    }
    
    # Score each intent
    scores = {}
    for intent, keywords in intent_patterns.items():
        scores[intent] = sum(2 if kw in query_lower else 0 for kw in keywords)
    
    # Return highest scoring intent
    max_intent = max(scores, key=scores.get)
    return max_intent if scores[max_intent] > 0 else "general"

def llm_classify_intent(query: str) -> str:
    """Fallback: Use LLM for intent classification when rules fail"""
    try:
        intent_prompt = f"""Classify this question into ONE of these categories: experience, projects, skills, education, contact, summary, certifications, achievements, general

Question: "{query}"

Reply with ONLY the category name, nothing else."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": intent_prompt}],
            temperature=0,
            max_tokens=10,
        )
        
        intent = response.choices[0].message.content.strip().lower()
        valid_intents = ["experience", "projects", "skills", "education", "contact", "summary", "certifications", "achievements", "general"]
        
        return intent if intent in valid_intents else "general"
    
    except Exception as e:
        st.warning(f"Intent classification failed: {e}")
        return "general"

# =========================
# Load Resume File
# =========================
resume_path = Path("resume_knowledge_base.md")
if not resume_path.exists():
    st.error("Resume file (resume_knowledge_base.md) not found.")
    st.stop()

try:
    resume_text = resume_path.read_text(encoding="utf-8")
except Exception as e:
    st.error(f"Failed to read resume file: {e}")
    st.stop()

# =========================
# Smart Resume Chunking
# =========================
def parse_resume_sections(content: str) -> dict:
    """Parse resume into sections with metadata"""
    sections = {}
    current_section = "header"
    current_content = []
    
    lines = content.split('\n')
    
    for line in lines:
        # Detect section headers
        if line.startswith('##') and not line.startswith('###'):
            # Save previous section
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            
            # Start new section
            section_name = line.replace('#', '').strip().lower()
            
            # Map to standard intent categories
            if 'experience' in section_name or 'work' in section_name:
                current_section = 'experience'
            elif 'project' in section_name:
                current_section = 'projects'
            elif 'skill' in section_name or 'technical' in section_name:
                current_section = 'skills'
            elif 'education' in section_name:
                current_section = 'education'
            elif 'certification' in section_name:
                current_section = 'certifications'
            elif 'achievement' in section_name:
                current_section = 'achievements'
            elif 'personal' in section_name or 'contact' in section_name:
                current_section = 'contact'
            elif 'summary' in section_name:
                current_section = 'summary'
            else:
                current_section = section_name
            
            current_content = [line]
        else:
            current_content.append(line)
    
    # Save last section
    if current_content:
        sections[current_section] = '\n'.join(current_content)
    
    return sections

# =========================
# Initialize Knowledge Base
# =========================
@st.cache_resource
def initialize_knowledge_base(_chroma_client, _embedding_model, resume_content):
    """Initialize knowledge base with sectioned chunks and metadata"""
    try:
        collection = _chroma_client.get_or_create_collection(
            name="resume_knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        
        if collection.count() == 0:
            sections = parse_resume_sections(resume_content)
            
            all_chunks = []
            all_metadatas = []
            
            for section_name, section_content in sections.items():
                # Split large sections into smaller chunks
                if len(section_content) > 1000:
                    # Split by paragraphs
                    paragraphs = [p.strip() for p in section_content.split('\n\n') if p.strip()]
                    for para in paragraphs:
                        all_chunks.append(para)
                        all_metadatas.append({
                            "section": section_name,
                            "char_count": len(para)
                        })
                else:
                    all_chunks.append(section_content)
                    all_metadatas.append({
                        "section": section_name,
                        "char_count": len(section_content)
                    })
            
            # Generate embeddings
            embeddings = _embedding_model.encode(all_chunks, show_progress_bar=False).tolist()
            
            # Add to collection
            collection.add(
                ids=[f"chunk_{i}" for i in range(len(all_chunks))],
                embeddings=embeddings,
                documents=all_chunks,
                metadatas=all_metadatas
            )
            
            st.success(f"✅ Initialized knowledge base with {len(all_chunks)} chunks across {len(sections)} sections")
        
        return collection
    
    except Exception as e:
        st.error(f"Failed to initialize knowledge base: {e}")
        st.stop()

collection = initialize_knowledge_base(chroma_client, embedding_model, resume_text)

# =========================
# Smart Retrieval Function
# =========================
def smart_search(query: str, intent: str, top_k: int = 4) -> list:
    """Search with intent-based filtering"""
    try:
        query_embedding = embedding_model.encode([query], show_progress_bar=False).tolist()
        
        # If intent is specific, filter by section
        if intent != "general":
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where={"section": intent}
            )
        else:
            # For general queries, search everything
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
# Chat Input Handler
# =========================
if user_input := st.chat_input("Your message to Alfred..."):
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Step 1: Classify Intent
    intent = classify_intent(user_input)
    
    # Fallback to LLM if needed
    if intent == "general" and len(user_input.split()) > 3:
        intent = llm_classify_intent(user_input)
    
    # Step 2: Smart Retrieval
    relevant_chunks = smart_search(user_input, intent, top_k=4)
    
    if relevant_chunks:
        resume_context = "\n\n---\n\n".join(relevant_chunks)
    else:
        resume_context = "No specific information found in the résumé for this query."
    
    # Step 3: Build Enhanced System Prompt
    system_prompt = f"""You are Alfred Pennyworth, Mr. Pavan Kumar's distinguished personal assistant.

Query Intent Detected: {intent.upper()}

Instructions:
- Provide a precise, well-structured answer based ONLY on the résumé context below
- If the query asks for multiple items (e.g., "list all companies"), enumerate them clearly
- If information is missing, state: "I don't have that specific detail in Mr. Kumar's records"
- Maintain a professional, refined tone
- Synthesize information intelligently rather than quoting verbatim

Résumé Context:
{resume_context}

Recent Conversation:
{' '.join([m['content'][:100] for m in st.session_state.messages[-4:-1]])}"""

    # Prepare API messages
    api_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    # Step 4: Stream Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response_text = ""
        
        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=api_messages,
                temperature=0.6,
                max_tokens=700,
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
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Trim history (keep last 20 messages)
    if len(st.session_state.messages) > 20:
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-19:]
