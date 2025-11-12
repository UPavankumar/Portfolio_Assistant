import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
import json
import re
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
import hashlib

# =========================
# Configuration
# =========================
st.set_page_config(
    page_title="💼 Alfred — Pavan Kumar's Personal AI Assistant",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""<style>
.stChatMessage {padding: 1rem; border-radius: 0.5rem;}
</style>""", unsafe_allow_html=True)

# =========================
# Initialize Session State FIRST
# =========================
if 'messages' not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Good day! I'm Alfred, Mr. Pavan Kumar's personal assistant. May I have the pleasure of knowing your name?"
    }]

if 'user_name' not in st.session_state:
    st.session_state.user_name = None

if 'last_intent' not in st.session_state:
    st.session_state.last_intent = None

if 'context_topics' not in st.session_state:
    st.session_state.context_topics = []

if 'interaction_mode' not in st.session_state:
    st.session_state.interaction_mode = "balanced"

if 'session_summary' not in st.session_state:
    st.session_state.session_summary = ""

# =========================
# Header with Reset Button
# =========================
col1, col2 = st.columns([4, 1])
with col1:
    st.title("💼 Alfred — Pavan Kumar's AI Assistant")
with col2:
    if st.button("🔄 Reset", help="Start fresh conversation"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Good day! I'm Alfred, Mr. Pavan Kumar's personal assistant. May I have the pleasure of knowing your name?"
        }]
        st.session_state.user_name = None
        st.session_state.last_intent = None
        st.session_state.context_topics = []
        st.session_state.session_summary = ""
        st.rerun()

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("🎛️ Alfred Controls")
    
    st.subheader("👤 Identity")
    if st.session_state.user_name:
        st.success(f"Hello, {st.session_state.user_name}!")
    else:
        st.info("Name not yet provided")
    
    st.subheader("💬 Mode")
    mode = st.selectbox(
        "Response Style",
        ["concise", "balanced", "detailed"],
        index=["concise", "balanced", "detailed"].index(st.session_state.interaction_mode)
    )
    st.session_state.interaction_mode = mode
    
    st.subheader("📊 Session Info")
    st.caption(f"Messages: {len(st.session_state.messages)-1}")
    if st.session_state.last_intent:
        st.caption(f"Context: {st.session_state.last_intent.title()}")
    if st.session_state.session_summary:
        with st.expander("Session Summary"):
            st.caption(st.session_state.session_summary)

# =========================
# Initialize Groq Client
# =========================
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("⚠️ GROQ_API_KEY not found")
    st.stop()

try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"❌ Failed to initialize Groq: {e}")
    st.stop()

# =========================
# Optional: Initialize Advanced Features
# =========================
@st.cache_resource
def get_embeddings_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except:
        return None

@st.cache_resource
def get_chroma_client():
    try:
        return chromadb.PersistentClient(path="./chroma_db")
    except:
        return None

embedding_model = get_embeddings_model()
chroma_client = get_chroma_client()

# =========================
# Load Resume
# =========================
resume_path = Path("resume_knowledge_base.md")
if resume_path.exists():
    resume_text = resume_path.read_text(encoding="utf-8")
    resume_hash = hashlib.md5(resume_text.encode()).hexdigest()
else:
    resume_text = """**Pavan Kumar's Resume Knowledge Base**
**Personal Information**:
- Name: Pavan Kumar
- Location: Bengaluru, India
- Contact: +91-8050737339, u.pavankumar2002@gmail.com
- LinkedIn: linkedin.com/in/u-pavankumar
- Portfolio: portfolio-u-pavankumar.web.app
**Professional Summary**:
- Business Analyst and Data Professional with expertise in AI automation, workflow integration, database management, and data analytics.
**Skills**:
- Programming: Python, SQL, R, Flutter
- AI & Automation: IBM Watsonx, IBM RPA, UIPath, N8n
- Machine Learning: Scikit-learn, TensorFlow, PyTorch
- Data Visualization: Power BI, Tableau
**Work Experience**:
**Business Analyst** - Envision Beyond, Bengaluru (Oct 2025 - Present)
- Double promoted, skipping Associate level
- AI & Automation with IBM Watsonx and RPA
- Database management with PostgreSQL
- Flutter app development
**Data Analyst Consultant** - Spire Technologies (Sept 2024 - Jan 2025)
- Built data pipelines with Python and SQL
- Designed Power BI dashboards
**Education**:
- B.E. in Computer Science (Data Science), MVJ College of Engineering"""
    resume_hash = ""

# =========================
# Structured Data Extraction (Optional)
# =========================
@st.cache_data
def extract_structured_data(resume_content: str, content_hash: str) -> Dict:
    cache_file = Path("structured_resume_cache.json")
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                if cached.get("hash") == content_hash:
                    return cached.get("data", {})
        except:
            pass
    
    return {
        "experience": [],
        "projects": [],
        "skills": {},
        "education": [],
        "contact": {},
        "summary": ""
    }

structured_data = extract_structured_data(resume_text, resume_hash)

# =========================
# Vector DB (Optional)
# =========================
@st.cache_resource
def get_vector_collection(_chroma_client, _embedding_model, resume_content: str, content_hash: str):
    if not _chroma_client or not _embedding_model:
        return None
    
    collection_name = "alfred_kb"
    
    try:
        collection = _chroma_client.get_collection(name=collection_name)
        if collection.metadata and collection.metadata.get("hash") == content_hash:
            return collection
    except:
        pass
    
    try:
        _chroma_client.delete_collection(collection_name)
    except:
        pass
    
    try:
        collection = _chroma_client.create_collection(
            name=collection_name,
            metadata={"hash": content_hash}
        )
        
        sections = []
        current_section = []
        
        for line in resume_content.split('\n'):
            if line.startswith('**') and line.endswith('**:'):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        if sections:
            embeddings = _embedding_model.encode(sections, show_progress_bar=False).tolist()
            collection.add(
                ids=[f"sec_{i}" for i in range(len(sections))],
                documents=sections,
                embeddings=embeddings
            )
        
        return collection
    except:
        return None

collection = get_vector_collection(chroma_client, embedding_model, resume_text, resume_hash)

# =========================
# Intent Classification
# =========================
def classify_intent(query: str, last_intent: Optional[str]) -> str:
    q = query.lower()
    
    # Follow-ups
    if any(p in q for p in ['tell me more', 'what about', 'also', 'and', 'more']) and last_intent:
        return last_intent
    
    # Keywords
    if any(w in q for w in ['work', 'job', 'company', 'employer', 'experience', 'career']):
        return 'experience'
    if any(w in q for w in ['project', 'built', 'created', 'developed']):
        return 'projects'
    if any(w in q for w in ['skill', 'technology', 'tech', 'programming', 'tool']):
        return 'skills'
    if any(w in q for w in ['education', 'degree', 'college', 'university']):
        return 'education'
    if any(w in q for w in ['contact', 'email', 'phone', 'linkedin', 'reach']):
        return 'contact'
    
    return 'general'

# =========================
# Enhanced Context Search (Optional)
# =========================
def get_enhanced_context(query: str, intent: str) -> str:
    if collection and embedding_model:
        try:
            query_emb = embedding_model.encode([query], show_progress_bar=False).tolist()
            results = collection.query(query_embeddings=query_emb, n_results=2)
            if results and results['documents'] and results['documents'][0]:
                return '\n\n'.join(results['documents'][0])
        except:
            pass
    
    return resume_text[:3000]  # Fallback to full text

# =========================
# Name Extraction
# =========================
def extract_name(user_input: str) -> Optional[str]:
    lower_input = user_input.lower()
    name_indicators = ["my name is", "i'm", "i am", "call me", "this is", "name's", "actually", "it's"]
    
    for indicator in name_indicators:
        if indicator in lower_input:
            parts = lower_input.split(indicator, 1)
            if len(parts) > 1:
                potential_name = parts[1].strip().split()[0] if parts[1].strip() else None
                if potential_name and len(potential_name) > 1:
                    extracted_name = potential_name.capitalize()
                    extracted_name = extracted_name.rstrip('.,!?;:')
                    return extracted_name
    
    return None

# =========================
# Display Messages
# =========================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# Chat Input
# =========================
if user_input := st.chat_input("Your message to Alfred..."):
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Extract name if provided
    extracted_name = extract_name(user_input)
    if extracted_name and extracted_name != st.session_state.user_name:
        st.session_state.user_name = extracted_name
    
    # Classify intent
    intent = classify_intent(user_input, st.session_state.last_intent)
    st.session_state.last_intent = intent
    
    if intent not in st.session_state.context_topics:
        st.session_state.context_topics.append(intent)
    st.session_state.context_topics = st.session_state.context_topics[-5:]
    
    # Get context
    context = get_enhanced_context(user_input, intent)
    
    # Mode instructions
    mode_instructions = {
        "concise": "Keep responses brief (2-3 sentences maximum) unless user asks for details.",
        "balanced": "Provide clear, well-structured responses with key details.",
        "detailed": "Give comprehensive explanations with context and examples."
    }
    
    # Build conversation context
    recent_context = ""
    if len(st.session_state.messages) > 1:
        recent_messages = st.session_state.messages[-7:-1]  # Last 3 exchanges
        if recent_messages:
            recent_context = "\n\nRECENT CONVERSATION:\n" + "\n".join([
                f"{msg['role'].upper()}: {msg['content'][:100]}"
                for msg in recent_messages
            ])
    
    # System prompt
    system_prompt = f"""You are Alfred Pennyworth, Pavan Kumar's esteemed personal assistant. You embody the master strategist and confidant—exceptionally knowledgeable, astute, commanding presence, and possessing deep expertise across business, technology, and professional domains.

CRITICAL INSTRUCTIONS:
{mode_instructions[st.session_state.interaction_mode]}

YOUR CHARACTER:
- Speak with quiet authority and confidence, like a trusted senior advisor
- Use refined British expressions: "indeed," "I dare say," "quite remarkable," "most astute," "precisely"
- Balance warmth with gravitas—approachable yet commanding respect
- Be direct and purposeful; your words carry weight
- NEVER be condescending or dismissive
- Treat every user with utmost dignity and courtesy

CURRENT USER:
Name: {st.session_state.user_name if st.session_state.user_name else "Not yet provided"}
Current Query Intent: {intent.upper()}

NAME TRACKING:
- When user provides their name, acknowledge it warmly and use it naturally
- If they correct their name, gracefully acknowledge: "My apologies, [New Name]. I shall address you correctly from now on."
- Use their name naturally in responses for personalization
{recent_context}

KNOWLEDGE BASE:
{context}

Respond as Alfred would—professional, insightful, and commanding."""

    # Generate response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Build messages for API
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add recent message history (last 6 messages)
            for msg in st.session_state.messages[-7:-1]:
                if msg.get("role") != "system":
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            # Stream response
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=800,
                top_p=0.9,
                stream=True
            )
            
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        except Exception as e:
            full_response = f"My apologies. A slight technical complication has arisen: {str(e)[:100]}. Might I suggest trying again?"
            message_placeholder.markdown(full_response)
    
    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Trim history
    if len(st.session_state.messages) > 30:
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-24:]
    
    # Periodic summarization (every 12 messages)
    if len(st.session_state.messages) > 12 and len(st.session_state.messages) % 12 == 0:
        try:
            old_messages = st.session_state.messages[1:-6]
            if old_messages:
                summary_text = "\n".join([f"{msg['role']}: {msg['content'][:80]}" for msg in old_messages[-10:]])
                
                summary_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{
                        "role": "user",
                        "content": f"Summarize this conversation in 2 sentences:\n{summary_text}"
                    }],
                    temperature=0.3,
                    max_tokens=100
                )
                st.session_state.session_summary = summary_response.choices[0].message.content.strip()
        except:
            pass

# =========================
# Footer
# =========================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"💬 {st.session_state.interaction_mode.title()} Mode")
with col2:
    st.caption(f"📊 {len(st.session_state.messages)-1} messages")
with col3:
    if st.session_state.last_intent:
        st.caption(f"🎯 {st.session_state.last_intent.title()}")
