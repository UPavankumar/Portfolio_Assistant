import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import os
import hashlib

# =========================
# Configuration
# =========================
st.set_page_config(
    page_title="💼 Alfred — Pavan Kumar's AI Assistant",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""<style>
.stChatMessage {padding: 1rem; border-radius: 0.5rem;}
</style>""", unsafe_allow_html=True)

st.title("💼 Alfred — Pavan Kumar's AI Assistant")

# =========================
# Initialize Session State - FIRST THING
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Good day. I am Alfred, Mr. Pavan Kumar's personal AI assistant. How may I assist you today?"
    }]

if "session_summary" not in st.session_state:
    st.session_state.session_summary = ""

if "last_intent" not in st.session_state:
    st.session_state.last_intent = None

if "context_topics" not in st.session_state:
    st.session_state.context_topics = []

if "interaction_mode" not in st.session_state:
    st.session_state.interaction_mode = "balanced"

if "user_name" not in st.session_state:
    st.session_state.user_name = "Guest"

# =========================
# Simple Sidebar
# =========================
with st.sidebar:
    st.header("🎛️ Controls")
    
    user_name = st.text_input("Your Name", value=st.session_state.user_name, key="user_input")
    st.session_state.user_name = user_name
    
    mode = st.selectbox(
        "Mode",
        ["concise", "balanced", "detailed"],
        index=["concise", "balanced", "detailed"].index(st.session_state.interaction_mode),
        key="mode_select"
    )
    st.session_state.interaction_mode = mode
    
    if st.button("🔄 Reset Chat", key="reset_btn"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Good day. I am Alfred, Mr. Pavan Kumar's personal AI assistant. How may I assist you today?"
        }]
        st.session_state.session_summary = ""
        st.session_state.last_intent = None
        st.session_state.context_topics = []
        st.rerun()

# =========================
# Initialize Clients with Error Handling
# =========================
@st.cache_resource
def get_clients():
    try:
        # Groq client
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("⚠️ GROQ_API_KEY not found")
            st.stop()
        groq_client = Groq(api_key=api_key)
        
        # ChromaDB
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Embeddings
        embeddings = SentenceTransformer("all-MiniLM-L6-v2")
        
        return groq_client, chroma_client, embeddings
    
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        st.stop()

client, chroma_client, embedding_model = get_clients()

# =========================
# Load Resume
# =========================
resume_path = Path("resume_knowledge_base.md")
if not resume_path.exists():
    st.error("❌ resume_knowledge_base.md not found")
    st.stop()

resume_text = resume_path.read_text(encoding="utf-8")
resume_hash = hashlib.md5(resume_text.encode()).hexdigest()

# =========================
# Structured Data
# =========================
@st.cache_data
def get_structured_data(resume_content: str, content_hash: str) -> Dict:
    cache_file = Path("structured_resume_cache.json")
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                if cached.get("hash") == content_hash:
                    return cached.get("data", {})
        except:
            pass
    
    try:
        prompt = f"""Extract info from this résumé as JSON:

{{
  "experience": [{{"company": "X", "role": "Y", "duration": "Z", "location": "L", "highlights": ["h1"]}}],
  "projects": [{{"name": "P", "description": "D", "technologies": ["t1"], "achievements": ["a1"]}}],
  "skills": {{"programming": ["Python"], "tools": ["Tool1"]}},
  "education": [{{"degree": "D", "field": "F", "institution": "I", "location": "L"}}],
  "certifications": [{{"name": "C", "issuer": "I", "year": "Y"}}],
  "achievements": ["A1"],
  "contact": {{"name": "N", "email": "E", "phone": "P", "location": "L", "linkedin": "L"}},
  "summary": "Summary text"
}}

RÉSUMÉ:
{resume_content}

JSON only:"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=3000
        )
        
        json_text = response.choices[0].message.content.strip()
        json_text = re.sub(r'^```json\s*', '', json_text)
        json_text = re.sub(r'\s*```$', '', json_text).strip()
        
        data = json.loads(json_text)
        
        with open(cache_file, 'w') as f:
            json.dump({"hash": content_hash, "data": data}, f)
        
        return data
    
    except:
        return {"experience": [], "projects": [], "skills": {}, "education": [], "certifications": [], "achievements": [], "contact": {}, "summary": ""}

structured_data = get_structured_data(resume_text, resume_hash)

# =========================
# Vector DB
# =========================
@st.cache_resource
def get_vector_db(_chroma_client, _embedding_model, resume_content: str, content_hash: str):
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
    
    collection = _chroma_client.create_collection(
        name=collection_name,
        metadata={"hash": content_hash}
    )
    
    # Simple section parsing
    sections = []
    current_section = []
    
    for line in resume_content.split('\n'):
        if line.startswith('##') and not line.startswith('###'):
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

collection = get_vector_db(chroma_client, embedding_model, resume_text, resume_hash)

# =========================
# Intent Classification
# =========================
def classify_intent(query: str, last_intent: Optional[str]) -> str:
    q = query.lower()
    
    if any(p in q for p in ['tell me more', 'what about', 'also', 'and']) and last_intent:
        return last_intent
    
    if any(w in q for w in ['work', 'job', 'company', 'employer', 'role', 'career']):
        return 'experience'
    if any(w in q for w in ['project', 'built', 'created', 'developed']):
        return 'projects'
    if any(w in q for w in ['skill', 'technology', 'tech', 'programming']):
        return 'skills'
    if any(w in q for w in ['education', 'degree', 'college', 'university']):
        return 'education'
    if any(w in q for w in ['certification', 'certificate', 'certified']):
        return 'certifications'
    if any(w in q for w in ['achievement', 'accomplishment', 'award']):
        return 'achievements'
    if any(w in q for w in ['contact', 'email', 'phone', 'linkedin']):
        return 'contact'
    
    return 'summary'

# =========================
# Formatters
# =========================
def format_experience(exp_list: List[Dict]) -> str:
    if not exp_list:
        return "No experience information available."
    result = "**Professional Experience:**\n\n"
    for exp in exp_list:
        result += f"### {exp.get('role', 'N/A')} at {exp.get('company', 'N/A')}\n"
        result += f"📍 {exp.get('location', 'N/A')} | 📅 {exp.get('duration', 'N/A')}\n\n"
        if exp.get('highlights'):
            for h in exp['highlights']:
                result += f"• {h}\n"
            result += "\n"
    return result

def format_projects(proj_list: List[Dict]) -> str:
    if not proj_list:
        return "No projects available."
    result = "**Projects:**\n\n"
    for proj in proj_list:
        result += f"### {proj.get('name', 'N/A')}\n"
        result += f"{proj.get('description', 'N/A')}\n\n"
        if proj.get('technologies'):
            result += f"**Tech:** {', '.join(proj['technologies'])}\n\n"
    return result

def format_skills(skills_dict: Dict) -> str:
    if not skills_dict:
        return "No skills available."
    result = "**Technical Skills:**\n\n"
    for category, skills in skills_dict.items():
        if skills:
            result += f"**{category.replace('_', ' ').title()}:** {', '.join(skills)}\n\n"
    return result

def format_contact(contact_dict: Dict) -> str:
    if not contact_dict:
        return "Contact information not available."
    result = "**Contact:**\n\n"
    result += f"📧 {contact_dict.get('email', 'N/A')}\n"
    result += f"📱 {contact_dict.get('phone', 'N/A')}\n"
    result += f"📍 {contact_dict.get('location', 'N/A')}\n"
    result += f"💼 {contact_dict.get('linkedin', 'N/A')}\n"
    return result

def format_list(items: List) -> str:
    if not items:
        return "No information available."
    result = ""
    for item in items:
        if isinstance(item, dict):
            result += f"• {item.get('name', str(item))}\n"
        else:
            result += f"• {item}\n"
    return result

# =========================
# Search
# =========================
def search(query: str, intent: str, structured: Dict) -> str:
    # Try structured first
    if intent == 'experience' and structured.get('experience'):
        return format_experience(structured['experience'])
    elif intent == 'projects' and structured.get('projects'):
        return format_projects(structured['projects'])
    elif intent == 'skills' and structured.get('skills'):
        return format_skills(structured['skills'])
    elif intent == 'contact' and structured.get('contact'):
        return format_contact(structured['contact'])
    elif intent == 'certifications' and structured.get('certifications'):
        return format_list(structured['certifications'])
    elif intent == 'achievements' and structured.get('achievements'):
        return format_list(structured['achievements'])
    elif intent == 'education' and structured.get('education'):
        return format_list(structured['education'])
    elif intent == 'summary' and structured.get('summary'):
        return structured['summary']
    
    # Vector search
    try:
        query_emb = embedding_model.encode([query], show_progress_bar=False).tolist()
        results = collection.query(query_embeddings=query_emb, n_results=2)
        if results and results['documents'] and results['documents'][0]:
            return '\n\n'.join(results['documents'][0])
    except:
        pass
    
    return "Information not available in records."

# =========================
# Display Messages
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# Chat Input
# =========================
user_input = st.chat_input("Your message...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get intent
    intent = classify_intent(user_input, st.session_state.last_intent)
    st.session_state.last_intent = intent
    
    # Search
    context = search(user_input, intent, structured_data)
    
    # Mode instructions
    mode_inst = {
        "concise": "Be brief. 2-3 sentences max.",
        "balanced": "Clear, structured responses.",
        "detailed": "Comprehensive explanations."
    }
    
    # System prompt
    system_prompt = f"""You are Alfred, Mr. Pavan Kumar's AI assistant.

User: {st.session_state.user_name}
Mode: {mode_inst[st.session_state.interaction_mode]}

Use ONLY this context:
{context}

Be professional and direct."""

    # Generate response
    assistant_message = ""
    
    try:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.6,
                max_tokens=1000,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    assistant_message += chunk.choices[0].delta.content
                    placeholder.markdown(assistant_message + "▌")
            
            placeholder.markdown(assistant_message)
    
    except Exception as e:
        assistant_message = f"I apologize, I'm experiencing a technical issue. Error: {str(e)[:100]}"
        with st.chat_message("assistant"):
            st.markdown(assistant_message)
    
    # Save message
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    
    # Trim history
    if len(st.session_state.messages) > 30:
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-24:]

# =========================
# Footer
# =========================
st.markdown("---")
st.caption(f"💬 {st.session_state.interaction_mode.title()} Mode | 📊 {len(st.session_state.messages)-1} messages")
