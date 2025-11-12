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
    page_title="💼 Alfred — Pavan Kumar's Personal AI Assistant",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stChatMessage {padding: 1rem; border-radius: 0.5rem;}
    .mode-indicator {
        position: fixed;
        top: 60px;
        right: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)

# Mode indicator badge
if 'interaction_mode' in st.session_state:
    st.markdown(f'<div class="mode-indicator">💬 {st.session_state.interaction_mode.title()} Mode</div>', unsafe_allow_html=True)

st.title("💼 Alfred — Pavan Kumar's Personal AI Assistant")

# =========================
# Error Logging
# =========================
ERROR_LOG = Path("error_log.txt")

def log_error(error_type: str, message: str):
    """Silent error logging for debugging"""
    try:
        with open(ERROR_LOG, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().isoformat()
            f.write(f"[{timestamp}] {error_type}: {message}\n")
    except:
        pass

# =========================
# Cache Management with Size Control
# =========================
CACHE_FILE = Path("alfred_session_cache.json")
CACHE_MAX_SIZE_MB = 1
CACHE_EXPIRY_HOURS = 48

class SessionCache:
    @staticmethod
    def save_cache(data: Dict):
        try:
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            # Size check
            if CACHE_FILE.stat().st_size > CACHE_MAX_SIZE_MB * 1024 * 1024:
                SessionCache.clear_cache()
        except Exception as e:
            log_error("CACHE_SAVE", str(e))
    
    @staticmethod
    def load_cache() -> Optional[Dict]:
        try:
            if not CACHE_FILE.exists():
                return None
            
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            cache_time = datetime.fromisoformat(cache_data["timestamp"])
            if datetime.now() - cache_time > timedelta(hours=CACHE_EXPIRY_HOURS):
                SessionCache.clear_cache()
                return None
            
            return cache_data["data"]
        except Exception as e:
            log_error("CACHE_LOAD", str(e))
            return None
    
    @staticmethod
    def clear_cache():
        try:
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
        except Exception as e:
            log_error("CACHE_CLEAR", str(e))

# =========================
# Conversation Summarization
# =========================
def summarize_old_context(messages: List[Dict], client: Groq) -> str:
    """Compress older conversation turns into summary"""
    if len(messages) <= 10:
        return ""
    
    # Take messages from position 1 (skip greeting) to -6 (keep last 3 turns)
    old_messages = messages[1:-6]
    
    if not old_messages:
        return ""
    
    conversation_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content'][:150]}" 
        for msg in old_messages
    ])
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""Summarize this conversation in 2-3 concise sentences. Focus on:
- Topics discussed
- Information requested
- User's focus areas

Conversation:
{conversation_text}

Summary (2-3 sentences):"""
            }],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log_error("SUMMARIZATION", str(e))
        return ""

# =========================
# Initialize Session State
# =========================
if "messages" not in st.session_state:
    cached_data = SessionCache.load_cache()
    
    if cached_data and "messages" in cached_data:
        st.session_state.messages = cached_data["messages"]
        st.session_state.session_summary = cached_data.get("session_summary", "")
        st.session_state.last_intent = cached_data.get("last_intent", None)
        st.session_state.context_topics = cached_data.get("context_topics", [])
        st.session_state.active_goal = cached_data.get("active_goal", None)
        st.session_state.interaction_mode = cached_data.get("interaction_mode", "balanced")
        st.session_state.user_name = cached_data.get("user_name", "Guest")
        st.session_state.user_relationship = cached_data.get("user_relationship", "visitor")
    else:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Good day. I am Alfred, Mr. Pavan Kumar's personal AI assistant. How may I assist you today?"
        }]
        st.session_state.session_summary = ""
        st.session_state.last_intent = None
        st.session_state.context_topics = []
        st.session_state.active_goal = None
        st.session_state.interaction_mode = "balanced"
        st.session_state.user_name = "Guest"
        st.session_state.user_relationship = "visitor"

# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.header("🎛️ Alfred Controls")
    
    st.subheader("👤 Identity")
    prev_name = st.session_state.user_name
    user_name = st.text_input("Your Name", value=st.session_state.user_name)
    
    if user_name != prev_name:
        st.session_state.user_name = user_name
        
        # Identity change detection
        if user_name.lower() in ["pavan", "pavan kumar"]:
            st.session_state.user_relationship = "owner"
            st.success("✅ Owner verified")
        else:
            st.session_state.user_relationship = "visitor"
        
        # Reset context on identity change
        if prev_name != "Guest":
            st.session_state.context_topics = []
            st.session_state.last_intent = None
    
    st.caption(f"Role: {st.session_state.user_relationship.title()}")
    
    st.subheader("💬 Mode")
    mode = st.selectbox(
        "Response Style",
        ["concise", "balanced", "detailed", "builder"],
        index=["concise", "balanced", "detailed", "builder"].index(st.session_state.interaction_mode)
    )
    if mode != st.session_state.interaction_mode:
        st.session_state.interaction_mode = mode
    
    st.subheader("🎯 Active Goal")
    if st.session_state.active_goal:
        st.info(st.session_state.active_goal)
        if st.button("✓ Complete Goal"):
            st.session_state.active_goal = None
            st.rerun()
    else:
        st.caption("No active goal")
    
    st.subheader("💾 Session")
    
    if st.session_state.session_summary:
        with st.expander("📝 Session Summary"):
            st.caption(st.session_state.session_summary)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save"):
            cache_data = {
                "messages": st.session_state.messages,
                "session_summary": st.session_state.session_summary,
                "last_intent": st.session_state.last_intent,
                "context_topics": st.session_state.context_topics,
                "active_goal": st.session_state.active_goal,
                "interaction_mode": st.session_state.interaction_mode,
                "user_name": st.session_state.user_name,
                "user_relationship": st.session_state.user_relationship
            }
            SessionCache.save_cache(cache_data)
            st.success("✅")
    
    with col2:
        if st.button("🔄 Reset"):
            SessionCache.clear_cache()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    DEBUG = st.checkbox("🔧 Debug", value=False)
    
    if DEBUG:
        st.json({
            "messages": len(st.session_state.messages),
            "summary_exists": bool(st.session_state.session_summary),
            "last_intent": st.session_state.last_intent,
            "context_topics": st.session_state.context_topics[-3:],
            "relationship": st.session_state.user_relationship
        })

# =========================
# Initialize Clients
# =========================
@st.cache_resource
def get_groq_client():
    try:
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("⚠️ GROQ_API_KEY not found in secrets")
            st.stop()
        return Groq(api_key=api_key)
    except Exception as e:
        log_error("GROQ_INIT", str(e))
        st.error(f"❌ Could not initialize AI service. Please check configuration.")
        st.stop()

@st.cache_resource
def get_chroma_client():
    try:
        return chromadb.PersistentClient(path="./chroma_db")
    except Exception as e:
        log_error("CHROMA_INIT", str(e))
        st.error("❌ Vector database initialization failed")
        st.stop()

@st.cache_resource
def get_embeddings_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        log_error("EMBEDDINGS_INIT", str(e))
        st.error("❌ Embeddings model failed to load")
        st.stop()

client = get_groq_client()
chroma_client = get_chroma_client()
embedding_model = get_embeddings_model()

# =========================
# Load Resume
# =========================
resume_path = Path("resume_knowledge_base.md")
if not resume_path.exists():
    st.error("❌ resume_knowledge_base.md not found. Please add the file to continue.")
    st.stop()

resume_text = resume_path.read_text(encoding="utf-8")

# Create hash of resume for change detection
resume_hash = hashlib.md5(resume_text.encode()).hexdigest()

# =========================
# Extract Structured Data
# =========================
@st.cache_data
def extract_structured_data(resume_content: str, content_hash: str) -> Dict:
    """Extract with cache invalidation on resume change"""
    cache_file = Path("structured_resume_cache.json")
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                if cached.get("hash") == content_hash:
                    return cached.get("data", {})
        except Exception as e:
            log_error("STRUCTURED_CACHE_LOAD", str(e))
    
    # Extract fresh
    try:
        extraction_prompt = f"""Extract structured information from this résumé and return ONLY valid JSON:

{{
  "experience": [{{"company": "Company", "role": "Role", "duration": "Duration", "location": "Location", "highlights": ["highlight1"]}}],
  "projects": [{{"name": "Project", "description": "Description", "technologies": ["tech1"], "achievements": ["achievement1"]}}],
  "skills": {{"programming": ["Python"], "ai_ml": ["TensorFlow"], "tools": ["Power BI"], "databases": ["PostgreSQL"]}},
  "education": [{{"degree": "Degree", "field": "Field", "institution": "Institution", "location": "Location"}}],
  "certifications": [{{"name": "Cert", "issuer": "Issuer", "year": "Year"}}],
  "achievements": ["Achievement 1"],
  "contact": {{"name": "Name", "email": "email", "phone": "phone", "location": "City", "linkedin": "url", "portfolio": "url"}},
  "summary": "Summary"
}}

RÉSUMÉ:
{resume_content}

Return ONLY JSON."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0,
            max_tokens=3000
        )
        
        json_text = response.choices[0].message.content.strip()
        json_text = re.sub(r'^```json\s*', '', json_text)
        json_text = re.sub(r'\s*```$', '', json_text).strip()
        
        structured_data = json.loads(json_text)
        
        # Cache with hash
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({"hash": content_hash, "data": structured_data}, f, indent=2)
        
        return structured_data
    
    except Exception as e:
        log_error("STRUCTURED_EXTRACT", str(e))
        return {
            "experience": [], "projects": [], "skills": {},
            "education": [], "certifications": [], "achievements": [],
            "contact": {}, "summary": ""
        }

structured_data = extract_structured_data(resume_text, resume_hash)

# =========================
# Vector Database (Persistent)
# =========================
@st.cache_resource
def initialize_vector_db(_chroma_client, _embedding_model, resume_content: str, content_hash: str):
    """Initialize once, reuse unless resume changes"""
    collection_name = "alfred_kb_persistent"
    
    try:
        # Check if collection exists and is current
        try:
            collection = _chroma_client.get_collection(name=collection_name)
            metadata = collection.metadata or {}
            
            if metadata.get("resume_hash") == content_hash:
                # Collection is current, reuse it
                return collection, []
        except:
            pass
        
        # Delete old collection
        try:
            _chroma_client.delete_collection(collection_name)
        except:
            pass
        
        # Create fresh collection
        collection = _chroma_client.create_collection(
            name=collection_name,
            metadata={"resume_hash": content_hash}
        )
        
        # Parse sections
        sections = {}
        current_section = "header"
        current_content = []
        
        for line in resume_content.split('\n'):
            if line.startswith('##') and not line.startswith('###'):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                section_name = line.replace('#', '').strip().lower()
                
                if 'experience' in section_name or 'work' in section_name:
                    current_section = 'experience'
                elif 'project' in section_name:
                    current_section = 'projects'
                elif 'skill' in section_name:
                    current_section = 'skills'
                elif 'education' in section_name:
                    current_section = 'education'
                elif 'certification' in section_name:
                    current_section = 'certifications'
                elif 'achievement' in section_name:
                    current_section = 'achievements'
                elif 'contact' in section_name:
                    current_section = 'contact'
                elif 'summary' in section_name:
                    current_section = 'summary'
                else:
                    current_section = section_name
                
                current_content = [line]
            else:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        # Add to collection
        all_docs, all_metas, all_ids = [], [], []
        
        for section, content in sections.items():
            if content.strip():
                all_docs.append(content)
                all_metas.append({"section": section})
                all_ids.append(f"{section}_chunk")
        
        if all_docs:
            embeddings = _embedding_model.encode(all_docs, show_progress_bar=False).tolist()
            
            collection.add(
                ids=all_ids,
                documents=all_docs,
                embeddings=embeddings,
                metadatas=all_metas
            )
        
        return collection, list(sections.keys())
    
    except Exception as e:
        log_error("VECTOR_DB_INIT", str(e))
        st.error("❌ Vector database setup failed")
        st.stop()

collection, available_sections = initialize_vector_db(chroma_client, embedding_model, resume_text, resume_hash)

# =========================
# Enhanced Intent Classification with Confidence
# =========================
def classify_intent_with_confidence(query: str, last_intent: Optional[str], context_topics: List[str]) -> Tuple[str, float, bool]:
    """Returns (intent, confidence, needs_clarification)"""
    q = query.lower()
    
    # Follow-ups - high confidence
    follow_up_patterns = ['tell me more', 'what about', 'and', 'also', 'what else', 'more', 'continue']
    if any(p in q for p in follow_up_patterns) and last_intent:
        return last_intent, 1.0, False
    
    # Strong keyword matching with scoring
    scores = {
        'experience': 0,
        'projects': 0,
        'skills': 0,
        'education': 0,
        'certifications': 0,
        'achievements': 0,
        'contact': 0,
        'summary': 0
    }
    
    # Experience keywords
    if any(w in q for w in ['work', 'job', 'company', 'employer', 'role', 'position', 'career']):
        scores['experience'] += 3
    if 'worked' in q:
        scores['experience'] += 2
    
    # Projects keywords
    if any(w in q for w in ['project', 'built', 'created', 'developed', 'application']):
        scores['projects'] += 3
    
    # Skills keywords
    if any(w in q for w in ['skill', 'technology', 'tech', 'programming', 'language', 'tool']):
        scores['skills'] += 3
    
    # Education keywords
    if any(w in q for w in ['education', 'degree', 'college', 'university', 'studied']):
        scores['education'] += 3
    
    # Certifications keywords
    if any(w in q for w in ['certification', 'certificate', 'certified', 'course']):
        scores['certifications'] += 3
    
    # Achievements keywords
    if any(w in q for w in ['achievement', 'accomplishment', 'award']):
        scores['achievements'] += 3
    
    # Contact keywords
    if any(w in q for w in ['contact', 'email', 'phone', 'reach', 'linkedin']):
        scores['contact'] += 3
    
    # Summary keywords
    if any(w in q for w in ['who', 'about', 'overview', 'summary', 'background', 'introduce']):
        scores['summary'] += 2
    
    # Context boost
    for topic in context_topics[-2:]:
        if topic in scores:
            scores[topic] += 1
    
    # Get max score
    max_intent = max(scores.items(), key=lambda x: x[1])
    max_score = max_intent[1]
    
    if max_score >= 3:
        # High confidence
        return max_intent[0], 0.9, False
    elif max_score >= 1:
        # Medium confidence
        return max_intent[0], 0.6, False
    elif context_topics:
        # Use context
        return context_topics[-1], 0.5, False
    else:
        # Low confidence - needs clarification
        return 'summary', 0.3, True

# =========================
# Formatters with Validation
# =========================
def format_experience(exp_list: List[Dict]) -> str:
    if not exp_list:
        return "That information is not specified in Mr. Kumar's records."
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
        return "That information is not specified in Mr. Kumar's records."
    result = "**Projects:**\n\n"
    for proj in proj_list:
        result += f"### {proj.get('name', 'N/A')}\n{proj.get('description', 'N/A')}\n\n"
        if proj.get('technologies'):
            result += f"**Technologies:** {', '.join(proj['technologies'])}\n\n"
        if proj.get('achievements'):
            for a in proj['achievements']:
                result += f"• {a}\n"
            result += "\n"
    return result

def format_skills(skills_dict: Dict) -> str:
    if not skills_dict:
        return "That information is not specified in Mr. Kumar's records."
    result = "**Technical Skills:**\n\n"
    for category, skills in skills_dict.items():
        if skills:
            result += f"**{category.replace('_', ' ').title()}:** {', '.join(skills)}\n\n"
    return result

def format_contact(contact_dict: Dict) -> str:
    if not contact_dict:
        return "That information is not specified in Mr. Kumar's records."
    result = "**Contact Information:**\n\n"
    result += f"📧 {contact_dict.get('email', 'N/A')}\n"
    result += f"📱 {contact_dict.get('phone', 'N/A')}\n"
    result += f"📍 {contact_dict.get('location', 'N/A')}\n"
    result += f"💼 {contact_dict.get('linkedin', 'N/A')}\n"
    result += f"🌐 {contact_dict.get('portfolio', 'N/A')}\n"
    return result

def format_certifications(cert_list: List) -> str:
    if not cert_list:
        return "That information is not specified in Mr. Kumar's records."
    result = "**Certifications:**\n\n"
    for cert in cert_list:
        if isinstance(cert, dict):
            result += f"• **{cert.get('name', 'N/A')}** — {cert.get('issuer', 'N/A')}"
            if cert.get('year'):
                result += f" ({cert['year']})"
            result += "\n"
        else:
            result += f"• {cert}\n"
    return result

def format_achievements(ach_list: List[str]) -> str:
    if not ach_list:
        return "That information is not specified in Mr. Kumar's records."
    result = "**Achievements:**\n\n"
    for ach in ach_list:
        result += f"• {ach}\n"
    return result

def format_education(edu_list: List[Dict]) -> str:
    if not edu_list:
        return "That information is not specified in Mr. Kumar's records."
    result = "**Education:**\n\n"
    for edu in edu_list:
        result += f"• **{edu.get('degree', 'N/A')}** in {edu.get('field', 'N/A')}\n"
        result += f"  {edu.get('institution', 'N/A')}, {edu.get('location', 'N/A')}\n\n"
    return result

# =========================
# Intelligent Search with Metadata Filtering
# =========================
def intelligent_search(query: str, intent: str, structured: Dict) -> Tuple[str, str]:
    """Search with intent-based filtering"""
    
    # Try structured data first (Tier 1)
    if intent == 'experience' and structured.get('experience'):
        return format_experience(structured['experience']), 'structured'
    elif intent == 'projects' and structured.get('projects'):
        return format_projects(structured['projects']), 'structured'
    elif intent == 'skills' and structured.get('skills'):
        return format_skills(structured['skills']), 'structured'
    elif intent == 'contact' and structured.get('contact'):
        return format_contact(structured['contact']), 'structured'
    elif intent == 'certifications' and structured.get('certifications'):
        return format_certifications(structured['certifications']), 'structured'
    elif intent == 'achievements' and structured.get('achievements'):
        return format_achievements(structured['achievements']), 'structured'
    elif intent == 'education' and structured.get('education'):
        return format_education(structured['education']), 'structured'
    elif intent == 'summary' and structured.get('summary'):
        return structured['summary'], 'structured'
    
    # Fallback to vector search with metadata filtering (Tier 2)
    try:
        query_embedding = embedding_model.encode([query], show_progress_bar=False).tolist()
        
        # Filter by intent section if available
        if intent in available_sections:
            results = collection.query(
                query_embeddings=query_embedding, 
                n_results=2,
                where={"section": intent}
            )
        else:
            results = collection.query(query_embeddings=query_embedding, n_results=2)
        
        if results and results['documents'] and results['documents'][0]:
            return '\n\n'.join(results['documents'][0]), 'vector'
        
        return "That specific information is not in Mr. Kumar's records.", 'none'
    
    except Exception as e:
        log_error("VECTOR_SEARCH", str(e))
        # Graceful degradation - return structured fallback
        return "I'm experiencing a technical issue with search. Let me provide what I can from structured data.", 'error'

# =========================
# Build Windowed Context
# =========================
def build_windowed_context(messages: List[Dict], session_summary: str) -> str:
    """Build context from summary + last 3 turns"""
    context_parts = []
    
    if session_summary:
        context_parts.append(f"PREVIOUS DISCUSSION:\n{session_summary}\n")
    
    if len(messages) > 1:
        recent_messages = messages[-7:-1]  # Last 3 exchanges (6 messages)
        if recent_messages:
            context_parts.append("RECENT CONVERSATION:")
            for msg in recent_messages:
                role = "USER" if msg['role'] == 'user' else "ALFRED"
                context_parts.append(f"{role}: {msg['content'][:120]}")
    
    return "\n".join(context_parts)

# =========================
# Display Messages
# =========================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# Main Chat Handler
# =========================
if user_input := st.chat_input("Your message..."):
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Classify intent with confidence
    intent, confidence, needs_clarification = classify_intent_with_confidence(
        user_input, 
        st.session_state.last_intent,
        st.session_state.context_topics
    )
    
    # Handle low confidence with clarification
    if needs_clarification and not st.session_state.context_topics:
        clarification = "I want to ensure I provide the most relevant information. Are you asking about:\n\n• His **work experience**\n• His **projects**\n• His **technical skills**\n• His **education**\n• **Contact information**\n• A general **overview**\n\nPlease let me know which interests you."
        
        st.session_state.messages.append({"role": "assistant", "content": clarification})
        with st.chat_message("assistant"):
            st.markdown(clarification)
        st.stop()
    
    # Update context
    st.session_state.last_intent = intent
    if intent not in st.session_state.context_topics:
        st.session_state.context_topics.append(intent)
    st.session_state.context_topics = st.session_state.context_topics[-5:]
    
    # Get context
    context, source = intelligent_search(user_input, intent, structured_data)
    
    # Build conversation context
    conversation_context = build_windowed_context(
        st.session_state.messages, 
        st.session_state.session_summary
    )
    
    # Mode settings
    mode_map = {
        "concise": "Be brief and direct. Maximum 2-3 sentences.",
        "balanced": "Provide clear, well-structured responses with key details.",
        "detailed": "Give comprehensive explanations with context and examples.",
        "builder": "Focus on technical implementation details, step-by-step guidance, and actionable insights."
    }
    
    # Identity reinforcement
    identity_context = f"""You are Alfred Pennyworth, the distinguished personal AI assistant to Mr. Pavan Kumar.

CURRENT USER:
- Name: {st.session_state.user_name}
- Relationship: {st.session_state.user_relationship.title()}
- Speaking to: {"Mr. Kumar himself" if st.session_state.user_relationship == "owner" else "A visitor inquiring about Mr. Kumar"}"""

    if st.session_state.active_goal:
        identity_context += f"\n- Active Goal: {st.session_state.active_goal}"
    
    # System prompt with full awareness
    system_prompt = f"""{identity_context}

{conversation_context}

CURRENT QUERY:
- Intent: {intent.upper()}
- Confidence: {confidence:.1f}
- Mode: {mode_map[st.session_state.interaction_mode]}

CORE INSTRUCTIONS:
1. Answer using ONLY the CONTEXT below - never fabricate information
2. Maintain the refined, professional tone of Alfred Pennyworth
3. If information is missing from context, state: "That detail is not specified in Mr. Kumar's records"
4. Use markdown formatting for clarity and readability
5. Reference the conversation history naturally when relevant
6. Stay in character - you are Mr. Kumar's trusted assistant

CONTEXT FOR THIS QUERY:
{context}

Now respond to the user's query with precision and professionalism."""

    # Generate response
    if source == 'structured':
        # Direct structured response
        response_text = context
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)
    
    else:
        # LLM synthesis with error handling tiers
        try:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                response_text = ""
                
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.6,
                    max_tokens=1200,
                    stream=True,
                )
                
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content
                        placeholder.markdown(response_text + "▌")
                
                placeholder.markdown(response_text)
        
        except Exception as e:
            error_msg = str(e).lower()
            log_error("LLM_GENERATION", str(e))
            
            # Tiered error handling
            if "rate" in error_msg or "limit" in error_msg:
                # Soft error - rate limit
                response_text = "My apologies, the AI service is experiencing high demand at the moment. Please try again in a few seconds."
            elif "timeout" in error_msg or "connection" in error_msg:
                # Medium error - connection issue
                response_text = "I'm experiencing a temporary connection issue. Please try your query again."
            else:
                # Try fallback to structured data
                if source == 'vector' or source == 'none':
                    response_text = "I'm experiencing a technical difficulty with my language processing. However, here's what I found in the structured records:\n\n" + context
                else:
                    response_text = "I encountered a technical issue. Please rephrase your question or try again."
            
            with st.chat_message("assistant"):
                st.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Post-response validation
    if source == 'vector' and response_text and len(response_text) > 50:
        # Check if response seems to use the context
        context_keywords = set(context.lower().split()[:20])
        response_keywords = set(response_text.lower().split()[:30])
        overlap = context_keywords.intersection(response_keywords)
        
        if len(overlap) < 2:
            # Low overlap - possible hallucination
            log_error("VALIDATION_WARNING", f"Low context overlap for intent: {intent}")
    
    # Periodic summarization (every 10 messages after 15 total)
    if len(st.session_state.messages) > 15 and len(st.session_state.messages) % 10 == 0:
        try:
            st.session_state.session_summary = summarize_old_context(
                st.session_state.messages, 
                client
            )
        except Exception as e:
            log_error("SUMMARIZATION_PERIODIC", str(e))
    
    # Trim message history (keep greeting + last 20 messages)
    if len(st.session_state.messages) > 25:
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-20:]
    
    # Auto-save session
    try:
        cache_data = {
            "messages": st.session_state.messages,
            "session_summary": st.session_state.session_summary,
            "last_intent": st.session_state.last_intent,
            "context_topics": st.session_state.context_topics,
            "active_goal": st.session_state.active_goal,
            "interaction_mode": st.session_state.interaction_mode,
            "user_name": st.session_state.user_name,
            "user_relationship": st.session_state.user_relationship
        }
        SessionCache.save_cache(cache_data)
    except Exception as e:
        log_error("AUTO_SAVE", str(e))

# =========================
# Footer Status
# =========================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.session_state.session_summary:
        st.caption("💾 Memory Active")
    else:
        st.caption("🆕 Fresh Session")

with col2:
    msg_count = len(st.session_state.messages) - 1
    st.caption(f"📊 {msg_count} messages")

with col3:
    if st.session_state.last_intent:
        st.caption(f"🎯 Context: {st.session_state.last_intent.title()}")
    else:
        st.caption("🎯 No context")

with col4:
    st.caption(f"👤 {st.session_state.user_relationship.title()}")
