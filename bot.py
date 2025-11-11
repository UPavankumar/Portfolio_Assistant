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
    .structured-info {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;}
    .mode-badge {background-color: #0066cc; color: white; padding: 0.3rem 0.8rem; border-radius: 1rem; font-size: 0.85rem;}
    .goal-badge {background-color: #28a745; color: white; padding: 0.3rem 0.8rem; border-radius: 1rem; font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)

st.title("💼 Alfred — Pavan Kumar's Personal AI Assistant")

# =========================
# Cache Management System
# =========================
CACHE_FILE = Path("alfred_session_cache.json")
CACHE_EXPIRY_HOURS = 48

class SessionCache:
    @staticmethod
    def save_cache(data: Dict):
        """Save session data to persistent cache"""
        try:
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            st.warning(f"⚠️ Cache save failed: {e}")
    
    @staticmethod
    def load_cache() -> Optional[Dict]:
        """Load session data from cache if not expired"""
        try:
            if not CACHE_FILE.exists():
                return None
            
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check expiry
            cache_time = datetime.fromisoformat(cache_data["timestamp"])
            if datetime.now() - cache_time > timedelta(hours=CACHE_EXPIRY_HOURS):
                return None
            
            return cache_data["data"]
        except Exception as e:
            return None
    
    @staticmethod
    def clear_cache():
        """Clear the cache file"""
        try:
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
        except Exception as e:
            pass

# =========================
# Initialize Session State
# =========================
def init_session_state():
    """Initialize or restore session state with memory"""
    
    if "initialized" not in st.session_state:
        cached_data = SessionCache.load_cache()
        
        if cached_data:
            st.session_state.update(cached_data)
            st.session_state.cache_restored = True
        else:
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Good day. I am Alfred, Mr. Pavan Kumar's personal AI assistant. I have his complete professional dossier at hand. How may I assist you today?"
            }]
            st.session_state.conversation_summary = ""
            st.session_state.user_identity = {
                "name": "Guest",
                "relationship": "visitor",
                "verified": False
            }
            st.session_state.last_intent = None
            st.session_state.active_goal = None
            st.session_state.interaction_mode = "balanced"
            st.session_state.user_preferences = {
                "tone": "professional",
                "verbosity": "medium"
            }
            st.session_state.context_topics = []
            st.session_state.cache_restored = False
        
        st.session_state.initialized = True

init_session_state()

# Sidebar Controls
with st.sidebar:
    st.header("🎛️ Alfred Controls")
    
    # User Identity
    st.subheader("👤 User Identity")
    user_name = st.text_input("Your Name", value=st.session_state.user_identity.get("name", "Guest"))
    if user_name != st.session_state.user_identity["name"]:
        st.session_state.user_identity["name"] = user_name
        if user_name.lower() in ["pavan", "pavan kumar"]:
            st.session_state.user_identity["relationship"] = "owner"
            st.session_state.user_identity["verified"] = True
            st.success("✅ Owner verified")
    
    # Interaction Mode
    st.subheader("💬 Interaction Mode")
    mode = st.selectbox(
        "Response Style",
        ["concise", "balanced", "detailed", "builder"],
        index=["concise", "balanced", "detailed", "builder"].index(st.session_state.interaction_mode)
    )
    if mode != st.session_state.interaction_mode:
        st.session_state.interaction_mode = mode
    
    # Active Goal
    st.subheader("🎯 Active Goal")
    if st.session_state.active_goal:
        st.markdown(f'<div class="goal-badge">🎯 {st.session_state.active_goal}</div>', unsafe_allow_html=True)
        if st.button("Clear Goal"):
            st.session_state.active_goal = None
            st.rerun()
    else:
        st.info("No active goal")
    
    # Session Management
    st.subheader("💾 Session Management")
    
    if st.session_state.cache_restored:
        st.success("✅ Session restored")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save"):
            cache_data = {
                "messages": st.session_state.messages,
                "conversation_summary": st.session_state.conversation_summary,
                "user_identity": st.session_state.user_identity,
                "last_intent": st.session_state.last_intent,
                "active_goal": st.session_state.active_goal,
                "interaction_mode": st.session_state.interaction_mode,
                "user_preferences": st.session_state.user_preferences,
                "context_topics": st.session_state.context_topics
            }
            SessionCache.save_cache(cache_data)
            st.success("✅ Saved!")
    
    with col2:
        if st.button("🔄 Reset"):
            SessionCache.clear_cache()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Debug Mode
    DEBUG_MODE = st.checkbox("🔧 Debug", value=False)

# =========================
# Initialize Clients
# =========================
try:
    # Try secrets first, then environment variable
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found. Please add it to .streamlit/secrets.toml or environment variables.")
        st.stop()
    
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"❌ Failed to initialize Groq: {e}")
    st.stop()

@st.cache_resource
def get_chroma_client():
    try:
        return chromadb.PersistentClient(path="./chroma_db")
    except Exception as e:
        st.error(f"❌ ChromaDB initialization failed: {e}")
        st.stop()

@st.cache_resource
def get_embeddings_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"❌ Embeddings model failed: {e}")
        st.stop()

chroma_client = get_chroma_client()
embedding_model = get_embeddings_model()

# =========================
# Load Resume
# =========================
resume_path = Path("resume_knowledge_base.md")
if not resume_path.exists():
    st.error("❌ resume_knowledge_base.md not found in root directory.")
    st.info("Please ensure the file exists in the same folder as bot.py")
    st.stop()

resume_text = resume_path.read_text(encoding="utf-8")

# =========================
# Structured Data Extraction
# =========================
@st.cache_data
def extract_structured_data(_client, resume_content: str) -> Dict:
    """Extract structured information from resume using LLM"""
    
    extraction_prompt = f"""Extract structured information from this résumé and return ONLY valid JSON:

{{
  "experience": [
    {{"company": "Company Name", "role": "Job Title", "duration": "Start - End", "location": "City, Country", "highlights": ["achievement 1", "achievement 2"]}}
  ],
  "projects": [
    {{"name": "Project Name", "description": "Brief description", "technologies": ["tech1", "tech2"], "achievements": ["achievement 1"]}}
  ],
  "skills": {{
    "programming": ["Python", "SQL"],
    "ai_ml": ["TensorFlow"],
    "tools": ["Power BI"],
    "databases": ["PostgreSQL"],
    "automation": ["RPA"]
  }},
  "education": [
    {{"degree": "Degree", "field": "Field", "institution": "Institution", "location": "City"}}
  ],
  "certifications": [
    {{"name": "Cert Name", "issuer": "Issuer", "year": "Year"}}
  ],
  "achievements": ["Achievement 1", "Achievement 2"],
  "contact": {{
    "name": "Name", "email": "email", "phone": "phone", "location": "City", "linkedin": "url", "portfolio": "url"
  }},
  "summary": "Professional summary"
}}

RÉSUMÉ:
{resume_content}

Return ONLY JSON, no markdown."""

    cache_file = Path("structured_resume_cache.json")
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    
    try:
        with st.spinner("🔄 Analyzing résumé..."):
            response = _client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0,
                max_tokens=3000
            )
        
        json_text = response.choices[0].message.content.strip()
        json_text = re.sub(r'^```json\s*', '', json_text)
        json_text = re.sub(r'\s*```$', '', json_text).strip()
        
        structured_data = json.loads(json_text)
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2)
        
        return structured_data
    
    except Exception as e:
        st.warning(f"⚠️ Extraction failed: {e}")
        return {
            "experience": [], "projects": [], "skills": {},
            "education": [], "certifications": [], "achievements": [],
            "contact": {}, "summary": ""
        }

structured_data = extract_structured_data(client, resume_text)

# =========================
# Vector Database
# =========================
@st.cache_resource
def initialize_vector_db(_chroma_client, _embedding_model, resume_content: str):
    collection_name = "alfred_kb_v5"
    
    try:
        try:
            _chroma_client.delete_collection(collection_name)
        except:
            pass
        
        collection = _chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
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
                elif 'contact' in section_name or 'personal' in section_name:
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
        
        all_docs, all_metas, all_ids = [], [], []
        
        for section, content in sections.items():
            if content.strip():
                all_docs.append(content)
                all_metas.append({"section": section})
                all_ids.append(f"{section}_chunk")
        
        embeddings = _embedding_model.encode(all_docs, show_progress_bar=False).tolist()
        
        collection.add(
            ids=all_ids,
            documents=all_docs,
            embeddings=embeddings,
            metadatas=all_metas
        )
        
        return collection, list(sections.keys())
    
    except Exception as e:
        st.error(f"❌ Vector DB failed: {e}")
        st.stop()

collection, available_sections = initialize_vector_db(chroma_client, embedding_model, resume_text)

# =========================
# Intent Classification
# =========================
def classify_intent_with_context(query: str, last_intent: Optional[str], context_topics: List[str]) -> Tuple[str, float]:
    """Classify intent considering conversation context"""
    q = query.lower()
    
    follow_up_patterns = ['tell me more', 'what about', 'and the', 'how about', 'also', 'additionally', 'what else']
    is_follow_up = any(pattern in q for pattern in follow_up_patterns)
    
    if is_follow_up and last_intent:
        return last_intent, 0.9
    
    patterns = {
        'experience': {
            'keywords': ['work', 'worked', 'job', 'company', 'companies', 'employer', 'position', 'role', 'career'],
            'phrases': ['where did he work', 'places he worked', 'work experience']
        },
        'projects': {
            'keywords': ['project', 'built', 'created', 'developed', 'application', 'model', 'bot'],
            'phrases': ['list projects', 'what projects']
        },
        'skills': {
            'keywords': ['skill', 'technology', 'tech', 'programming', 'language', 'tool'],
            'phrases': ['what skills', 'tech stack']
        },
        'education': {
            'keywords': ['education', 'degree', 'college', 'university', 'studied'],
            'phrases': ['where did he study']
        },
        'certifications': {
            'keywords': ['certification', 'certificate', 'certified', 'course'],
            'phrases': ['certifications']
        },
        'achievements': {
            'keywords': ['achievement', 'accomplishment', 'award'],
            'phrases': ['achievements']
        },
        'contact': {
            'keywords': ['contact', 'email', 'phone', 'reach', 'linkedin'],
            'phrases': ['how to contact']
        },
        'summary': {
            'keywords': ['who', 'about', 'overview', 'summary'],
            'phrases': ['who is', 'tell me about']
        }
    }
    
    scores = {intent: 0 for intent in patterns}
    
    for intent in scores:
        if intent in context_topics:
            scores[intent] += 2
    
    for intent, data in patterns.items():
        for phrase in data['phrases']:
            if phrase in q:
                scores[intent] += len(phrase.split()) * 3
        for keyword in data['keywords']:
            if keyword in q:
                scores[intent] += 1
    
    max_intent = max(scores.items(), key=lambda x: x[1])
    confidence = max_intent[1] / 10.0
    
    return (max_intent[0] if max_intent[1] > 0 else 'general'), confidence

# =========================
# Goal Detection
# =========================
def detect_goal(query: str) -> Optional[str]:
    """Detect if user is starting a new goal/project"""
    goal_patterns = {
        'build': r'(build|create|make|develop) (?:a |an )?(.*?)(?:\.|$)',
        'learn': r'(learn|understand|explain) (?:about )?(.*?)(?:\.|$)',
        'analyze': r'(analyze|review|examine) (.*?)(?:\.|$)',
        'plan': r'(plan|organize|schedule) (.*?)(?:\.|$)'
    }
    
    for goal_type, pattern in goal_patterns.items():
        match = re.search(pattern, query.lower())
        if match:
            return f"{goal_type.title()}: {match.group(2).strip()}"
    
    return None

# =========================
# Formatters
# =========================
def format_experience(exp_list: List[Dict]) -> str:
    if not exp_list:
        return "No experience information found."
    
    result = "**Professional Experience:**\n\n"
    for exp in exp_list:
        result += f"### {exp.get('role', 'N/A')} at {exp.get('company', 'N/A')}\n"
        result += f"📍 {exp.get('location', 'N/A')} | 📅 {exp.get('duration', 'N/A')}\n\n"
        if exp.get('highlights'):
            result += "**Key Achievements:**\n"
            for h in exp['highlights']:
                result += f"- {h}\n"
            result += "\n"
    return result

def format_projects(proj_list: List[Dict]) -> str:
    if not proj_list:
        return "No project information found."
    
    result = "**Projects:**\n\n"
    for proj in proj_list:
        result += f"### {proj.get('name', 'N/A')}\n"
        result += f"{proj.get('description', 'N/A')}\n\n"
        if proj.get('technologies'):
            result += f"**Technologies:** {', '.join(proj['technologies'])}\n\n"
        if proj.get('achievements'):
            result += "**Key Results:**\n"
            for a in proj['achievements']:
                result += f"- {a}\n"
            result += "\n"
    return result

def format_skills(skills_dict: Dict) -> str:
    if not skills_dict:
        return "No skills information found."
    
    result = "**Technical Skills:**\n\n"
    for category, skills in skills_dict.items():
        if skills:
            cat_name = category.replace('_', ' ').title()
            result += f"**{cat_name}:** {', '.join(skills)}\n\n"
    return result

def format_contact(contact_dict: Dict) -> str:
    if not contact_dict:
        return "Contact information not available."
    
    result = "**Contact Information:**\n\n"
    result += f"📧 **Email:** {contact_dict.get('email', 'N/A')}\n"
    result += f"📱 **Phone:** {contact_dict.get('phone', 'N/A')}\n"
    result += f"📍 **Location:** {contact_dict.get('location', 'N/A')}\n"
    result += f"💼 **LinkedIn:** {contact_dict.get('linkedin', 'N/A')}\n"
    result += f"🌐 **Portfolio:** {contact_dict.get('portfolio', 'N/A')}\n"
    return result

def format_certifications(cert_list: List) -> str:
    if not cert_list:
        return "No certifications found."
    
    result = "**Certifications:**\n\n"
    for cert in cert_list:
        if isinstance(cert, dict):
            result += f"- **{cert.get('name', 'N/A')}** — {cert.get('issuer', 'N/A')}"
            if cert.get('year'):
                result += f" ({cert['year']})"
            result += "\n"
        else:
            result += f"- {cert}\n"
    return result

def format_achievements(ach_list: List[str]) -> str:
    if not ach_list:
        return "No achievements information found."
    
    result = "**Notable Achievements:**\n\n"
    for ach in ach_list:
        result += f"- {ach}\n"
    return result

def format_education(edu_list: List[Dict]) -> str:
    if not edu_list:
        return "No education information found."
    
    result = "**Education:**\n\n"
    for edu in edu_list:
        result += f"- **{edu.get('degree', 'N/A')}** in {edu.get('field', 'N/A')}\n"
        result += f"  {edu.get('institution', 'N/A')}, {edu.get('location', 'N/A')}\n\n"
    return result

# =========================
# Context-Aware Search
# =========================
def intelligent_search_with_context(query: str, intent: str, structured: Dict, context_topics: List[str]) -> Tuple[str, str]:
    """Enhanced search with conversation context"""
    
    enriched_query = query
    if context_topics:
        enriched_query = f"{' '.join(context_topics[-2:])} {query}"
    
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
    
    try:
        query_embedding = embedding_model.encode([enriched_query], show_progress_bar=False).tolist()
        
        if intent in available_sections:
            results = collection.query(query_embeddings=query_embedding, n_results=3, where={"section": intent})
        else:
            results = collection.query(query_embeddings=query_embedding, n_results=3)
        
        if results and results['documents'] and results['documents'][0]:
            return '\n\n---\n\n'.join(results['documents'][0]), 'vector'
        
        return "I couldn't find specific information for that query.", 'none'
    except Exception as e:
        return f"Search error: {e}", 'error'

# =========================
# Display Chat History
# =========================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# Chat Handler
# =========================
if user_input := st.chat_input("Your message to Alfred..."):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    detected_goal = detect_goal(user_input)
    if detected_goal and not st.session_state.active_goal:
        st.session_state.active_goal = detected_goal
    
    intent, confidence = classify_intent_with_context(
        user_input, 
        st.session_state.last_intent,
        st.session_state.context_topics
    )
    
    if confidence < 0.3 and not any(word in user_input.lower() for word in ['tell me more', 'what about']):
        clarification = "I want to ensure I understand correctly. Could you please clarify what specific information you need about Mr. Kumar?"
        
        st.session_state.messages.append({"role": "assistant", "content": clarification})
        with st.chat_message("assistant"):
            st.markdown(clarification)
        st.rerun()
    
    st.session_state.last_intent = intent
    if intent not in st.session_state.context_topics:
        st.session_state.context_topics.append(intent)
    st.session_state.context_topics = st.session_state.context_topics[-5:]
    
    context, source = intelligent_search_with_context(
        user_input, 
        intent, 
        structured_data,
        st.session_state.context_topics
    )
    
    mode_instructions = {
        "concise": "Keep responses brief (2-3 sentences max).",
        "balanced": "Provide clear, well-structured responses.",
        "detailed": "Give comprehensive, thorough explanations.",
        "builder": "Focus on technical implementation details."
    }
    
    user_context = f"User: {st.session_state.user_identity['name']}"
    if st.session_state.user_identity['relationship'] == 'owner':
        user_context += " (Mr. Kumar himself)"
    
    goal_context = f"\nActive Goal: {st.session_state.active_goal}" if st.session_state.active_goal else ""
    
    system_prompt = f"""You are Alfred Pennyworth, Mr. Pavan Kumar's distinguished personal AI assistant.

{user_context}{goal_context}

Query Intent: {intent.upper()}
Mode: {mode_instructions[st.session_state.interaction_mode]}

INSTRUCTIONS:
1. Answer using ONLY the context below
2. Maintain refined, professional British butler tone
3. Never invent details
4. If missing info: "That detail is not specified in Mr. Kumar's records"

CONTEXT:
{context}

Respond as Alfred would."""

    if source == 'structured':
        response_text = f"Certainly. {context}"
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)
    else:
        api_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response_text = ""
            
            try:
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=api_messages,
                    temperature=0.5,
                    max_tokens=1500,
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
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    if len(st.session_state.messages) > 30:
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-24:]
    
    cache_data = {
        "messages": st.session_state.messages,
        "conversation_summary": st.session_state.conversation_summary,
        "user_identity": st.session_state.user_identity,
        "last_intent": st.session_state.last_intent,
        "active_goal": st.session_state.active_goal,
        "interaction_mode": st.session_state.interaction_mode,
        "user_preferences": st.session_state.user_preferences,
        "context_topics": st.session_state.context_topics
    }
    SessionCache.save_cache(cache_data)

# =========================
# Footer
# =========================
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("💾 Memory Active" if st.session_state.conversation_summary else "🆕 Fresh Session")

with col2:
    st.caption(f"💬 {st.session_state.interaction_mode.title()}")

with col3:
    st.caption(f"📊 {len(st.session_state.messages)-1} msgs")
