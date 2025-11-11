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
        try:
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except:
            pass
    
    @staticmethod
    def load_cache() -> Optional[Dict]:
        try:
            if not CACHE_FILE.exists():
                return None
            
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            cache_time = datetime.fromisoformat(cache_data["timestamp"])
            if datetime.now() - cache_time > timedelta(hours=CACHE_EXPIRY_HOURS):
                return None
            
            return cache_data["data"]
        except:
            return None
    
    @staticmethod
    def clear_cache():
        try:
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
        except:
            pass

# =========================
# Initialize Session State
# =========================
def init_session_state():
    if "initialized" not in st.session_state:
        cached_data = SessionCache.load_cache()
        
        if cached_data:
            st.session_state.update(cached_data)
            st.session_state.cache_restored = True
        else:
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Good day. I am Alfred, Mr. Pavan Kumar's personal AI assistant. How may I assist you today?"
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
            st.session_state.context_topics = []
            st.session_state.cache_restored = False
        
        st.session_state.initialized = True

init_session_state()

# Sidebar Controls
with st.sidebar:
    st.header("🎛️ Alfred Controls")
    
    st.subheader("👤 User Identity")
    user_name = st.text_input("Your Name", value=st.session_state.user_identity.get("name", "Guest"))
    if user_name != st.session_state.user_identity["name"]:
        st.session_state.user_identity["name"] = user_name
        if user_name.lower() in ["pavan", "pavan kumar"]:
            st.session_state.user_identity["relationship"] = "owner"
            st.session_state.user_identity["verified"] = True
            st.success("✅ Owner verified")
    
    st.subheader("💬 Interaction Mode")
    mode = st.selectbox(
        "Response Style",
        ["concise", "balanced", "detailed", "builder"],
        index=["concise", "balanced", "detailed", "builder"].index(st.session_state.interaction_mode)
    )
    if mode != st.session_state.interaction_mode:
        st.session_state.interaction_mode = mode
    
    st.subheader("🎯 Active Goal")
    if st.session_state.active_goal:
        st.markdown(f'<div class="goal-badge">🎯 {st.session_state.active_goal}</div>', unsafe_allow_html=True)
        if st.button("Clear Goal"):
            st.session_state.active_goal = None
            st.rerun()
    else:
        st.info("No active goal")
    
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
    
    DEBUG_MODE = st.checkbox("🔧 Debug", value=False)
    
    if DEBUG_MODE:
        st.json({
            "last_intent": st.session_state.last_intent,
            "active_goal": st.session_state.active_goal,
            "mode": st.session_state.interaction_mode,
            "messages": len(st.session_state.messages),
            "topics": st.session_state.context_topics[-3:] if st.session_state.context_topics else []
        })

# =========================
# Initialize Clients
# =========================
try:
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found. Add it to .streamlit/secrets.toml")
        st.stop()
    
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"❌ Groq initialization failed: {e}")
    st.stop()

@st.cache_resource
def get_chroma_client():
    try:
        return chromadb.PersistentClient(path="./chroma_db")
    except Exception as e:
        st.error(f"❌ ChromaDB failed: {e}")
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
    st.error("❌ resume_knowledge_base.md not found")
    st.stop()

resume_text = resume_path.read_text(encoding="utf-8")

# =========================
# Structured Data Extraction
# =========================
@st.cache_data
def extract_structured_data(_client, resume_content: str) -> Dict:
    extraction_prompt = f"""Extract structured information from this résumé and return ONLY valid JSON with NO markdown:

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

Return ONLY the JSON object, nothing else."""

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
        st.warning(f"⚠️ Using fallback data extraction")
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
    collection_name = "alfred_kb_v6"
    
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
# Smart Intent Classification
# =========================
def classify_intent(query: str, last_intent: Optional[str], context_topics: List[str]) -> str:
    """Smart intent classification - NO clarification questions"""
    q = query.lower()
    
    # Detect follow-ups - use last intent immediately
    follow_up_patterns = ['tell me more', 'what about', 'and', 'also', 'what else', 'more about', 'continue', 'go on']
    if any(pattern in q for pattern in follow_up_patterns) and last_intent:
        return last_intent
    
    # Strong keyword matching
    if any(word in q for word in ['work', 'job', 'company', 'employer', 'role', 'position', 'career', 'worked']):
        return 'experience'
    
    if any(word in q for word in ['project', 'built', 'created', 'developed', 'application', 'model', 'bot']):
        return 'projects'
    
    if any(word in q for word in ['skill', 'technology', 'tech', 'programming', 'language', 'tool', 'expertise']):
        return 'skills'
    
    if any(word in q for word in ['education', 'degree', 'college', 'university', 'studied', 'graduated']):
        return 'education'
    
    if any(word in q for word in ['certification', 'certificate', 'certified', 'course']):
        return 'certifications'
    
    if any(word in q for word in ['achievement', 'accomplishment', 'award', 'promoted']):
        return 'achievements'
    
    if any(word in q for word in ['contact', 'email', 'phone', 'reach', 'linkedin', 'portfolio', 'website']):
        return 'contact'
    
    if any(word in q for word in ['who', 'about', 'overview', 'summary', 'background', 'tell me about', 'introduce']):
        return 'summary'
    
    # Use context if available
    if context_topics:
        return context_topics[-1]
    
    # Default to summary for general queries
    return 'summary'

# =========================
# Goal Detection
# =========================
def detect_goal(query: str) -> Optional[str]:
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
        return "No project information available."
    
    result = "**Projects:**\n\n"
    for proj in proj_list:
        result += f"### {proj.get('name', 'N/A')}\n"
        result += f"{proj.get('description', 'N/A')}\n\n"
        if proj.get('technologies'):
            result += f"**Technologies:** {', '.join(proj['technologies'])}\n\n"
        if proj.get('achievements'):
            for a in proj['achievements']:
                result += f"• {a}\n"
            result += "\n"
    return result

def format_skills(skills_dict: Dict) -> str:
    if not skills_dict:
        return "No skills information available."
    
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
        return "No certifications available."
    
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
        return "No achievements information available."
    
    result = "**Notable Achievements:**\n\n"
    for ach in ach_list:
        result += f"• {ach}\n"
    return result

def format_education(edu_list: List[Dict]) -> str:
    if not edu_list:
        return "No education information available."
    
    result = "**Education:**\n\n"
    for edu in edu_list:
        result += f"• **{edu.get('degree', 'N/A')}** in {edu.get('field', 'N/A')}\n"
        result += f"  {edu.get('institution', 'N/A')}, {edu.get('location', 'N/A')}\n\n"
    return result

# =========================
# Intelligent Search
# =========================
def intelligent_search(query: str, intent: str, structured: Dict, context_topics: List[str]) -> Tuple[str, str]:
    # Enrich query with context
    enriched_query = query
    if context_topics:
        enriched_query = f"{' '.join(context_topics[-2:])} {query}"
    
    # Try structured data first
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
    
    # Fallback to vector search
    try:
        query_embedding = embedding_model.encode([enriched_query], show_progress_bar=False).tolist()
        
        if intent in available_sections:
            results = collection.query(query_embeddings=query_embedding, n_results=3, where={"section": intent})
        else:
            results = collection.query(query_embeddings=query_embedding, n_results=3)
        
        if results and results['documents'] and results['documents'][0]:
            return '\n\n'.join(results['documents'][0]), 'vector'
        
        return "I don't have specific information on that topic in Mr. Kumar's records.", 'none'
    except Exception as e:
        return f"I encountered a technical issue: {str(e)}", 'error'

# =========================
# Build Conversation Context
# =========================
def build_conversation_context(messages: List[Dict], max_messages: int = 4) -> str:
    """Build conversation history for context"""
    if len(messages) <= 1:
        return ""
    
    recent = messages[-(max_messages*2):-1]  # Skip current message
    context = []
    
    for msg in recent:
        role = msg['role'].upper()
        content = msg['content'][:150]  # Truncate long messages
        context.append(f"{role}: {content}")
    
    return "\n".join(context)

# =========================
# Display Chat History
# =========================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# Main Chat Handler
# =========================
if user_input := st.chat_input("Your message to Alfred..."):
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Detect goal
    detected_goal = detect_goal(user_input)
    if detected_goal and not st.session_state.active_goal:
        st.session_state.active_goal = detected_goal
    
    # Classify intent - NO CLARIFICATION, ALWAYS PROCEED
    intent = classify_intent(
        user_input, 
        st.session_state.last_intent,
        st.session_state.context_topics
    )
    
    # Update context
    st.session_state.last_intent = intent
    if intent not in st.session_state.context_topics:
        st.session_state.context_topics.append(intent)
    st.session_state.context_topics = st.session_state.context_topics[-5:]
    
    # Get context
    context, source = intelligent_search(
        user_input, 
        intent, 
        structured_data,
        st.session_state.context_topics
    )
    
    # Build conversation history
    conversation_history = build_conversation_context(st.session_state.messages)
    
    # Mode instructions
    mode_instructions = {
        "concise": "Be brief and direct. 2-3 sentences maximum.",
        "balanced": "Provide clear, well-structured responses with key details.",
        "detailed": "Give comprehensive explanations with context and examples.",
        "builder": "Focus on technical details, implementation steps, and actionable insights."
    }
    
    # User context
    user_context = f"Speaking with: {st.session_state.user_identity['name']}"
    if st.session_state.user_identity['relationship'] == 'owner':
        user_context += " (Mr. Kumar himself)"
    
    goal_info = f"\n\nActive Goal: {st.session_state.active_goal}" if st.session_state.active_goal else ""
    
    # System prompt
    system_prompt = f"""You are Alfred Pennyworth, Mr. Pavan Kumar's distinguished British personal AI assistant.

{user_context}{goal_info}

CONVERSATION CONTEXT:
{conversation_history}

CURRENT QUERY INTENT: {intent.upper()}
RESPONSE MODE: {mode_instructions[st.session_state.interaction_mode]}

CORE RULES:
1. Answer ONLY using the CONTEXT below - never invent information
2. Be professional, refined, and confident like Alfred Pennyworth
3. If information is missing, say "That detail is not in Mr. Kumar's records" - DON'T ask clarifying questions
4. Use markdown formatting for better readability
5. Maintain conversational flow based on conversation history
6. NEVER ask "what would you like to know" - just answer the query directly

CONTEXT FOR THIS QUERY:
{context}

Now respond to the user's query as Alfred would - direct, professional, and helpful."""

    # Generate response
    if source == 'structured' and intent in ['experience', 'projects', 'skills', 'contact', 'certifications', 'achievements', 'education']:
        # Direct structured response
        response_text = context
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)
    else:
        # LLM synthesis
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
                    temperature=0.6,
                    max_tokens=1200,
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
                error_msg = str(e)
                if "rate" in error_msg.lower():
                    response_text = "My apologies, I'm experiencing high demand. Please try again in a moment."
                else:
                    response_text = f"I encountered a technical issue. Please try rephrasing your query."
                message_placeholder.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Manage message history
    if len(st.session_state.messages) > 30:
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-24:]
    
    # Auto-save session
    cache_data = {
        "messages": st.session_state.messages,
        "conversation_summary": st.session_state.conversation_summary,
        "user_identity": st.session_state.user_identity,
        "last_intent": st.session_state.last_intent,
        "active_goal": st.session_state.active_goal,
        "interaction_mode": st.session_state.interaction_mode,
        "context_topics": st.session_state.context_topics
    }
    SessionCache.save_cache(cache_data)

# =========================
# Footer
# =========================
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.conversation_summary or len(st.session_state.messages) > 5:
        st.caption("💾 Memory Active")
    else:
        st.caption("🆕 Fresh Session")

with col2:
    st.caption(f"💬 {st.session_state.interaction_mode.title()}")

with col3:
    msg_count = len(st.session_state.messages) - 1
    st.caption(f"📊 {msg_count} messages")
