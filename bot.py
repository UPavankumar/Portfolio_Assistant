import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional

# =========================
# Streamlit Page Setup
# =========================
st.set_page_config(
    page_title="💼 Alfred — Pavan Kumar's Personal AI Assistant",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for better UX
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .structured-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("💼 Alfred — Pavan Kumar's Personal AI Assistant")

# Debug mode
DEBUG_MODE = st.sidebar.checkbox("🔧 Debug Mode", value=False)

# =========================
# Initialize Groq Client
# =========================
api_key = st.secrets.get("GROQ_API_KEY", None)
if not api_key:
    st.error("⚠️ GROQ_API_KEY not found in Streamlit secrets.")
    st.stop()

try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"❌ Failed to initialize Groq client: {e}")
    st.stop()

# =========================
# Load Models
# =========================
@st.cache_resource
def get_chroma_client():
    """Initialize ChromaDB"""
    try:
        return chromadb.PersistentClient(path="./chroma_db")
    except Exception as e:
        st.error(f"❌ ChromaDB initialization failed: {e}")
        st.stop()

@st.cache_resource
def get_embeddings_model():
    """Load sentence transformer"""
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
    st.error("❌ Resume file (resume_knowledge_base.md) not found.")
    st.stop()

resume_text = resume_path.read_text(encoding="utf-8")

# =========================
# Structured Data Extraction
# =========================
@st.cache_data
def extract_structured_data(_client, resume_content: str) -> Dict:
    """Extract structured information from resume using LLM"""
    
    extraction_prompt = f"""You are a resume parser. Extract structured information from this résumé and return ONLY valid JSON with this exact structure:

{{
  "experience": [
    {{
      "company": "Company Name",
      "role": "Job Title",
      "duration": "Start - End",
      "location": "City, Country",
      "highlights": ["achievement 1", "achievement 2"]
    }}
  ],
  "projects": [
    {{
      "name": "Project Name",
      "description": "Brief description",
      "technologies": ["tech1", "tech2"],
      "achievements": ["achievement 1"]
    }}
  ],
  "skills": {{
    "programming": ["Python", "SQL"],
    "ai_ml": ["TensorFlow", "PyTorch"],
    "tools": ["Power BI", "Tableau"],
    "databases": ["PostgreSQL", "MongoDB"],
    "automation": ["RPA", "N8n"]
  }},
  "education": [
    {{
      "degree": "Degree Name",
      "field": "Field of Study",
      "institution": "Institution Name",
      "location": "City, Country"
    }}
  ],
  "certifications": [
    {{
      "name": "Certification Name",
      "issuer": "Issuing Organization",
      "year": "Year or Date"
    }}
  ],
  "achievements": [
    "Achievement 1",
    "Achievement 2"
  ],
  "contact": {{
    "name": "Full Name",
    "email": "email@example.com",
    "phone": "phone number",
    "location": "City, Country",
    "linkedin": "linkedin url",
    "portfolio": "portfolio url"
  }},
  "summary": "2-3 sentence professional summary"
}}

Extract ALL information accurately. Do not invent information.

RÉSUMÉ TEXT:
{resume_content}

Return ONLY the JSON, no markdown formatting, no additional text."""

    try:
        with st.spinner("🔄 Analyzing résumé structure..."):
            response = _client.chat.completions.create(
                model="llama-3.1-70b-versatile",  # Use larger model for extraction
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0,
                max_tokens=3000
            )
        
        json_text = response.choices[0].message.content.strip()
        
        # Clean markdown formatting
        json_text = re.sub(r'^```json\s*', '', json_text)
        json_text = re.sub(r'\s*```$', '', json_text)
        json_text = json_text.strip()
        
        structured_data = json.loads(json_text)
        
        # Save to cache file
        cache_file = Path("structured_resume_cache.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2)
        
        return structured_data
    
    except json.JSONDecodeError as e:
        st.error(f"⚠️ Failed to parse structured data: {e}")
        st.code(json_text)
        return {}
    except Exception as e:
        st.error(f"⚠️ Extraction error: {e}")
        return {}

# Try to load from cache first
cache_file = Path("structured_resume_cache.json")
if cache_file.exists():
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
        st.sidebar.success("✅ Loaded structured data from cache")
    except:
        structured_data = extract_structured_data(client, resume_text)
else:
    structured_data = extract_structured_data(client, resume_text)

if DEBUG_MODE:
    st.sidebar.json(structured_data)

# =========================
# Vector Database Setup
# =========================
@st.cache_resource
def initialize_vector_db(_chroma_client, _embedding_model, resume_content: str):
    """Initialize vector database with resume chunks"""
    
    collection_name = "alfred_resume_kb_v3"
    
    try:
        # Delete old collection if exists
        try:
            _chroma_client.delete_collection(collection_name)
        except:
            pass
        
        collection = _chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Smart chunking by sections
        sections = {}
        current_section = "header"
        current_content = []
        
        for line in resume_content.split('\n'):
            if line.startswith('##') and not line.startswith('###'):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                section_name = line.replace('#', '').strip().lower()
                # Normalize section names
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
        
        # Add to vector DB
        all_docs = []
        all_metas = []
        all_ids = []
        
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
        st.error(f"❌ Vector DB initialization failed: {e}")
        st.stop()

collection, available_sections = initialize_vector_db(chroma_client, embedding_model, resume_text)

# =========================
# Intent Classification
# =========================
def classify_intent(query: str) -> str:
    """Advanced intent classification"""
    q = query.lower()
    
    # Weighted keyword patterns
    patterns = {
        'experience': {
            'keywords': ['work', 'worked', 'job', 'company', 'companies', 'employer', 'position', 'role', 'career', 'employment'],
            'phrases': ['where did he work', 'places he worked', 'work experience', 'work history', 'previous jobs']
        },
        'projects': {
            'keywords': ['project', 'built', 'created', 'developed', 'application', 'model', 'bot'],
            'phrases': ['list projects', 'what projects', 'projects he built', 'portfolio work']
        },
        'skills': {
            'keywords': ['skill', 'technology', 'tech', 'programming', 'language', 'tool', 'expertise', 'proficient'],
            'phrases': ['what skills', 'tech stack', 'technologies', 'programming languages', 'tools he knows']
        },
        'education': {
            'keywords': ['education', 'degree', 'college', 'university', 'studied', 'graduated', 'bachelor', 'school'],
            'phrases': ['where did he study', 'educational background', 'what degree']
        },
        'certifications': {
            'keywords': ['certification', 'certificate', 'certified', 'course', 'training'],
            'phrases': ['certifications', 'certificates', 'certified in']
        },
        'achievements': {
            'keywords': ['achievement', 'accomplishment', 'award', 'recognition', 'promotion', 'notable'],
            'phrases': ['achievements', 'accomplishments', 'awards']
        },
        'contact': {
            'keywords': ['contact', 'email', 'phone', 'reach', 'linkedin', 'portfolio', 'location', 'available'],
            'phrases': ['how to contact', 'contact information', 'reach him', 'email address']
        },
        'summary': {
            'keywords': ['who', 'about', 'overview', 'summary', 'introduction', 'background', 'tell me'],
            'phrases': ['who is', 'tell me about', 'give overview', 'background', 'all info', 'everything']
        }
    }
    
    scores = {intent: 0 for intent in patterns}
    
    for intent, data in patterns.items():
        # Phrase matching (higher weight)
        for phrase in data['phrases']:
            if phrase in q:
                scores[intent] += len(phrase.split()) * 3
        
        # Keyword matching
        for keyword in data['keywords']:
            if keyword in q:
                scores[intent] += 1
    
    max_intent = max(scores.items(), key=lambda x: x[1])
    return max_intent[0] if max_intent[1] > 0 else 'general'

# =========================
# Formatting Helpers
# =========================
def format_experience(exp_list: List[Dict]) -> str:
    """Format experience data beautifully"""
    if not exp_list:
        return "No experience information found."
    
    result = "**Professional Experience:**\n\n"
    for exp in exp_list:
        result += f"### {exp.get('role', 'N/A')} at {exp.get('company', 'N/A')}\n"
        result += f"📍 {exp.get('location', 'N/A')} | 📅 {exp.get('duration', 'N/A')}\n\n"
        
        if exp.get('highlights'):
            result += "**Key Achievements:**\n"
            for highlight in exp['highlights']:
                result += f"- {highlight}\n"
            result += "\n"
    
    return result

def format_projects(proj_list: List[Dict]) -> str:
    """Format projects data"""
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
            for ach in proj['achievements']:
                result += f"- {ach}\n"
            result += "\n"
    
    return result

def format_skills(skills_dict: Dict) -> str:
    """Format skills data"""
    if not skills_dict:
        return "No skills information found."
    
    result = "**Technical Skills:**\n\n"
    
    for category, skills in skills_dict.items():
        if skills:
            category_name = category.replace('_', ' ').title()
            result += f"**{category_name}:**\n"
            result += f"{', '.join(skills)}\n\n"
    
    return result

def format_contact(contact_dict: Dict) -> str:
    """Format contact information"""
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
    """Format certifications"""
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
    """Format achievements"""
    if not ach_list:
        return "No achievements information found."
    
    result = "**Notable Achievements:**\n\n"
    for ach in ach_list:
        result += f"- {ach}\n"
    
    return result

# =========================
# Intelligent Search
# =========================
def intelligent_search(query: str, intent: str, structured: Dict) -> Tuple[str, str]:
    """Hybrid search: structured first, then vector fallback"""
    
    # Try structured data first for direct queries
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
        edu_list = structured['education']
        result = "**Education:**\n\n"
        for edu in edu_list:
            result += f"- **{edu.get('degree', 'N/A')}** in {edu.get('field', 'N/A')}\n"
            result += f"  {edu.get('institution', 'N/A')}, {edu.get('location', 'N/A')}\n\n"
        return result, 'structured'
    
    elif intent == 'summary' and structured.get('summary'):
        return structured['summary'], 'structured'
    
    # Fallback to vector search
    try:
        query_embedding = embedding_model.encode([query], show_progress_bar=False).tolist()
        
        # Try filtered search first
        if intent in available_sections:
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=3,
                where={"section": intent}
            )
        else:
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=3
            )
        
        if results and results['documents'] and results['documents'][0]:
            return '\n\n---\n\n'.join(results['documents'][0]), 'vector'
        
        return "I couldn't find specific information for that query.", 'none'
    
    except Exception as e:
        return f"Search error: {e}", 'error'

# =========================
# Answer Validation
# =========================
def validate_answer(answer: str, structured: Dict) -> str:
    """Validate that answer doesn't contain invented information"""
    
    # Extract known entities
    known_companies = set()
    if structured.get('experience'):
        known_companies = {exp.get('company', '').lower() for exp in structured['experience']}
    
    known_projects = set()
    if structured.get('projects'):
        known_projects = {proj.get('name', '').lower() for proj in structured['projects']}
    
    # Check for suspicious corporate names not in our data
    suspicious = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Technologies|Inc|Corp|Ltd|Solutions|Systems|Group)\b', answer)
    
    for company in suspicious:
        if company.lower() not in known_companies and company.lower() not in known_projects:
            return "I apologize, but I may have included unverified information. Let me provide only confirmed details from Mr. Kumar's records."
    
    return answer

# =========================
# Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Good day. I am Alfred, Mr. Pavan Kumar's personal AI assistant. I have his complete professional dossier at hand. How may I assist you today?"
    }]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# Chat Handler
# =========================
if user_input := st.chat_input("Your message to Alfred..."):
    
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Classify intent
    intent = classify_intent(user_input)
    
    if DEBUG_MODE:
        st.sidebar.write(f"**Intent:** {intent}")
    
    # Intelligent search
    context, source = intelligent_search(user_input, intent, structured_data)
    
    if DEBUG_MODE:
        st.sidebar.write(f"**Source:** {source}")
        st.sidebar.text_area("Context", context[:500], height=200)
    
    # If structured data gave us a direct answer, use it
    if source == 'structured':
        response_text = f"Certainly. {context}"
        
        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        with st.chat_message("assistant"):
            st.markdown(response_text)
    
    else:
        # Use LLM for synthesis
        system_prompt = f"""You are Alfred Pennyworth, Mr. Pavan Kumar's distinguished personal AI assistant.

You are responding to a query about: **{intent.upper()}**

CRITICAL INSTRUCTIONS:
1. Answer using ONLY the information in the context below
2. If asked for a list, provide ALL items clearly with proper formatting
3. Maintain Alfred's refined, professional British butler tone
4. Never invent companies, projects, or details not in the context
5. If information is missing, state: "That detail is not specified in Mr. Kumar's records"
6. Be comprehensive but concise
7. Use markdown formatting for better readability

CONTEXT:
{context}

STRUCTURED DATA SUMMARY:
- Experience: {len(structured_data.get('experience', []))} positions
- Projects: {len(structured_data.get('projects', []))} projects
- Skills: {sum(len(v) for v in structured_data.get('skills', {}).values())} total skills
- Certifications: {len(structured_data.get('certifications', []))} certifications

Respond as Alfred would, professionally and accurately."""

        api_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response_text = ""
            
            try:
                completion = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=api_messages,
                    temperature=0.5,
                    max_tokens=1500,
                    top_p=0.9,
                    stream=True,
                )
                
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        response_text += delta
                        message_placeholder.markdown(response_text + "▌")
                
                # Validate answer
                response_text = validate_answer(response_text, structured_data)
                message_placeholder.markdown(response_text)
                
            except Exception as e:
                response_text = f"My apologies, I encountered a difficulty: {str(e)}"
                message_placeholder.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Manage conversation history
    if len(st.session_state.messages) > 30:
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-29:]
