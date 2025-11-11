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

# Debug mode toggle (set to False in production)
DEBUG_MODE = st.sidebar.checkbox("🔧 Debug Mode", value=False)

# =========================
# Load API Key
# =========================
api_key = st.secrets.get("GROQ_API_KEY", None)
if not api_key:
    st.error("GROQ_API_KEY not found in Streamlit secrets.")
    st.stop()

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
    try:
        return chromadb.PersistentClient(path="./chroma_db")
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {e}")
        st.stop()

chroma_client = get_chroma_client()

@st.cache_resource
def get_embeddings_model():
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
        "experience": ["work", "worked", "job", "company", "companies", "employer", "position", "role", "career", "employment", "places he worked", "where did he work"],
        "projects": ["project", "projects", "built", "created", "developed", "model", "application", "churn", "discord", "bot", "prediction", "portfolio work"],
        "skills": ["skill", "technology", "tech stack", "programming", "language", "tool", "knows", "proficient", "expertise", "can he", "technologies"],
        "education": ["study", "studied", "degree", "college", "university", "education", "graduated", "bachelor", "school"],
        "contact": ["contact", "email", "phone", "reach", "linkedin", "portfolio", "connect", "availability", "location"],
        "summary": ["who is", "about", "overview", "summary", "tell me about", "introduction", "background", "all info", "everything"],
        "certifications": ["certification", "certified", "certificate", "course", "training"],
        "achievements": ["achievement", "promotion", "award", "recognition", "accomplishment", "notable"],
    }
    
    # Score each intent with weighted matching
    scores = {}
    for intent, keywords in intent_patterns.items():
        score = 0
        for kw in keywords:
            if kw in query_lower:
                # Give higher weight to longer, more specific matches
                score += len(kw.split()) * 2
        scores[intent] = score
    
    max_intent = max(scores, key=scores.get)
    return max_intent if scores[max_intent] > 0 else "general"

# =========================
# Load Resume File
# =========================
resume_path = Path("resume_knowledge_base.md")
if not resume_path.exists():
    st.error("Resume file not found.")
    st.stop()

try:
    resume_text = resume_path.read_text(encoding="utf-8")
except Exception as e:
    st.error(f"Failed to read resume file: {e}")
    st.stop()

# =========================
# Enhanced Section Parsing
# =========================
def parse_resume_sections(content: str) -> dict:
    """Parse resume into sections with intelligent mapping"""
    sections = {}
    current_section = "header"
    current_content = []
    
    lines = content.split('\n')
    
    # Section mapping rules
    section_mapping = {
        'work experience': 'experience',
        'professional experience': 'experience',
        'experience': 'experience',
        'employment': 'experience',
        'career': 'experience',
        
        'projects': 'projects',
        'portfolio': 'projects',
        
        'skills': 'skills',
        'technical skills': 'skills',
        'expertise': 'skills',
        'tools': 'skills',
        
        'education': 'education',
        'academic': 'education',
        
        'certifications': 'certifications',
        'certificates': 'certifications',
        'training': 'certifications',
        
        'achievements': 'achievements',
        'accomplishments': 'achievements',
        
        'contact': 'contact',
        'personal information': 'contact',
        
        'summary': 'summary',
        'professional summary': 'summary',
        'about': 'summary',
    }
    
    for line in lines:
        # Detect section headers (## or ###)
        if line.strip().startswith('##') and not line.strip().startswith('####'):
            # Save previous section
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            
            # Extract and normalize section name
            section_name = line.replace('#', '').strip().lower()
            
            # Map to standard category
            current_section = section_mapping.get(section_name, section_name)
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
    """Initialize knowledge base with properly sectioned chunks"""
    try:
        collection = _chroma_client.get_or_create_collection(
            name="resume_knowledge_base_v2",  # New version to force rebuild
            metadata={"hnsw:space": "cosine"}
        )
        
        # Always rebuild for consistency
        if collection.count() > 0:
            _chroma_client.delete_collection("resume_knowledge_base_v2")
            collection = _chroma_client.create_collection(
                name="resume_knowledge_base_v2",
                metadata={"hnsw:space": "cosine"}
            )
        
        sections = parse_resume_sections(resume_content)
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        chunk_counter = 0
        
        for section_name, section_content in sections.items():
            # Don't split - keep sections whole for better context
            if section_content.strip():
                all_chunks.append(section_content)
                all_metadatas.append({
                    "section": section_name,
                    "char_count": len(section_content)
                })
                all_ids.append(f"{section_name}_{chunk_counter}")
                chunk_counter += 1
        
        # Generate embeddings
        embeddings = _embedding_model.encode(all_chunks, show_progress_bar=False).tolist()
        
        # Add to collection
        collection.add(
            ids=all_ids,
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=all_metadatas
        )
        
        # Show sections found
        section_list = list(sections.keys())
        st.success(f"✅ Initialized with {len(all_chunks)} sections: {', '.join(section_list)}")
        
        return collection, sections
    
    except Exception as e:
        st.error(f"Failed to initialize knowledge base: {e}")
        st.stop()

collection, parsed_sections = initialize_knowledge_base(chroma_client, embedding_model, resume_text)

# =========================
# Smart Retrieval Function
# =========================
def smart_search(query: str, intent: str, top_k: int = 3) -> tuple:
    """Search with intent-based filtering and fallback"""
    try:
        query_embedding = embedding_model.encode([query], show_progress_bar=False).tolist()
        
        retrieved_chunks = []
        search_metadata = {"intent": intent, "filtered": False}
        
        # Try intent-filtered search first
        if intent != "general":
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where={"section": intent}
            )
            
            if results and results["documents"] and results["documents"][0]:
                retrieved_chunks = results["documents"][0]
                search_metadata["filtered"] = True
        
        # Fallback: search all if no results or general intent
        if not retrieved_chunks:
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )
            
            if results and results["documents"] and results["documents"][0]:
                retrieved_chunks = results["documents"][0]
                search_metadata["filtered"] = False
        
        return retrieved_chunks, search_metadata
    
    except Exception as e:
        st.warning(f"Search error: {e}")
        return [], {"error": str(e)}

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
    
    if DEBUG_MODE:
        st.sidebar.write(f"**Detected Intent:** `{intent}`")
    
    # Step 2: Smart Retrieval
    relevant_chunks, search_meta = smart_search(user_input, intent, top_k=3)
    
    if DEBUG_MODE:
        st.sidebar.write(f"**Search Type:** {'Filtered' if search_meta.get('filtered') else 'Unfiltered'}")
        st.sidebar.write(f"**Chunks Retrieved:** {len(relevant_chunks)}")
        if relevant_chunks:
            st.sidebar.write("**Preview:**")
            st.sidebar.code(relevant_chunks[0][:200] + "...")
    
    if relevant_chunks:
        resume_context = "\n\n---SECTION BREAK---\n\n".join(relevant_chunks)
    else:
        resume_context = "I apologize, but I couldn't locate relevant information in Mr. Kumar's records."
    
    # Step 3: Build Enhanced System Prompt
    system_prompt = f"""You are Alfred Pennyworth, Mr. Pavan Kumar's distinguished personal assistant.

USER QUERY INTENT: {intent.upper()}

CRITICAL INSTRUCTIONS:
1. Answer ONLY using the Résumé Context provided below
2. If the query asks for a list (e.g., "list all projects"), enumerate ALL items clearly with bullet points
3. For "experience" questions, list ALL companies, roles, and durations
4. For "projects" questions, describe ALL projects with their key details
5. If specific information is missing, state: "That detail is not specified in Mr. Kumar's records"
6. Be comprehensive - don't summarize if the user asks for "all" or "list"
7. Maintain Alfred's refined, professional tone

RÉSUMÉ CONTEXT:
{resume_context}

Respond now as Alfred would."""

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
                temperature=0.5,
                max_tokens=1000,
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
    
    # Trim history
    if len(st.session_state.messages) > 20:
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-19:]
