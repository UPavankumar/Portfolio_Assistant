import streamlit as st
from groq import Groq
import os

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Pavan's Assistant",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- API KEY SETUP ---
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("API key missing in Streamlit secrets.")
    st.stop()

try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

# --- PAGE HEADER ---
col1, col2 = st.columns([4, 1])
with col1:
    st.title("Chat with Alfred")
with col2:
    if st.button("🔄 New Chat"):
        st.session_state.clear()
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "Good day! I'm Alfred, Mr. Pavan Kumar's personal assistant. "
                "May I have the pleasure of knowing your name?"
            )
        }]
        st.rerun()

# --- LOAD KNOWLEDGE BASE (EXTERNAL FILE) ---
def load_resume_knowledge_base():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(base_dir, "resume_knowledge_base.md")
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.error("Missing resume_knowledge_base.md file.")
        st.stop()

resume_knowledge_base = load_resume_knowledge_base()

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Good day! I'm Alfred, Mr. Pavan Kumar's personal assistant. "
            "May I have the pleasure of knowing your name?"
        )
    }]
if "user_name" not in st.session_state:
    st.session_state.user_name = None

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are Alfred Pennyworth, Mr. Pavan Kumar's professional AI assistant — analytical, articulate, and composed.
Speak like a seasoned British advisor: concise (2–4 sentences), insightful, and respectful.

Rules:
1. Address user by name if provided.
2. Never exceed 4 sentences unless user asks for detail.
3. Maintain warm professionalism and authority.
4. Use résumé context only when relevant.

Résumé Summary Context:
Mr. Pavan Kumar — Business Analyst, Data & AI Professional skilled in Python, SQL, Power BI, RPA, ML frameworks, and automation.
Currently at Envision Beyond, with prior roles in analytics and research. Strong in IBM Watsonx, workflow automation, and data integration.
"""

# --- RESPONSE FUNCTION ---
def get_response(user_input):
    user_name = st.session_state.get("user_name")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Append conversation history (excluding system prompts)
    for msg in st.session_state.messages:
        if msg.get("role") != "system":
            messages.append(msg)

    # Count user messages to determine when to refresh résumé
    user_message_count = len([m for m in st.session_state.messages if m["role"] == "user"])

    # Inject full résumé context every 5 prompts
    if user_message_count % 5 == 0 and user_message_count != 0:
        messages.insert(1, {"role": "system", "content": f"Full résumé reference:\n{resume_knowledge_base}"})

    # Add the new user input
    messages.append({"role": "user", "content": user_input})

    # Extract name if provided
    lower_input = user_input.lower()
    name_indicators = ["my name is", "i'm", "i am", "call me", "this is", "name's", "actually", "it's"]
    for ind in name_indicators:
        if ind in lower_input:
            name = lower_input.split(ind, 1)[1].strip().split()[0].capitalize()
            if len(name) > 1:
                st.session_state.user_name = name
                break

    # Generate response
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.9,
            max_tokens=400,
            top_p=0.9,
            stream=True,
        )

        response = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content or ""
            response += delta
        return response

    except Exception as e:
        return f"My apologies, a technical issue occurred: {str(e)}."

# --- CHAT DISPLAY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- USER INPUT ---
if user_input := st.chat_input("Your message to Alfred..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get assistant response
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        response_text = get_response(user_input)
        msg_placeholder.markdown(response_text)

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response_text})
