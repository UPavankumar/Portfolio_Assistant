import streamlit as st
from groq import Groq
from datetime import datetime

# Load API key from secrets
api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

st.set_page_config(page_title="Chat with Alfred", layout="centered", initial_sidebar_state="collapsed")
st.title("Chat with Alfred")

# Initialize chat history
if 'conversation' not in st.session_state:
    st.session_state.conversation = [{
        "role": "assistant",
        "content": "Good day! I’m Alfred, Mr. Pavan Kumar’s personal assistant. May I have the pleasure of knowing your name?",
        "timestamp": datetime.now().strftime("%I:%M %p")
    }]

# Resume knowledge base (replace with actual content)
resume_knowledge_base = """Pavan Kumar is an aspiring Data Analyst with expertise in Python, SQL, Power BI..."""

# Function to get Groq response
def get_response(user_input):
    try:
        system_prompt = f"""
You are Alfred Pennyworth, Pavan Kumar's refined and witty personal assistant. Respond with a touch of British charm and professionalism.
Knowledge Base: {resume_knowledge_base}
"""
        completion = client.chat.completions.create(
            model="llama-3-1-8b",
            messages=[
                {"role": "system", "content": system_prompt},
                *[{ "role": msg["role"], "content": msg["content"] } for msg in st.session_state.conversation],
                {"role": "user", "content": user_input}
            ],
            temperature=0.85,
            max_tokens=512,
            top_p=0.9,
            stream=True,
        )

        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
        return response
    except Exception as e:
        return f"Apologies, Sir/Madam. We encountered an issue: {str(e)}"

# Custom CSS for layout
st.markdown("""
<style>
.chat-container {
    max-width: 700px;
    margin: auto;
}
.chat-history {
    height: 60vh;
    overflow-y: auto;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 10px;
    background: #1e1e1e;
    color: white;
}
.chat-message {
    margin: 10px 0;
    padding: 10px;
    border-radius: 10px;
}
.chat-message.user {
    background-color: #2a9d8f;
    text-align: right;
    margin-left: auto;
    width: fit-content;
}
.chat-message.assistant {
    background-color: #264653;
    text-align: left;
    margin-right: auto;
    width: fit-content;
}
.timestamp {
    font-size: 0.75em;
    color: #cccccc;
    display: block;
    margin-top: 5px;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

# Layout container
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Chat History
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    for msg in st.session_state.conversation:
        role = msg["role"]
        content = msg["content"]
        timestamp = msg["timestamp"]
        css_class = "user" if role == "user" else "assistant"
        sender = "You" if role == "user" else "Alfred"
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <b>{sender}:</b> {content}
            <span class="timestamp">{timestamp}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # end chat-history

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Message", placeholder="Type your message here", label_visibility="collapsed")
        submitted = st.form_submit_button("Send")

    st.markdown('</div>', unsafe_allow_html=True)  # end chat-container

# Handle input
if submitted and user_input.strip():
    now = datetime.now().strftime("%I:%M %p")

    # Prevent repeated messages
    last_user_msgs = [msg for msg in st.session_state.conversation if msg["role"] == "user"]
    if last_user_msgs and last_user_msgs[-1]["content"] == user_input:
        st.session_state.conversation.append({
            "role": "assistant",
            "content": "We appear to be circling the same topic, Sir/Madam. Might you wish to inquire about something else?",
            "timestamp": now
        })
    else:
        # Add user message
        st.session_state.conversation.append({
            "role": "user",
            "content": user_input,
            "timestamp": now
        })
        # Get AI response
        ai_response = get_response(user_input)
        st.session_state.conversation.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })

    st.rerun()
