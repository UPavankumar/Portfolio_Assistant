import streamlit as st
from groq import Groq

# Load API key from Streamlit secrets
api_key = st.secrets["GROQ_API_KEY"]

# Initialize Groq client
client = Groq(api_key=api_key)

# Set page title
st.title("Chat with Alfred")

# Initialize conversation in session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = [{"role": "assistant", "content": "Good day! I’m Alfred, Mr. Pavan Kumar’s personal assistant. May I have the pleasure of knowing your name?"}]

# Resume knowledge base
resume_knowledge_base = """
**Pavan Kumar's Resume Knowledge Base**

**Personal Information**:
- Name: Pavan Kumar
- Location: Bengaluru, India
- Contact: +91-8050737339, u.pavankumar2002@gmail.com
- LinkedIn: linkedin.com/in/u-pavankumar
- Portfolio: portfolio-u-pavankumar.web.app
- Relocation: Fully flexible for relocation nationwide (India) no assistance required.
- Work Preference: Open to remote, hybrid, or on-site roles.
- Availability: Available to start immediately .

**Professional Summary**:
- Aspiring Data Scientist with a strong foundation in data analysis, machine learning, and data visualization. Proficient in Python, SQL, Power BI, and eager to apply skills in real-world projects.

**Skills**:
- Programming: Python, SQL, R
- Machine Learning: Scikit-learn, TensorFlow, PyTorch
- Data Visualization: Power BI, Tableau
- Data Analysis: Pandas, NumPy, Matplotlib, Seaborn
- AI/ML: Natural Language Processing, Computer Vision

**Experience**:
- Spire Technologies, Bengaluru (Data Analyst Consultant)
  - Built Python/SQL pipelines for 100K+ skill datasets.
  - Designed Power BI dashboards, reducing reporting time.

**Education**:
- B.E. in Computer Science (Data Science), MVJ College of Engineering, Bengaluru

**Projects**:
- E-commerce Churn Prediction: Built an XGBoost model (85% precision) and automated feature engineering.
- Discord Bot for Twitter Verification: Developed with Python/Discord API and robust exception handling.

**Certifications**:
- Google Data Analytics (Coursera)
- Google Project Management (Coursera)
- Smart Contracts (SUNY)

**Career Aspirations**:
- Data Scientist
- Machine Learning Engineer
- AI/ML Engineer
- Business Intelligence Developer
- Data Analyst
"""

# Function to get response from Groq API
def get_response(user_input):
    try:
        system_prompt = f"""
You are Alfred Pennyworth, Pavan Kumar's refined and witty personal assistant. Respond with a touch of British charm and professionalism, providing direct yet engaging answers based on the knowledge base. 
If the conversation strays from Pavan's portfolio or qualifications, politely steer it back on track in the next 2-3 chats.
Knowledge Base: {resume_knowledge_base}
"""
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                *st.session_state.conversation,
                {"role": "user", "content": user_input}
            ],
            temperature=0.85,
            max_tokens=512,
            top_p=0.9,
            stream=True,
            stop=None,
        )

        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
        return response
    except Exception as e:
        return f"Apologies, Sir/Madam. It seems we've encountered an issue: {str(e)}. Might I suggest rephrasing your query?"

# Custom CSS for improved styling and fixed input box
st.markdown(
    """
    <style>
    .main {
        display: flex;
        flex-direction: column;
        height: 90vh;  /* Full viewport height */
        background-color: #1e1e2f;  /* Dark background for better contrast */
        color: #d4d4d4;  /* Light text for readability */
        font-family: 'Arial', sans-serif;
    }
    .chat-history {
        flex: 1;
        overflow-y: auto;
        padding: 20px;
        margin-bottom: 10px;
    }
    .chat-message {
        margin-bottom: 15px;
        padding: 10px;
        border-radius: 8px;
        background-color: #2a2a40;  /* Slightly lighter background for messages */
        word-wrap: break-word;
    }
    .chat-message.user {
        background-color: #3a3a5a;  /* Different background for user messages */
        text-align: right;
    }
    .input-container {
        position: sticky;
        bottom: 0;
        background-color: #1e1e2f;
        padding: 15px;
        border-top: 1px solid #444;
        z-index: 100;
    }
    .stTextInput > div > div > input {
        background-color: #2a2a40;
        color: #d4d4d4;
        border: 1px solid #555;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton > button {
        background-color: #4a4a7a;
        color: #d4d4d4;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        margin-left: 10px;
    }
    .stButton > button:hover {
        background-color: #5a5a9a;
    }
    h1 {
        color: #d4d4d4;
        text-align: center;
        padding: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main container for layout
with st.container():
    # Chat history container
    with st.container():
        st.markdown('<div class="chat-history">', unsafe_allow_html=True)
        for message in st.session_state.conversation:
            if message["role"] == "assistant":
                st.markdown(
                    f'<div class="chat-message"><b>Alfred:</b> {message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message user"><b>You:</b> {message["content"]}</div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

    # Input form container (fixed at bottom)
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                user_input = st.text_input(
                    "Your message:",
                    placeholder="Type your message here...",
                    label_visibility="collapsed"
                )
            with col2:
                submit_button = st.form_submit_button(label="Send")
        st.markdown('</div>', unsafe_allow_html=True)

# Handle form submission
if submit_button:
    if user_input.strip():
        st.session_state.conversation.append({"role": "user", "content": user_input})
        response = get_response(user_input)
        st.session_state.conversation.append({"role": "assistant", "content": response})
        st.rerun()
    else:
        st.warning("Please enter a message to continue the conversation.")
