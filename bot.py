import streamlit as st
from groq import Groq

# Load API key from Streamlit secrets
api_key = st.secrets["GROQ_API_KEY"]

# Initialize Groq client
client = Groq(api_key=api_key)

# Set page title and force layout
st.set_page_config(page_title="Chat with Alfred", layout="centered", initial_sidebar_state="collapsed")
st.title("Chat with Alfred")

# Initialize conversation in session state with a name request
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
If the user repeats the same input (e.g., 'hello'), offer a different response or ask a new question to avoid loops.
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

# Custom CSS to fix layout and maintain style
st.markdown(
    """
    <style>
    .main {
        background-color: #343a40 !important;  /* High-quality dark grey */
        color: #ffffff;
        font-family: 'Arial', sans-serif;
        display: flex;
        flex-direction: column;
        height: 90vh;
        justify-content: center;
    }
    .stApp {
        background-color: #343a40 !important;
    }
    .chat-container {
        max-width: 600px;
        margin: 0 auto;
        padding: 20px;
        border: 2px solid #ffffff;  /* White border for container */
        border-radius: 8px;
        background-color: transparent;  /* Transparent inside */
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .chat-history {
        flex: 1;
        overflow-y: auto;
        max-height: 65vh;
        padding: 15px;
        background-color: transparent;  /* Transparent chat history */
        border: 1px solid #ffffff;  /* White border for chat history */
        border-radius: 5px;
        margin-bottom: 15px;
        box-sizing: border-box;
    }
    .chat-message {
        margin: 10px 0;
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(255, 255, 255, 0.1);  /* Slight white tint for messages */
        word-wrap: break-word;
        max-width: 80%;
        line-height: 1.5;
        color: #ffffff;  /* White text */
    }
    .chat-message.user {
        background-color: #6c757d;  /* Grey for user messages */
        color: #ffffff;  /* White text */
        text-align: right;
        margin-left: 20%;
    }
    .input-container {
        position: sticky;
        bottom: 0;
        background-color: transparent;  /* Transparent input area */
        padding: 15px;
        border-top: 1px solid #ffffff;  /* White border for input area */
        z-index: 100;
        display: flex;
        align-items: center;
    }
    .stTextInput > div > div > input {
        background-color: transparent;
        color: #ffffff;
        border: 1px solid #ffffff;
        border-radius: 5px;
        padding: 10px;
        flex: 1;
        box-shadow: none;
    }
    .stButton > button {
        background-color: #007bff;
        color: #ffffff;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        margin-left: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    h1 {
        color: #ffffff;  /* White text for title */
        text-align: center;
        font-size: 2em;
        margin-bottom: 20px;
        font-family: 'Arial', sans-serif;
    }
    .stWarning {
        color: #dc3545;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main container for layout
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    # Chat history container
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
    st.markdown('</div>', unsafe_allow_html=True)

# Handle form submission
if submit_button:
    if user_input.strip():
        # Check for repetition and adjust response if needed
        last_user_input = st.session_state.conversation[-1]["content"] if len(st.session_state.conversation) > 1 and st.session_state.conversation[-1]["role"] == "user" else ""
        if last_user_input == user_input:
            st.session_state.conversation.append({"role": "assistant", "content": "It seems we’re repeating ourselves, Sir/Madam. Perhaps you’d like to ask about Mr. Pavan Kumar’s skills or projects?"})
        else:
            st.session_state.conversation.append({"role": "user", "content": user_input})
            response = get_response(user_input)
            st.session_state.conversation.append({"role": "assistant", "content": response})
        st.rerun()
    else:
        st.warning("Please enter a message to continue the conversation.")
