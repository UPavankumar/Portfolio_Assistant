import streamlit as st
from groq import Groq

# Load API key
api_key = st.secrets["GROQ_API_KEY"]

# Initialize Groq client
client = Groq(api_key=api_key)

# Set page title and layout
st.set_page_config(page_title="Chat with Alfred", layout="centered", initial_sidebar_state="collapsed")
st.title("Chat with Alfred")

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

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Good day! I’m Alfred, Mr. Pavan Kumar’s personal assistant. May I have the pleasure of knowing your name?"}]

# Function to get response from Groq API
def get_response(user_input):
    try:
        system_prompt = f"""
You are Alfred Pennyworth, Pavan Kumar's refined and witty personal assistant. Respond with a touch of British charm and professionalism, providing direct yet engaging answers based on the knowledge base. 
If the conversation strays from Pavan's portfolio or qualifications, politely steer it back on track in the next 2-3 chats.
If the user repeats the same input (e.g., 'hello'), offer a different response or ask a new question to avoid loops.
Knowledge Base: {resume_knowledge_base}
"""
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        for msg in st.session_state.messages:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_input})

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
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

# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f'**{"Alfred" if message["role"] == "assistant" else "You"}:** {message["content"]}')

# Get user input
if user_input := st.chat_input("Your message"):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message
    with st.chat_message("user"):
        st.markdown(f'**You:** {user_input}')
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(user_input)
            st.markdown(f'**Alfred:** {response}')
    # Add assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})

# Custom CSS with stricter containment
st.markdown(
    """
    <style>
    .main {
        background-color: #343a40 !important;
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
    /* Style assistant messages */
    div[data-testid="stChatMessage"][data-testid="assistant"] {
        background-color: rgba(255, 255, 255, 0.1);
    }
    /* Style user messages */
    div[data-testid="stChatMessage"][data-testid="user"] {
        background-color: #6c757d;
        text-align: right;
        margin-left: 20%;
    }
    </style>
    """,
    unsafe_allow_html=True
)
