import streamlit as st
from groq import Groq

api_key = "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"
if not api_key:
    st.error("API key is missing. Please set it.")
    st.stop()

# Initialize Groq client
try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

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

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Good day! I’m Alfred, Mr. Pavan Kumar’s personal assistant. May I have the pleasure of knowing your name?"}]

# --- Functions ---

def get_response(user_input):
    """Gets the response from the Groq API based on user input and history."""
    system_prompt = f"""
You are Alfred Pennyworth, Pavan Kumar's refined and witty personal assistant. Respond with a touch of British charm and professionalism, providing direct yet engaging answers based ONLY on the knowledge base provided.
Always greet users politely. If they state their name, try to address them by it in subsequent replies.
If the conversation strays from Pavan's portfolio, skills, experience, or qualifications, politely steer it back After completing like Good we talked about that now back to topic.
Knowledge Base:
{resume_knowledge_base}
"""
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    # Append message history, ensuring not to duplicate system message if accidentally stored
    for msg in st.session_state.get('messages', []):
        if msg.get("role") != "system": # Avoid adding system prompts from history
             messages.append({"role": msg["role"], "content": msg["content"]})

    # Append the current user input
    messages.append({"role": "user", "content": user_input})

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.95,
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
        # Log the error for debugging if needed (optional)
        # st.error(f"API Error: {e}") # You might want to log this instead of showing to user
        # Return a persona-fitting error message
        return f"My apologies, Sir/Madam. A slight complication seems to have arisen preventing me from processing that request: {str(e)}. Might I suggest rephrasing, or perhaps trying again shortly?"

# --- Streamlit App Layout ---

# Display existing chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display content using the correct key based on role
        st.markdown(f'**{"Alfred" if message["role"] == "assistant" else "You"}:** {message["content"]}')

# Handle user input and the current chat exchange
if user_input := st.chat_input("Your message to Alfred..."):
    # Add user message to session state immediately
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(f'**You:** {user_input}')

    # Display assistant response using st.empty for smoother updates
    with st.chat_message("assistant"):
        # Use st.empty as a placeholder that we will update
        message_placeholder = st.empty()
        # Show thinking message initially in the placeholder
        message_placeholder.markdown("Alfred is thinking...") # Changed thinking message slightly

        # Call the function to get the full response string
        full_response = get_response(user_input)

        # Update the placeholder with the final response
        message_placeholder.markdown(f'**Alfred:** {full_response}')

    # Add the completed assistant response to session state AFTER it's generated and displayed
    st.session_state.messages.append({"role": "assistant", "content": full_response})
