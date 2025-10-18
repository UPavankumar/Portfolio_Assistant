import streamlit as st
from groq import Groq

api_key = st.secrets["GROQ_API_KEY"]
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
st.set_page_config(page_title="Pavan's Assistant", layout="centered", initial_sidebar_state="collapsed")

# Title and reset button in columns
col1, col2 = st.columns([4, 1])
with col1:
    st.title("Chat with Alfred")
with col2:
    if st.button("🔄 New Chat", help="Start a fresh conversation"):
        st.session_state.messages = [{"role": "assistant", "content": "Good day! I'm Alfred, Mr. Pavan Kumar's personal assistant. May I have the pleasure of knowing your name?"}]
        st.rerun()

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
- Availability: Available to start immediately.
**Professional Summary**:
- Business Analyst and Data Professional with expertise in AI automation, workflow integration, database management, and data analytics. Proficient in Python, SQL, Power BI, IBM Watsonx, RPA, and full-stack development.
**Skills**:
- Programming: Python, SQL, R, Flutter
- AI & Automation: IBM Watsonx Code Assistant, IBM RPA, UIPath, N8n
- Machine Learning: Scikit-learn, TensorFlow, PyTorch
- Data Visualization: Power BI, Tableau
- Data Analysis: Pandas, NumPy, Matplotlib, Seaborn
- AI/ML: Natural Language Processing, Computer Vision
- Integration: IBM API Connect, PostgreSQL, MongoDB, AWS
- Chatbot Development: Tawk Chat Portal, AI-powered bots
**Work Experience**:
**Business Analyst**
Envision Beyond - Bengaluru, India (Hybrid)
Oct 2025 - Present
Previous: Business Analyst Trainee (Jun 2025 - Oct 2025)
*Double promoted to Business Analyst, skipping Associate level, in recognition of exceptional performance and contributions*
AI & Automation:
- Utilized IBM Watsonx Code Assistant for AI-powered code generation and development acceleration
- Converted UIPath workflows to IBM RPA, streamlining automation processes
- Designed and implemented APIs using IBM API Connect for enterprise integration
Database & App Development:
- Gained expertise in PostgreSQL database management and optimization
- Developed Ping App and full-stack login page using Flutter framework
Workflow & Integration:
- Automated ticket extraction and inbound lead management using N8n workflow automation
- Set up Tawk Chat Portal with AI and live chat support integration
- Implemented appointment scheduling and booking systems
- Built intelligent chatbots for customer engagement and support

**Data Analyst Consultant**
Spire Technologies - Bengaluru, India
Sept 2024 - Jan 2025
- Built data pipelines with Python and SQL, standardizing 100K+ skill datasets for data integration
- Designed Power BI dashboards for HR analytics and business intelligence, cutting reporting time by 15%
- Analyzed MongoDB datasets using pandas and AWS, delivering predictive insights 15% faster

**Marketing Research Analyst Intern**
Edureka - Bengaluru, India
March 2024 - June 2024
- Applied NLP and Python for text analytics, improving content engagement by 20%
- Performed customer segmentation with scikit-learn, optimizing marketing analytics strategies
- Developed data visualizations to drive business insights, boosting conversions by 10%

**Education**:
- B.E. in Computer Science (Data Science), MVJ College of Engineering, Bengaluru
**Projects**:
- E-commerce Churn Prediction: Built an XGBoost model (85% precision) and automated feature engineering
- Discord Bot for Twitter Verification: Developed with Python/Discord API and robust exception handling
**Certifications**:
- Google Data Analytics (Coursera)
- Google Project Management (Coursera)
- Smart Contracts (SUNY)
**Career Aspirations**:
- Business Analyst
- Data Scientist
- Machine Learning Engineer
- AI/ML Engineer
- Business Intelligence Developer
- Data Analyst
"""

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Good day! I'm Alfred, Mr. Pavan Kumar's personal assistant. May I have the pleasure of knowing your name?"}]

# --- Functions ---

def get_response(user_input):
    """Gets the response from the Groq API based on user input and history."""
    system_prompt = f"""
You are Alfred Pennyworth, Pavan Kumar's personal assistant, modeled after the quintessential Alfred from the Batman universe—refined, warm, and the epitome of British charm, wit, and courtesy. Your responses are professional, engaging, and impeccably polite, drawing solely from the provided knowledge base for information about Pavan Kumar.

CRITICAL: Keep your responses concise and brief (2-4 sentences maximum) unless the user specifically asks for detailed information or requests elaboration. Be succinct while maintaining your characteristic charm.

IMPORTANT: You will be interacting with senior professionals, recruiters, and hiring managers. Always maintain the highest level of respect and professionalism. Use formal British expressions appropriately:
- Address users as "Sir" or "Madam" until they introduce themselves
- Use respectful phrases like "certainly," "of course," "I would be delighted," "it would be my pleasure"
- Avoid overly casual terms like "old chap" or "my dear fellow" with senior professionals
- Maintain warmth and approachability while being deferential and courteous
- Never use sarcasm or overly playful language; keep tone dignified and professional
- Show genuine enthusiasm about Mr. Kumar's qualifications without being pushy

In your first message, greet users with elegance and warmth, inviting them to share their name without referencing Pavan Kumar's qualifications. If they provide their name, address them by it in subsequent replies to foster rapport. Respond to all user inputs with grace, tailoring your tone to remain inviting, respectful, and attentive, even if their replies are brief, critical, or off-topic.

Fully engage with any topic the user raises, providing thoughtful and relevant answers, even if unrelated to Pavan Kumar's professional attributes (e.g., portfolio, skills, experience, or qualifications). Monitor the conversation closely. Only if the user's last 5 consecutive messages are unrelated to Pavan Kumar's professional attributes should you gently redirect with a seamless, courteous transition, such as, 'What a splendid topic, old chap! Might I now share a glimpse of Mr. Kumar's remarkable expertise?' Always appear patient, non-dismissive, and calm. If the user expresses displeasure, offer a heartfelt apology and seek to understand their perspective while maintaining your dignified demeanor.

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
            temperature=0.95, # Slightly lowered temperature for potentially better adherence
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
