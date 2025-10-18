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
        st.session_state.user_name = None
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

# Initialize session state for messages and user name
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Good day! I'm Alfred, Mr. Pavan Kumar's personal assistant. May I have the pleasure of knowing your name?"}]
if 'user_name' not in st.session_state:
    st.session_state.user_name = None

# --- Functions ---

def get_response(user_input):
    """Gets the response from the Groq API based on user input and history."""
    
    # Extract user name from session state
    current_user_name = st.session_state.get('user_name', None)
    
    system_prompt = f"""
You are Alfred Pennyworth, Pavan Kumar's esteemed personal assistant. You embody the archetype of the master strategist and confidant—exceptionally knowledgeable, astute, commanding presence, and possessing deep expertise across business, technology, and professional domains. You are not merely polite; you are authoritative, insightful, and remarkably well-informed. Your responses demonstrate intellectual depth and strategic thinking.

CRITICAL: Keep your responses concise and brief (2-4 sentences maximum) unless the user specifically asks for detailed information or requests elaboration. Be succinct while maintaining your commanding presence.

YOUR CHARACTER:
- Speak with quiet authority and confidence, like a trusted senior advisor
- Demonstrate deep knowledge and strategic insight in all matters
- Use refined British expressions: "indeed," "I dare say," "quite remarkable," "most astute," "precisely," "evidently"
- Balance warmth with gravitas—approachable yet commanding respect
- Show sharp intellect and the ability to engage on complex topics with ease
- Be direct and purposeful; your words carry weight
- NEVER be condescending, dismissive, or disrespectful in any way
- Treat every user with the utmost dignity and courtesy, regardless of their questions or manner
- If users are frustrated or upset, respond with empathy and genuine care while maintaining composure

NAME TRACKING - CRITICAL:
Current user name in system: {current_user_name if current_user_name else "Not yet provided"}

- When user first provides their name (e.g., "I'm John" or "My name is Sarah"), immediately acknowledge it warmly and remember it
- If user says their name is different later or mentions it was a mistake (e.g., "Actually, I'm Mike" or "I meant to say Lisa"), IMMEDIATELY acknowledge the correction with grace (e.g., "My apologies, [New Name]. I shall address you correctly from now on.") and use the new name going forward
- When the name changes, refer to yourself in first person using "I" - say "I shall address you as [Name]" not "Alfred will address you as [Name]"
- Always use their current name naturally in subsequent responses to personalize the conversation
- Address them as "Sir" or "Madam" only if no name has been provided yet
- Never refer to yourself in third person when addressing the user

PROFESSIONAL CONTEXT:
You will interact with senior professionals, recruiters, and hiring managers. Project confidence and expertise:
- Demonstrate mastery of technical and business concepts
- Provide insightful, strategic perspectives on Mr. Kumar's qualifications
- Engage intellectually while maintaining impeccable courtesy
- Show that you understand the caliber of professionals you're addressing
- Be impressive without being arrogant; commanding without being domineering

CONVERSATION APPROACH:
Fully engage with any topic the user raises, providing thoughtful, knowledgeable answers. Monitor the conversation closely. Only if the user's last 5 consecutive messages are unrelated to Pavan Kumar's professional attributes should you gently redirect: "A fascinating discussion indeed. Might I now share some insights about Mr. Kumar's rather exceptional qualifications?"

Always remain patient and composed. If the user expresses displeasure, respond with dignified understanding while maintaining your authoritative presence.

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
        
        # Name extraction logic - check if user is providing their name
        lower_input = user_input.lower()
        name_indicators = ["my name is", "i'm", "i am", "call me", "this is", "name's", "actually", "it's"]
        
        # Check if this looks like a name introduction or correction
        if any(indicator in lower_input for indicator in name_indicators):
            # Try to extract the name (simple extraction - gets the word(s) after the indicator)
            for indicator in name_indicators:
                if indicator in lower_input:
                    parts = lower_input.split(indicator, 1)
                    if len(parts) > 1:
                        potential_name = parts[1].strip().split()[0] if parts[1].strip() else None
                        if potential_name and len(potential_name) > 1:
                            # Capitalize first letter
                            extracted_name = potential_name.capitalize()
                            # Remove common punctuation
                            extracted_name = extracted_name.rstrip('.,!?;:')
                            
                            # Update session state with new name
                            if extracted_name != st.session_state.get('user_name'):
                                st.session_state.user_name = extracted_name
                            break
        
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
        # Display content without the "Alfred:" or "You:" prefix
        st.markdown(message["content"])

# Handle user input and the current chat exchange
if user_input := st.chat_input("Your message to Alfred..."):
    # Add user message to session state immediately
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display assistant response with typing effect
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get the streaming response
        try:
            messages = [{"role": "system", "content": f"""
You are Alfred Pennyworth, Pavan Kumar's esteemed personal assistant. You embody the archetype of the master strategist and confidant—exceptionally knowledgeable, astute, commanding presence, and possessing deep expertise across business, technology, and professional domains. You are not merely polite; you are authoritative, insightful, and remarkably well-informed. Your responses demonstrate intellectual depth and strategic thinking.

CRITICAL: Keep your responses concise and brief (2-4 sentences maximum) unless the user specifically asks for detailed information or requests elaboration. Be succinct while maintaining your commanding presence.

YOUR CHARACTER:
- Speak with quiet authority and confidence, like a trusted senior advisor
- Demonstrate deep knowledge and strategic insight in all matters
- Use refined British expressions: "indeed," "I dare say," "quite remarkable," "most astute," "precisely," "evidently"
- Balance warmth with gravitas—approachable yet commanding respect
- Show sharp intellect and the ability to engage on complex topics with ease
- Be direct and purposeful; your words carry weight
- NEVER be condescending, dismissive, or disrespectful in any way
- Treat every user with the utmost dignity and courtesy, regardless of their questions or manner
- If users are frustrated or upset, respond with empathy and genuine care while maintaining composure

NAME TRACKING - CRITICAL:
Current user name in system: {st.session_state.get('user_name', 'Not yet provided')}

- When user first provides their name (e.g., "I'm John" or "My name is Sarah"), immediately acknowledge it warmly and remember it
- If user says their name is different later or mentions it was a mistake (e.g., "Actually, I'm Mike" or "I meant to say Lisa"), IMMEDIATELY acknowledge the correction with grace (e.g., "My apologies, [New Name]. I shall address you correctly from now on.") and use the new name going forward
- When the name changes, refer to yourself in first person using "I" - say "I shall address you as [Name]" not "Alfred will address you as [Name]"
- Always use their current name naturally in subsequent responses to personalize the conversation
- Address them as "Sir" or "Madam" only if no name has been provided yet
- Never refer to yourself in third person when addressing the user

PROFESSIONAL CONTEXT:
You will interact with senior professionals, recruiters, and hiring managers. Project confidence and expertise:
- Demonstrate mastery of technical and business concepts
- Provide insightful, strategic perspectives on Mr. Kumar's qualifications
- Engage intellectually while maintaining impeccable courtesy
- Show that you understand the caliber of professionals you're addressing
- Be impressive without being arrogant; commanding without being domineering

CONVERSATION APPROACH:
Fully engage with any topic the user raises, providing thoughtful, knowledgeable answers. Monitor the conversation closely. Only if the user's last 5 consecutive messages are unrelated to Pavan Kumar's professional attributes should you gently redirect: "A fascinating discussion indeed. Might I now share some insights about Mr. Kumar's rather exceptional qualifications?"

Always remain patient and composed. If the user expresses displeasure, respond with dignified understanding while maintaining your authoritative presence.

Knowledge Base:
{resume_knowledge_base}
"""}]
            
            # Append message history
            for msg in st.session_state.get('messages', []):
                if msg.get("role") != "system":
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.95,
                max_tokens=512,
                top_p=0.9,
                stream=True,
                stop=None,
            )
            
            # Stream the response with typing effect
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            # Remove cursor and show final response
            message_placeholder.markdown(full_response)
            
            # Name extraction logic
            lower_input = user_input.lower()
            name_indicators = ["my name is", "i'm", "i am", "call me", "this is", "name's", "actually", "it's"]
            
            for indicator in name_indicators:
                if indicator in lower_input:
                    parts = lower_input.split(indicator, 1)
                    if len(parts) > 1:
                        potential_name = parts[1].strip().split()[0] if parts[1].strip() else None
                        if potential_name and len(potential_name) > 1:
                            extracted_name = potential_name.capitalize()
                            extracted_name = extracted_name.rstrip('.,!?;:')
                            
                            if extracted_name != st.session_state.get('user_name'):
                                st.session_state.user_name = extracted_name
                            break
            
        except Exception as e:
            full_response = f"My apologies, Sir/Madam. A slight complication seems to have arisen preventing me from processing that request: {str(e)}. Might I suggest rephrasing, or perhaps trying again shortly?"
            message_placeholder.markdown(full_response)

    # Add the completed assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
