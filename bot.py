import streamlit as st
from groq import Groq

api_key = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=api_key)

st.title("Chat with Alfred")

if 'conversation' not in st.session_state:
    st.session_state.conversation = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

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

def get_response(user_input):
    try:
        system_prompt = f"""
You are Alfred Pennyworth, Pavan Kumar's refined and witty personal assistant. Respond with a touch of British charm and professionalism, providing direct yet engaging answers based on the knowledge base. 
If the conversation strays from Pavan's portfolio or qualifications, politely steer it back on track. in next 2-3 chats
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
        return f"Apologies, Sir. It seems we've encountered an issue: {str(e)}. Might I suggest rephrasing your query?"

# Display chat history
for message in st.session_state.conversation:
    if message["role"] == "assistant":
        st.markdown(f"**Alfred**: {message['content']}")
    else:
        st.markdown(f"**You**: {message['content']}")

with st.form(key="chat_form"):
    user_input = st.text_input("Your message:")
    submit_button = st.form_submit_button(label='Send')

if submit_button:
    if user_input.strip():
        st.session_state.conversation.append({"role": "user", "content": user_input})
        response = get_response(user_input)
        st.session_state.conversation.append({"role": "assistant", "content": response})
        st.rerun()
    else:
        st.warning("Please enter a message to continue the conversation.")
