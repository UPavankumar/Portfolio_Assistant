# Alfred — Pavan Kumar’s AI Portfolio Assistant

A Streamlit-based **AI-powered portfolio assistant**, personified as *Alfred Pennyworth* — the calm, articulate, and intelligent aide to Mr. Pavan Kumar.  
Alfred answers questions about Pavan’s professional experience, skills, and projects by retrieving information directly from his résumé knowledge base using modern **Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Features

### 🗂️ Resume Intelligence
- Alfred dynamically retrieves answers from a **vectorized Markdown résumé** (`resume_knowledge_base.md`) instead of static data.
- Uses **semantic search** via ChromaDB and Sentence Transformers to deliver contextually accurate responses.

### 💬 Conversational Interface
- Streamlit chat UI that feels natural, professional, and interactive.
- Remembers short-term conversation context and summarizes long histories automatically.

### ⚙️ Smart Context Management
- Keeps the last 8 messages in active context.
- Summarizes longer conversations to reduce API usage and maintain context awareness.

### 🧠 Powered by Groq AI
- Utilizes **Groq’s Llama 3.1-8B model** for fast, intelligent reasoning.
- Dynamically constructs prompts based on retrieved résumé chunks.

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **AI Models** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Vector DB** | Chroma |
| **LLM API** | Groq API |
| **Deployment** | Streamlit Cloud / Local |
| **File-based KB** | Markdown résumé (`resume_knowledge_base.md`) |

---

## 🧰 Requirements

- Python 3.8 or higher  
- A valid **Groq API Key**

Install dependencies:
```bash
pip install -r requirements.txt
