# Portfolio Assistant

A Streamlit chat app that answers recruiters' and hiring managers' questions about Pavan Kumar's experience, skills, and projects. Every answer is grounded in a single Markdown knowledge base, and nothing about the candidate is hardcoded in the app.

## Features

- **Driven by the knowledge base:** the candidate's name, title, and contact links (shown in the sidebar) are parsed from `resume_knowledge_base.md`. Edit that file and the app updates; no code changes are needed.
- **Grounded, professional answers:** the assistant answers only from the profile, says when something isn't covered, and keeps a neutral, professional tone.
- **Suggested questions:** starter questions are generated from the profile and cached until the file changes.
- **Date-aware:** durations such as "how long has he been in his current role" are calculated against today's date.
- **Resilient:** models are tried in order, falling back automatically when one is rate-limited or retired.
- **Token-efficient:**
  - The profile is stripped of Markdown formatting before it is sent.
  - The prompt is compact, and only a short window of recent messages is sent.
  - Reply length and input length are capped.
  - Failed replies are kept out of future prompts.

## Getting Started

**Prerequisites:** Python 3.9+ and a [Groq API key](https://console.groq.com/keys).

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` (never commit this file):

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Run the app:

```bash
streamlit run bot.py
```

On Streamlit Community Cloud, add the secrets under **App settings → Secrets**.

## Configuration

All settings are optional except `GROQ_API_KEY`. They are read from Streamlit secrets first, then from environment variables.

| Setting | Default | Purpose |
|---------|---------|---------|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `GROQ_MODELS` | `llama-3.3-70b-versatile,llama-3.1-8b-instant` | Models to try, in order of preference |
| `ASSISTANT_NAME` | `Alfred` | The assistant's display name |
| `KNOWLEDGE_BASE_PATH` | `resume_knowledge_base.md` | Profile file, relative to `bot.py` |
| `TEMPERATURE` | `0.3` | Lower values give more consistent, factual answers |
| `MAX_TOKENS` | `350` | Maximum reply length |
| `MAX_HISTORY_MESSAGES` | `6` | Recent messages sent as context |
| `MAX_INPUT_CHARS` | `500` | Maximum length of a visitor's message |
| `SUGGESTION_COUNT` | `4` | Number of suggested questions; `0` disables them |

## Knowledge Base Format

Plain Markdown. For automatic parsing, use a top-level `# Heading` and `- **Key:** value` fields. `Name` and `Title` are recognised, and any field whose value is a URL (`LinkedIn`, `GitHub`, `Portfolio`, …) appears as a sidebar link. An email address anywhere in the file is linked automatically.

## Project Structure

```
├── bot.py                    # Streamlit app
├── resume_knowledge_base.md  # Source of truth for all answers
├── requirements.txt
└── .devcontainer/            # Codespaces / Dev Container configuration
```
