# Job Application Assistant

A goal-based AI agent that chats with an applicant until it has collected their **name**, **email** and **skills**. You can also upload a resume (PDF or TXT) to pre-fill the fields.

- **Frontend:** Streamlit chat UI (`frontend.py`)
- **Backend:** FastAPI (`backend.py`)
- **Agent:** LangChain/LangGraph agent on Google Gemini (`goal_based_agent.py`)

## How it works

The agent has one goal: every field in `name`, `email`, `skills` is filled with valid data. It has tools to save each field (the email is validated) and to check what is still missing. It asks for one missing item at a time and stops when the goal is complete. Conversation memory is kept per session.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file with your Google AI API key:

```
GOOGLE_API_KEY=your-key-here
```

## Run

Start the backend and the frontend in two terminals:

```bash
uvicorn backend:app --reload
streamlit run frontend.py
```

Then open http://localhost:8501. The frontend expects the backend at `http://localhost:8000` (`API_URL` in `frontend.py`).

You can also try the agent in the terminal with `python goal_based_agent.py`.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/chat` | Send a message: `{"session_id", "message"}` |
| GET | `/state/{session_id}` | Collected info and missing fields |
| POST | `/resume/{session_id}` | Upload a PDF/TXT resume to pre-fill fields |
| POST | `/reset/{session_id}` | Clear a session |

Interactive docs are at http://localhost:8000/docs.

## Configuration

The model is set in `goal_based_agent.py`. It uses `gemini-3.5-flash-lite` with `thinking_level="minimal"` for fast replies (about 2 seconds per message). Gemini 3 models take `thinking_level`, not `thinking_budget`.
