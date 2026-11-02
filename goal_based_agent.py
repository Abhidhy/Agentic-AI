import re

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()  # reads GOOGLE_API_KEY from .env

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, thinking_level="minimal")

FIELDS = ("name", "email", "skills")
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")

# ------------------------------------------------------------ goal state
# One goal per session: every field in FIELDS is filled with valid data.
sessions: dict[str, dict] = {}


def get_info(session_id: str) -> dict:
    return sessions.setdefault(session_id, {k: None for k in FIELDS})


def missing_fields(session_id: str) -> list[str]:
    return [k for k, v in get_info(session_id).items() if not v]


def goal_achieved(session_id: str) -> bool:
    return not missing_fields(session_id)


def check_status(session_id: str) -> str:
    if goal_achieved(session_id):
        return "GOAL COMPLETE: all information collected."
    return f"Still missing: {', '.join(missing_fields(session_id))}."


def _sid(config: RunnableConfig) -> str:
    return config["configurable"]["thread_id"]


# ----------------------------------------------------------------- tools
@tool
def save_name(name: str, config: RunnableConfig) -> str:
    """Save the applicant's full name."""
    if len(name.strip()) < 2:
        return "Invalid name. Ask the user again."
    get_info(_sid(config))["name"] = name.strip().title()
    return f"Name saved. {check_status(_sid(config))}"


@tool
def save_email(email: str, config: RunnableConfig) -> str:
    """Save the applicant's email address."""
    if not EMAIL_RE.match(email.strip()):
        return "Invalid email format. Ask the user for a valid email."
    get_info(_sid(config))["email"] = email.strip()
    return f"Email saved. {check_status(_sid(config))}"


@tool
def save_skills(skills: str, config: RunnableConfig) -> str:
    """Save the applicant's skills as a comma-separated string."""
    if not skills.strip():
        return "No skills given. Ask the user again."
    get_info(_sid(config))["skills"] = skills.strip()
    return f"Skills saved. {check_status(_sid(config))}"


@tool
def check_application_goal(config: RunnableConfig) -> str:
    """Check which application fields are still missing."""
    return check_status(_sid(config))


# ----------------------------------------------------------------- agent
SYSTEM_PROMPT = """You are a job application assistant.
Your GOAL is to collect the user's name, email, and skills.
- Whenever the user gives any of these, immediately call the matching save_* tool.
- The save_* tools already report what is still missing, so do not call check_application_goal after saving.
- Ask for ONE missing item at a time, politely.
- When the tools report GOAL COMPLETE, thank the user, summarise what was collected, and stop asking questions.
"""

agent = create_agent(
    model=llm,
    tools=[save_name, save_email, save_skills, check_application_goal],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),  # conversation memory, keyed by thread_id
)


# ------------------------------------------------------------- public API
def state(session_id: str) -> dict:
    return {
        "info": dict(get_info(session_id)),
        "missing": missing_fields(session_id),
        "complete": goal_achieved(session_id),
    }


def chat(session_id: str, message: str) -> dict:
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)
    return {"reply": result["messages"][-1].text, **state(session_id)}


def set_info(session_id: str, **fields: str | None) -> dict:
    """Pre-fill fields (e.g. from a resume). Invalid or empty values are ignored."""
    info = get_info(session_id)
    for key, value in fields.items():
        if key not in FIELDS or not value:
            continue
        if key == "email" and not EMAIL_RE.match(value.strip()):
            continue
        info[key] = value.strip()
    return state(session_id)


def reset(session_id: str) -> None:
    sessions.pop(session_id, None)
    agent.checkpointer.delete_thread(session_id)


# ------------------------------------------------------------------- CLI
def main() -> None:
    sid = "cli"
    print("📝 Hi! I'm your job application assistant. Please tell me your name, email, and skills.")
    for _ in range(15):  # safety limit
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("👋 Bye! Good luck.")
            return
        out = chat(sid, user_input)
        print("Bot:", out["reply"])
        if out["complete"]:
            print("🎉 Application info complete!", out["info"])
            return
    print("⚠️ Turn limit reached. Missing:", missing_fields(sid))


if __name__ == "__main__":
    main()
