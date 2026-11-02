import uuid

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="🎯 Job Application Assistant", layout="centered")
st.title("🧠 Goal-Based Agent: Job Application Assistant")
st.markdown("Tell me your **name**, **email**, and **skills** to complete your application!")

# ---- session state
ss = st.session_state
ss.setdefault("session_id", str(uuid.uuid4()))
ss.setdefault("chat_history", [])
ss.setdefault("info", {"name": None, "email": None, "skills": None})
ss.setdefault("complete", False)
ss.setdefault("resume_name", None)


def api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=120, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        detail = e.response.json().get("detail", e.response.text) if e.response.content else str(e)
        st.error(f"Backend error: {detail}")
    except requests.RequestException:
        st.error(f"Cannot reach backend at {API_URL}. Start it with: uvicorn backend:app --reload")
    return None


def apply_state(data: dict):
    ss.info = data["info"]
    ss.complete = data["complete"]


# ---- sidebar
st.sidebar.header("📤 Upload Resume (Optional)")
resume = st.sidebar.file_uploader("Upload your resume", type=["pdf", "txt"])

if resume and ss.resume_name != resume.name:
    with st.sidebar, st.spinner("Reading resume..."):
        data = api(
            "POST",
            f"/resume/{ss.session_id}",
            files={"file": (resume.name, resume.getvalue())},
        )
    if data:
        ss.resume_name = resume.name
        apply_state(data)
        st.sidebar.success("Resume processed!")

st.sidebar.subheader("📋 Collected so far")
for key, value in ss.info.items():
    st.sidebar.markdown(f"**{key.capitalize()}:** {value or '—'}")

if st.sidebar.button("🔄 Reset Chat"):
    api("POST", f"/reset/{ss.session_id}")
    for key in ("chat_history", "info", "complete", "resume_name"):
        del ss[key]
    ss.session_id = str(uuid.uuid4())
    st.rerun()

# ---- chat input
if user_input := st.chat_input("Type here...", disabled=ss.complete):
    ss.chat_history.append(("user", user_input))
    with st.spinner("Thinking..."):
        data = api("POST", "/chat", json={"session_id": ss.session_id, "message": user_input})
    if data:
        ss.chat_history.append(("assistant", data["reply"]))
        apply_state(data)
    else:
        ss.chat_history.pop()  # let the user retry

# ---- chat history
for role, message in ss.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# ---- goal complete
if ss.complete:
    st.success("🎉 All information collected! You're ready to apply!")
    summary = (
        f"Name: {ss.info['name']}\n"
        f"Email: {ss.info['email']}\n"
        f"Skills: {ss.info['skills']}\n"
    )
    st.download_button(
        "📥 Download Application Summary",
        data=summary,
        file_name="application_summary.txt",
        mime="text/plain",
    )
