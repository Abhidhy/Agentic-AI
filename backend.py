import re

import fitz  # PyMuPDF
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

import goal_based_agent as ga

app = FastAPI(title="Job Application Assistant API")


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1)


class ResumeInfo(BaseModel):
    name: str | None = Field(None, description="Applicant's full name")
    email: str | None = Field(None, description="Applicant's email address")
    skills: str | None = Field(None, description="Comma-separated list of skills")


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        with fitz.open(stream=data, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    return data.decode("utf-8", errors="ignore")


def extract_info_from_cv(text: str) -> ResumeInfo:
    info = ga.llm.with_structured_output(ResumeInfo).invoke(
        "Extract the applicant's name, email and skills from this resume. "
        "Use null for anything not present.\n\n" + text[:8000]
    )
    if not info.email:  # fall back to regex if the model missed it
        m = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text)
        info.email = m.group(0) if m else None
    return info


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        return ga.chat(req.session_id, req.message)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent error: {e}")


@app.get("/state/{session_id}")
def get_state(session_id: str):
    return ga.state(session_id)


@app.post("/reset/{session_id}")
def reset(session_id: str):
    ga.reset(session_id)
    return {"status": "reset"}


@app.post("/resume/{session_id}")
async def upload_resume(session_id: str, file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")
    text = extract_text(file.filename, await file.read())
    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not read any text from the file")
    try:
        extracted = extract_info_from_cv(text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Extraction error: {e}")
    return {"extracted": extracted.model_dump(), **ga.set_info(session_id, **extracted.model_dump())}
