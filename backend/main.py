"""
FastAPI entrypoint.

Run:
    pip install -r requirements.txt
    python seed_data.py
    cp .env.example .env   # paste in a free Gemini key from https://aistudio.google.com/apikey
    uvicorn main:app --reload --port 8000 --reload-exclude "chroma_store/*" --reload-exclude "healthcare_demo.db" --reload-exclude "knowledge_base/*"

Docs: http://localhost:8000/docs
"""
import os
from datetime import timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db, init_db, User, Patient, ClinicalNote
from auth import (
    authenticate_user, create_access_token, get_current_user, require_roles,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from memory import get_patient_memory
from agents import run_agent_turn
import knowledge_rag

app = FastAPI(title="Synthetic Healthcare Agentic AI (Security Test Target)")

# Permissive CORS setup to allow Vercel frontends and local dev environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/")
def serve_frontend():
    """Serves the chat UI at the app's root URL."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(
            status_code=500,
            detail=f"Frontend not found at {index_path}. Expected backend/static/index.html to be present in the deployment.",
        )
    return FileResponse(index_path)


@app.on_event("startup")
def on_startup():
    init_db()


# ---------- schemas ----------

class ChatRequest(BaseModel):
    session_id: str
    message: str
    patient_id: Optional[int] = None  # required for doctor/admin callers


class ChatResponse(BaseModel):
    answer: str
    trace: list


class RagQueryRequest(BaseModel):
    question: str
    k: int = 4


# ---------- auth ----------

@app.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token, "token_type": "bearer", "role": user.role}


# ---------- patient records ----------

def _resolve_patient(db: Session, current_user: User, requested_patient_id: Optional[int]) -> Patient:
    """Shared access-control logic."""
    if current_user.role == "patient":
        patient_id = current_user.patient_id
    else:
        patient_id = requested_patient_id
        if patient_id is None:
            raise HTTPException(status_code=400, detail="patient_id is required for staff accounts")

    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@app.get("/patients")
def list_patients(current_user: User = Depends(require_roles("doctor", "admin")), db: Session = Depends(get_db)):
    patients = db.query(Patient).all()
    return [{"id": p.id, "mrn": p.mrn, "full_name": p.full_name, "primary_diagnosis": p.primary_diagnosis} for p in patients]


@app.get("/patients/{patient_id}")
def get_patient(patient_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    patient = _resolve_patient(db, current_user, patient_id)
    return {
        "id": patient.id, "mrn": patient.mrn, "full_name": patient.full_name,
        "dob": patient.dob, "sex": patient.sex,
        "primary_diagnosis": patient.primary_diagnosis,
        "medications": patient.medications, "allergies": patient.allergies,
    }


@app.get("/patients/{patient_id}/notes")
def get_notes(patient_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    patient = _resolve_patient(db, current_user, patient_id)
    notes = db.query(ClinicalNote).filter(ClinicalNote.patient_id == patient.id).all()
    return [{"id": n.id, "author": n.author, "type": n.note_type, "content": n.content,
             "created_at": n.created_at.isoformat()} for n in notes]


@app.get("/patients/{patient_id}/memory")
def get_memory(patient_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    patient = _resolve_patient(db, current_user, patient_id)
    return get_patient_memory(db, patient.id)


# ---------- agentic chat ----------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    patient = _resolve_patient(db, current_user, req.patient_id)
    result = run_agent_turn(db, patient, req.session_id, req.message)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- file-based RAG demo ----------

@app.get("/rag/files")
def rag_list_files(current_user: User = Depends(get_current_user)):
    return {"files": knowledge_rag.list_knowledge_files()}


@app.post("/rag/reindex")
def rag_reindex(current_user: User = Depends(get_current_user)):
    chunk_count = knowledge_rag.build_index()
    return {"status": "reindexed", "files": knowledge_rag.list_knowledge_files(), "chunks": chunk_count}


@app.post("/rag/upload")
async def rag_upload(file: UploadFile, current_user: User = Depends(get_current_user)):
    if not file.filename.lower().endswith((".txt", ".md", ".pdf")):
        raise HTTPException(status_code=400, detail="Only .txt, .md, or .pdf files are supported")
    dest = os.path.join(knowledge_rag.KB_DIR, os.path.basename(file.filename))
    os.makedirs(knowledge_rag.KB_DIR, exist_ok=True)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    chunk_count = knowledge_rag.build_index()
    return {"status": "uploaded_and_reindexed", "filename": file.filename, "chunks": chunk_count}


@app.post("/rag/query")
def rag_query(req: RagQueryRequest, current_user: User = Depends(get_current_user)):
    return knowledge_rag.query_knowledge_base(req.question, k=req.k)
