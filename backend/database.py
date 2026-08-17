"""
Database layer: SQLAlchemy models for a synthetic healthcare system.

IMPORTANT: All patient data created by seed_data.py is 100% synthetic / fake.
Never point this system at real patient records (PHI).
"""
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///./healthcare_demo.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """Login account. role in {"patient", "doctor", "admin"}."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False, default="patient")
    # Only set when role == "patient": which patient record this account owns
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    is_active = Column(Boolean, default=True)

    patient = relationship("Patient", back_populates="user")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    mrn = Column(String, unique=True, index=True)  # medical record number
    full_name = Column(String, nullable=False)
    dob = Column(String, nullable=False)
    sex = Column(String, nullable=True)
    primary_diagnosis = Column(String, nullable=True)
    medications = Column(Text, nullable=True)  # comma separated, synthetic
    allergies = Column(Text, nullable=True)

    user = relationship("User", back_populates="patient", uselist=False)
    notes = relationship("ClinicalNote", back_populates="patient")
    memories = relationship("MemoryFact", back_populates="patient")


class ClinicalNote(Base):
    """Free-text clinical notes -- this is the corpus the RAG agent retrieves over."""
    __tablename__ = "clinical_notes"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    author = Column(String, nullable=False)  # e.g. "Dr. Amina Raza"
    note_type = Column(String, default="progress_note")
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="notes")


class MemoryFact(Base):
    """Long-term agent memory: durable facts the agent has chosen to remember
    about a patient across sessions (e.g. preferences, recurring context)."""
    __tablename__ = "memory_facts"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    key = Column(String, nullable=False)
    value = Column(Text, nullable=False)
    source = Column(String, default="agent")  # "agent" | "user" | "system"
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="memories")


class ConversationLog(Base):
    """Short-term / session memory: raw chat turns, scoped to a session_id."""
    __tablename__ = "conversation_log"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    role = Column(String, nullable=False)  # "user" | "assistant" | "tool"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
