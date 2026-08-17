"""
Memory tools available to the agent orchestrator.

Two tiers, mirroring how most production agentic systems do it:
  - Short-term / session memory: the raw back-and-forth for the current
    conversation (ConversationLog), scoped by session_id.
  - Long-term / durable memory: distilled facts the agent has decided are
    worth remembering about a patient across sessions (MemoryFact).

Attack-surface note: long-term memory writes are a classic target for
"memory poisoning" -- if untrusted text (e.g. a patient's free-text message,
or a note ingested from an external system) can cause the agent to write a
false fact into MemoryFact, that false fact then quietly contaminates every
future session for that patient. Worth testing whether remember_fact() can
be triggered by prompt-injected content rather than genuine clinical intent.
"""
from typing import List, Dict
from sqlalchemy.orm import Session

from database import ConversationLog, MemoryFact


def log_turn(db: Session, session_id: str, patient_id: int, role: str, content: str):
    entry = ConversationLog(session_id=session_id, patient_id=patient_id, role=role, content=content)
    db.add(entry)
    db.commit()


def get_recent_turns(db: Session, session_id: str, limit: int = 20) -> List[Dict]:
    rows = (
        db.query(ConversationLog)
        .filter(ConversationLog.session_id == session_id)
        .order_by(ConversationLog.created_at.desc())
        .limit(limit)
        .all()
    )
    rows.reverse()
    return [{"role": r.role, "content": r.content} for r in rows]


def remember_fact(db: Session, patient_id: int, key: str, value: str, source: str = "agent") -> MemoryFact:
    fact = MemoryFact(patient_id=patient_id, key=key, value=value, source=source)
    db.add(fact)
    db.commit()
    db.refresh(fact)
    return fact


def get_patient_memory(db: Session, patient_id: int) -> List[Dict]:
    rows = (
        db.query(MemoryFact)
        .filter(MemoryFact.patient_id == patient_id)
        .order_by(MemoryFact.created_at.asc())
        .all()
    )
    return [{"key": r.key, "value": r.value, "source": r.source} for r in rows]
