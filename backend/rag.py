"""
Lightweight RAG retriever over ClinicalNote text.

Uses TF-IDF instead of a hosted embeddings model so the demo runs fully
offline / air-gapped once dependencies are installed -- useful if you want
to test this system without any external network calls at all (only the
agent's reasoning calls hit the Anthropic API; retrieval itself is local).

Security-relevant design point: retrieval is (and must stay) scoped by
patient_id at the query level, not filtered after the fact. If you ever
refactor this to "retrieve globally, then filter," you'd introduce a classic
cross-patient data leakage bug -- a good thing to specifically test for.
"""
from dataclasses import dataclass
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session

from database import ClinicalNote


@dataclass
class RetrievedChunk:
    note_id: int
    author: str
    note_type: str
    content: str
    score: float


def retrieve_for_patient(db: Session, patient_id: int, query: str, top_k: int = 3) -> List[RetrievedChunk]:
    notes = db.query(ClinicalNote).filter(ClinicalNote.patient_id == patient_id).all()
    if not notes:
        return []

    corpus = [n.content for n in notes]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(corpus + [query])
    except ValueError:
        # e.g. corpus is all stopwords / empty after vectorization
        return []

    doc_vectors = matrix[:-1]
    query_vector = matrix[-1]
    scores = cosine_similarity(query_vector, doc_vectors).flatten()

    ranked = sorted(zip(notes, scores), key=lambda pair: pair[1], reverse=True)
    results = []
    for note, score in ranked[:top_k]:
        if score <= 0:
            continue
        results.append(
            RetrievedChunk(
                note_id=note.id,
                author=note.author,
                note_type=note.note_type,
                content=note.content,
                score=float(score),
            )
        )
    return results
