"""
Classic file-based RAG pipeline.

This is deliberately separate from rag.py (which does TF-IDF retrieval over
patient clinical notes stored in the database). This module is the "drop
files in a folder, ask a question, get an answer sourced from those files"
pattern:

    files on disk
        -> load (TextLoader / PyPDFLoader)
        -> split into chunks (RecursiveCharacterTextSplitter)
        -> embed (Gemini embeddings, free tier)
        -> store (Chroma vector store, persisted to disk)
        -> retrieve top-k chunks for a query (similarity search)
        -> pass chunks + question to the LLM, answer grounded in them

Source files live in backend/knowledge_base/*.txt or *.pdf. Drop your own
files in there (or use POST /rag/upload) and call POST /rag/query to ask
questions answered from them, with the source chunks returned alongside the
answer so you can see exactly what the model was grounded on.

This also gives you a second, independent RAG surface to test: e.g. does a
file with adversarial content injected into it change answers for
*unrelated* questions/files it shouldn't touch (context bleed across an
otherwise irrelevant retrieval)?
"""
import os
import logging
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

logger = logging.getLogger("knowledge_rag")
logging.basicConfig(level=logging.INFO)

KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")

API_KEY = os.environ.get("GEMINI_API_KEY")
CHAT_MODEL = os.environ.get("HEALTHCARE_DEMO_MODEL", "gemini-2.5-flash")
# Free-tier embedding model. text-embedding-004 was fully shut down by
# Google in early 2026 -- gemini-embedding-001 is the current replacement.
EMBED_MODEL = os.environ.get("HEALTHCARE_DEMO_EMBED_MODEL", "models/gemini-embedding-001")

_embeddings = None
_vectorstore = None
_llm = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        if not API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set (see backend/.env.example).")
        _embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=API_KEY)
    return _embeddings


def _get_llm():
    global _llm
    if _llm is None:
        if not API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set (see backend/.env.example).")
        _llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=API_KEY, temperature=0.1)
    return _llm


def _load_file(path: str) -> List[Document]:
    if path.lower().endswith(".pdf"):
        return PyPDFLoader(path).load()
    return TextLoader(path, encoding="utf-8").load()


def list_knowledge_files() -> List[str]:
    os.makedirs(KB_DIR, exist_ok=True)
    return sorted(f for f in os.listdir(KB_DIR) if f.lower().endswith((".txt", ".md", ".pdf")))


def build_index() -> int:
    """(Re)builds the vector store from every file currently in
    knowledge_base/. Call this after adding/removing files. Returns the
    number of chunks indexed."""
    os.makedirs(KB_DIR, exist_ok=True)
    docs: List[Document] = []
    for fname in list_knowledge_files():
        docs.extend(_load_file(os.path.join(KB_DIR, fname)))

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    logger.info("build_index: %d files -> %d chunks, embedding now (this hits the Gemini API)...", len(list_knowledge_files()), len(chunks))

    global _vectorstore
    # Fresh collection each rebuild -- simplest correct behavior for a demo;
    # avoids stale chunks from deleted/edited source files lingering.
    _vectorstore = Chroma.from_documents(
        chunks,
        embedding=_get_embeddings(),
        persist_directory=PERSIST_DIR,
        collection_name="knowledge_base",
    )
    logger.info("build_index: done, %d chunks indexed", len(chunks))
    return len(chunks)


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            logger.info("Loading existing Chroma index from %s", PERSIST_DIR)
            _vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=_get_embeddings(),
                collection_name="knowledge_base",
            )
        else:
            logger.info("No existing index found, building one now...")
            build_index()
    return _vectorstore


def query_knowledge_base(question: str, k: int = 4) -> Dict:
    """The actual RAG query: retrieve top-k chunks, then answer grounded in
    them. Returns the answer plus the source chunks so you can verify
    grounding (or spot when it isn't grounded)."""
    logger.info("query_knowledge_base: %r", question)
    vs = _get_vectorstore()
    logger.info("Running similarity search...")
    results = vs.similarity_search_with_relevance_scores(question, k=k)
    logger.info("Similarity search returned %d results", len(results))

    if not results:
        return {"answer": "No indexed documents to search. Add files to backend/knowledge_base/ and call /rag/reindex.",
                "sources": []}

    context = "\n\n".join(
        f"[Source: {os.path.basename(doc.metadata.get('source', 'unknown'))}]\n{doc.page_content}"
        for doc, _score in results
    )

    prompt = f"""Answer the question using ONLY the context below. If the
context does not contain the answer, say so plainly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""

    logger.info("Calling Gemini chat model for final answer...")
    response = _get_llm().invoke(prompt)
    logger.info("Got response from Gemini")
    sources = [
        {
            "source": os.path.basename(doc.metadata.get("source", "unknown")),
            "excerpt": doc.page_content[:300],
            "relevance": round(score, 3),
        }
        for doc, score in results
    ]
    return {"answer": response.content, "sources": sources}
