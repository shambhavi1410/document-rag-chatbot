"""Chunking, embeddings, FAISS retrieval and Gemini answer generation."""
import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_index(docs, embeddings=None):
    """Split documents into chunks, embed them, and store them in FAISS."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings or get_embeddings()), len(chunks)


def retrieve(index, question: str, k: int = 4):
    return index.similarity_search(question, k=k)


def label(doc) -> str:
    page = doc.metadata.get("page")
    return f"{doc.metadata['source']}" + (f", p.{page}" if page else "")


def answer(question: str, docs) -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return "Set GEMINI_API_KEY in your .env file to get answers."
    context = "\n\n".join(f"[{label(d)}]\n{d.page_content}" for d in docs)
    prompt = ("Answer the question using only the context below. Cite sources like [file, p.N]. "
              "If the answer is not in the context, say you could not find it in the documents.\n\n"
              f"Context:\n{context}\n\nQuestion: {question}")
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key}, timeout=60, json={"contents": [{"parts": [{"text": prompt}]}]})
    if r.status_code != 200:
        return f"Gemini error ({r.status_code}): {r.text[:200]}"
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]
