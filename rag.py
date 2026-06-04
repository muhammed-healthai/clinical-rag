"""
rag.py - Retrieval, grounded generation, and index management.

Holds everything that touches the vector store and the guideline library:
- adding/removing guidelines (used by both ingest.py and the app's admin panel)
- persisting the original PDF and recording who added it and when
- retrieving relevant passages and generating a grounded, cited answer

The on-disk index (chroma_db/) is treated as a *derived* artefact: the original
PDFs in guidelines/ plus the manifest (guidelines_meta.json) are the record of
what should be in it, so the index can always be rebuilt from them.
"""

import io
import os
import re
import json
from datetime import datetime

import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

load_dotenv()  # reads ANTHROPIC_API_KEY (and ADMIN_PASSWORD) from the .env file

DB_DIR = "chroma_db"
COLLECTION_NAME = "guidelines"
GUIDELINES_DIR = "guidelines"
MANIFEST_PATH = "guidelines_meta.json"
TOP_K = 8

# If you get a "model not found" error, check current names at docs.claude.com.
# For lower cost, swap to "claude-haiku-4-5-20251001".
MODEL = "claude-sonnet-4-6"

CHUNK_SIZE = 1200
OVERLAP_SENTENCES = 2
MIN_CHUNK_CHARS = 150

SYSTEM_PROMPT = """You are a clinical guideline assistant. Answer ONLY using the \
numbered context passages supplied by the user. Follow these rules strictly:
- Base every statement on the passages. Do NOT use outside knowledge.
- After each claim, cite the passage number(s) you used in square brackets, e.g. [1] or [2][4].
- If the answer is not contained in the passages, reply exactly: \
"I can't find this in the supplied guidelines." Do not guess or fill gaps.
- Be concise and clinical. This is a decision-support aid, not a substitute for \
clinical judgement."""


# ---------- shared resources ----------

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_collection():
    """Open the index, creating it if it doesn't exist yet."""
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)


# ---------- manifest (who added what, when) ----------

def load_manifest():
    try:
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def get_source_meta():
    """Return {filename: {added_by, added_at, chunks}} for display."""
    return load_manifest()


# ---------- chunking ----------

def split_sentences(text):
    text = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP_SENTENCES):
    sentences = split_sentences(text)
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        if current and current_len + len(sentence) > size:
            chunks.append(" ".join(current))
            current = current[-overlap:]
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


# ---------- low-level indexing ----------

def index_pdf(file_obj, filename):
    """Add a PDF to the index, replacing any existing copy of the same filename.
    Returns the number of chunks indexed."""
    collection = get_collection()
    try:
        collection.delete(where={"source": filename})
    except Exception:
        pass

    embedder = get_embedder()
    reader = PdfReader(file_obj)
    documents, metadatas, ids = [], [], []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for i, chunk in enumerate(chunk_text(text)):
            documents.append(chunk)
            metadatas.append({"source": filename, "page": page_num})
            ids.append(f"{filename}-p{page_num}-c{i}")

    if not documents:
        return 0

    embeddings = embedder.encode(documents).tolist()
    collection.add(documents=documents, embeddings=embeddings,
                   metadatas=metadatas, ids=ids)
    return len(documents)


def index_pdf_bytes(data, filename):
    return index_pdf(io.BytesIO(data), filename)


def remove_source(filename):
    get_collection().delete(where={"source": filename})


def list_sources():
    data = get_collection().get()
    metas = data.get("metadatas") or []
    return sorted({m["source"] for m in metas if m and "source" in m})


# ---------- high-level guideline library (used by the admin panel) ----------

def _safe_filename(filename):
    """Strip any path components so an upload can't be written outside the folder."""
    return os.path.basename(filename)


def add_guideline(data, filename, added_by="admin"):
    """Save the original PDF to guidelines/, index it, and record who added it.
    Returns the number of chunks indexed."""
    os.makedirs(GUIDELINES_DIR, exist_ok=True)
    filename = _safe_filename(filename)
    with open(os.path.join(GUIDELINES_DIR, filename), "wb") as f:
        f.write(data)

    n = index_pdf_bytes(data, filename)

    manifest = load_manifest()
    manifest[filename] = {
        "added_by": added_by or "admin",
        "added_at": datetime.now().isoformat(timespec="seconds"),
        "chunks": n,
    }
    save_manifest(manifest)
    return n


def delete_guideline(filename):
    """Remove a guideline from the index, delete its PDF, and forget its record."""
    filename = _safe_filename(filename)
    remove_source(filename)
    try:
        os.remove(os.path.join(GUIDELINES_DIR, filename))
    except FileNotFoundError:
        pass
    manifest = load_manifest()
    manifest.pop(filename, None)
    save_manifest(manifest)


# ---------- retrieval + generation ----------

def retrieve(question, k=TOP_K):
    q_emb = get_embedder().encode([question]).tolist()
    res = get_collection().query(query_embeddings=q_emb, n_results=k)
    docs = res.get("documents") or [[]]
    metas = res.get("metadatas") or [[]]
    return list(zip(docs[0], metas[0]))


def build_context(passages):
    blocks = []
    for i, (doc, meta) in enumerate(passages, start=1):
        blocks.append(f"[{i}] (Source: {meta['source']}, page {meta['page']})\n{doc}")
    return "\n\n".join(blocks)


def answer(question, k=TOP_K):
    passages = retrieve(question, k)
    if not passages:
        return ("No guidelines are indexed yet, so there is nothing to search. "
                "Add a guideline to the knowledge base first.", [])

    user_prompt = f"Context passages:\n\n{build_context(passages)}\n\nQuestion: {question}"
    client = Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text, passages