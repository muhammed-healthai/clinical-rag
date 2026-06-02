"""
rag.py - The core logic: find relevant passages, then answer using only those.

This file is imported by app.py. You normally don't run it directly, but you
can test it from a Python prompt:

    from rag import answer
    text, sources = answer("your question here")
    print(text)
"""

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

load_dotenv()  # reads ANTHROPIC_API_KEY from the .env file

DB_DIR = "chroma_db"
COLLECTION_NAME = "guidelines"
TOP_K = 8  # how many passages to retrieve per question

# If you get a "model not found" error, check current names at docs.claude.com.
# For lower cost, swap to "claude-haiku-4-5-20251001".
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a clinical guideline assistant. Answer ONLY using the \
numbered context passages supplied by the user. Follow these rules strictly:
- Base every statement on the passages. Do NOT use outside knowledge.
- After each claim, cite the passage number(s) you used in square brackets, e.g. [1] or [2][4].
- If the answer is not contained in the passages, reply exactly: \
"I can't find this in the supplied guidelines." Do not guess or fill gaps.
- Be concise and clinical. This is a decision-support aid, not a substitute for \
clinical judgement."""

# Load the embedding model once and reuse it.
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_collection(COLLECTION_NAME)


def retrieve(question, k=TOP_K):
    """Return the k most relevant (chunk, metadata) pairs for the question."""
    q_emb = get_embedder().encode([question]).tolist()
    res = get_collection().query(query_embeddings=q_emb, n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def build_context(passages):
    """Format retrieved passages as a numbered, citable list."""
    blocks = []
    for i, (doc, meta) in enumerate(passages, start=1):
        blocks.append(f"[{i}] (Source: {meta['source']}, page {meta['page']})\n{doc}")
    return "\n\n".join(blocks)


def answer(question, k=TOP_K):
    """Retrieve passages, then generate a grounded, cited answer."""
    passages = retrieve(question, k)
    user_prompt = f"Context passages:\n\n{build_context(passages)}\n\nQuestion: {question}"

    client = Anthropic()  # picks up ANTHROPIC_API_KEY from the environment
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text, passages


def list_sources():
    """Return the distinct source filenames currently in the index."""
    data = get_collection().get()
    metas = data.get("metadatas") or []
    return sorted({m["source"] for m in metas if m and "source" in m})
