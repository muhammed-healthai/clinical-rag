"""
ingest.py - Build the searchable index from your guideline PDFs.

Run this once at the start, and again any time you add or remove PDFs:

    python ingest.py

It reads every PDF in the guidelines/ folder, splits the text into chunks
(remembering the source file and page for citations), turns each chunk into a
vector with a small local model, and stores everything in a local database
folder called chroma_db/.

Chunking notes:
- Text is split on sentence boundaries, then sentences are packed together up
  to ~CHUNK_SIZE characters, so chunks don't cut off mid-sentence.
- Each chunk overlaps with the previous one by a couple of sentences, so an
  idea that spans a boundary isn't lost.
- Very short fragments (page numbers, headers, contents lines) are dropped,
  because they add noise to retrieval without carrying real content.
"""

import os
import re
import glob

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

GUIDELINES_DIR = "guidelines"
DB_DIR = "chroma_db"
COLLECTION_NAME = "guidelines"
CHUNK_SIZE = 1200        # target characters per chunk
OVERLAP_SENTENCES = 2    # sentences shared between neighbouring chunks
MIN_CHUNK_CHARS = 150    # drop fragments shorter than this (headers, page nums)


def split_sentences(text):
    """Tidy whitespace and split text into sentences."""
    text = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP_SENTENCES):
    """Pack whole sentences into ~`size`-character chunks with overlap."""
    sentences = split_sentences(text)
    chunks = []
    current, current_len = [], 0

    for sentence in sentences:
        if current and current_len + len(sentence) > size:
            chunks.append(" ".join(current))
            current = current[-overlap:]              # carry overlap forward
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def main():
    pdf_paths = glob.glob(os.path.join(GUIDELINES_DIR, "*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in '{GUIDELINES_DIR}/'. Add some guideline PDFs and try again.")
        return

    print(f"Found {len(pdf_paths)} PDF(s). Loading the embedding model "
          "(first run downloads ~90 MB)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=DB_DIR)
    # Rebuild from scratch so re-running always reflects the current folder.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    documents, metadatas, ids = [], [], []
    for path in pdf_paths:
        filename = os.path.basename(path)
        reader = PdfReader(path)
        print(f"  Reading {filename} ({len(reader.pages)} pages)...")
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for i, chunk in enumerate(chunk_text(text)):
                documents.append(chunk)
                metadatas.append({"source": filename, "page": page_num})
                ids.append(f"{filename}-p{page_num}-c{i}")

    if not documents:
        print("No usable text found. (Scanned/image-only PDFs need OCR first.)")
        return

    print(f"Embedding {len(documents)} chunks...")
    embeddings = embedder.encode(documents, show_progress_bar=True).tolist()

    collection.add(documents=documents, embeddings=embeddings,
                   metadatas=metadatas, ids=ids)
    print(f"Done. Indexed {len(documents)} chunks into '{DB_DIR}/'.")


if __name__ == "__main__":
    main()
