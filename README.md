# Clinical Guideline Assistant

A retrieval-augmented (RAG) question-answering tool that answers clinical
questions **using only the guideline documents you give it**, and **cites the
exact source passage** behind every claim. If the answer isn't in the supplied
guidelines, it says so rather than guessing.

Built by a doctor, with trust and explainability as the design goal: the
clinician can always check where an answer came from, and the model is
constrained from inventing content beyond the source material.

> Educational and exploratory prototype only. Not a clinical decision-making tool.

## How it works

```
  PDFs in guidelines/
          |
   [ ingest.py ]   split into chunks -> embed each chunk -> store in a vector database
          |
   chroma_db/  (local, on disk)
          |
   [ rag.py ]      embed the question -> retrieve the closest chunks
          |         -> ask the LLM to answer using ONLY those chunks, with citations
          |
   [ app.py ]      Streamlit interface: question in, cited answer + source passages out
```

The model is given numbered passages and a strict instruction: cite the
passages used, and refuse if the answer isn't present. That refusal behaviour
is the core safety feature.

## Tech stack

- **Python**
- **sentence-transformers** (`all-MiniLM-L6-v2`) for local embeddings - no API needed, runs on CPU
- **ChromaDB** as the local vector store
- **pypdf** for reading the source documents
- **Anthropic API** for the grounded answer generation
- **Streamlit** for the interface

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env
# then edit .env and paste your real Anthropic key

# 4. Add guideline PDFs to the guidelines/ folder

# 5. Build the index
python ingest.py

# 6. Run the app
streamlit run app.py
```

## A note on source documents

Add PDFs you are entitled to use (for example, freely published guideline PDFs)
to the `guidelines/` folder. The app processes them locally; the PDFs themselves
are git-ignored and never committed, so no copyrighted content is redistributed.
Some sources (e.g. licensed formularies) have usage restrictions - respect them.

## Limitations

- Quality of answers depends entirely on the guidelines you index.
- Scanned/image-only PDFs need OCR first (not included).
- Retrieval is semantic similarity, not clinical reasoning - always verify against the cited source.
