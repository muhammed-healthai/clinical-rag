"""
ingest.py - Bulk-build the index from every PDF in the guidelines/ folder.

Use this for the initial load, or to rebuild the index so it exactly matches
the guidelines/ folder:

    python ingest.py

Because admin uploads now also save the original PDF into guidelines/ and record
it in the manifest, a rebuild here stays consistent with what was added through
the app - it won't silently drop anything.
"""

import os
import glob
from datetime import datetime

import chromadb

from rag import (index_pdf, get_embedder, DB_DIR, COLLECTION_NAME,
                 GUIDELINES_DIR, load_manifest, save_manifest)


def main():
    pdf_paths = glob.glob(os.path.join(GUIDELINES_DIR, "*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in '{GUIDELINES_DIR}/'. Add some guideline PDFs and try again.")
        return

    print(f"Found {len(pdf_paths)} PDF(s). Loading the embedding model "
          "(first run downloads ~90 MB)...")
    get_embedder()

    # Fresh rebuild so the index matches the folder exactly.
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    manifest = load_manifest()
    present, total = set(), 0
    for path in pdf_paths:
        filename = os.path.basename(path)
        present.add(filename)
        with open(path, "rb") as f:
            n = index_pdf(f, filename)
        print(f"  {filename}: {n} chunks")
        total += n
        if filename in manifest:
            manifest[filename]["chunks"] = n
        else:
            manifest[filename] = {
                "added_by": "initial import",
                "added_at": datetime.now().isoformat(timespec="seconds"),
                "chunks": n,
            }

    # Forget manifest entries for files no longer in the folder.
    for fn in list(manifest):
        if fn not in present:
            manifest.pop(fn)
    save_manifest(manifest)

    print(f"Done. Indexed {total} chunks into '{DB_DIR}/'.")


if __name__ == "__main__":
    main()