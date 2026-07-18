"""
loader.py — reads raw .txt files from docs/ and returns structured documents.

Output shape (one dict per file):
{
    "text": "full raw text...",
    "metadata": {
        "source":     "docs/what_is_rag.txt",
        "filename":   "what_is_rag.txt",
        "doc_id":     "what_is_rag",        # used later to filter in ChromaDB
        "char_count": 1842,
    }
}

Why metadata matters:
  At query time you can tell ChromaDB "only search chunks from doc_id='what_is_rag'"
  instead of searching the entire collection. This is metadata filtering — every
  retrieved chunk carries its metadata forward from this step.
"""

import os


DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_documents(docs_dir: str = DOCS_DIR) -> list[dict]:
    """
    Reads all .txt files in docs_dir.
    Returns a list of document dicts, one per file.
    """
    documents = []
    docs_dir = os.path.abspath(docs_dir)

    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"docs/ directory not found at: {docs_dir}")

    txt_files = [f for f in os.listdir(docs_dir) if f.endswith(".txt")]

    if not txt_files:
        raise ValueError(f"No .txt files found in {docs_dir}")

    for filename in sorted(txt_files):     # sorted = deterministic order
        filepath = os.path.join(docs_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print(f"[LOADER] Skipping empty file: {filename}")
            continue

        doc = {
            "text": text,
            "metadata": {
                "source":     os.path.join("docs", filename),
                "filename":   filename,
                "doc_id":     filename.replace(".txt", ""),   # "what_is_rag"
                "char_count": len(text),
            }
        }
        documents.append(doc)
        print(f"[LOADER] Loaded: {filename} ({len(text)} chars)")

    print(f"\n[LOADER] Total documents loaded: {len(documents)}")
    return documents


# ── Quick test — run this file directly to verify ──────────────
if __name__ == "__main__":
    docs = load_documents()

    print("\n── Document preview ──────────────────────────────")
    for doc in docs:
        print(f"\nFile    : {doc['metadata']['filename']}")
        print(f"Doc ID  : {doc['metadata']['doc_id']}")
        print(f"Chars   : {doc['metadata']['char_count']}")
        print(f"Preview : {doc['text'][:120].replace(chr(10), ' ')}...")
