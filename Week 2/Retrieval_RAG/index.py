"""
index.py — builds the knowledge base. Run this ONCE before querying.

Usage:
    python index.py              # standard pipeline
    python index.py --enriched   # with contextual enrichment (more LLM calls)
    python index.py --reset      # wipe ChromaDB and reindex from scratch

What it does:
    loader → chunker → [enricher] → embedder → vectorstore + bm25_store

Two collections are created in ChromaDB:
    "rag_chunks"          — standard (used by query.py by default)
    "rag_chunks_enriched" — contextually enriched (used when --enriched passed to query.py)

BM25 is rebuilt in memory at query time (not persisted) — fast enough for
small corpora. For production, serialise with pickle.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.loader      import load_documents
from pipeline.chunker     import chunk_documents
from pipeline.embedder    import Embedder
from pipeline.vectorstore import VectorStore
from pipeline.enricher    import enrich_documents


def build_index(enriched: bool = False, reset: bool = False):
    mode = "ENRICHED" if enriched else "STANDARD"
    collection_name = "rag_chunks_enriched" if enriched else "rag_chunks"

    print(f"\n{'='*60}")
    print(f"INDEXING — mode: {mode}")
    print(f"Collection: {collection_name}")
    print(f"{'='*60}\n")

    # ── Step 1: Load ──────────────────────────────────────────
    documents = load_documents()

    # ── Step 2: Chunk ─────────────────────────────────────────
    chunks = chunk_documents(documents, chunk_size=512, overlap=50)

    # ── Step 3: Enrich (optional) ─────────────────────────────
    if enriched:
        print("\n[INDEX] Running contextual enrichment...")
        print("[INDEX] Warning: one LLM call per chunk — may be slow.\n")
        chunks = enrich_documents(documents, chunks)

    # ── Step 4: Embed ─────────────────────────────────────────
    embedder = Embedder()
    chunks   = embedder.embed_documents(chunks)

    # ── Step 5: Store in ChromaDB ─────────────────────────────
    store = VectorStore(collection_name=collection_name)

    if reset:
        print("\n[INDEX] Resetting collection...")
        store.reset()

    store.add_chunks(chunks)

    print(f"\n[INDEX] ✅ Done. {store.count()} chunks in '{collection_name}'.")
    return chunks   # returned so query.py can reuse for BM25 without re-loading


if __name__ == "__main__":
    enriched = "--enriched" in sys.argv
    reset    = "--reset"    in sys.argv
    build_index(enriched=enriched, reset=reset)
