"""
vectorstore.py — ChromaDB wrapper for storing and querying chunk embeddings.

What ChromaDB stores per chunk (4 parallel lists, same index = same chunk):
  ids        → ["what_is_rag__0", "what_is_rag__1", ...]   unique string ID
  embeddings → [[0.02, -0.14, ...], ...]                   384-dim float vectors
  documents  → ["chunk text...", ...]                       raw text (for LLM context)
  metadatas  → [{"doc_id": "what_is_rag", ...}, ...]       filter/display info

Why store documents inside ChromaDB and not just embeddings?
  At query time ChromaDB returns the matching chunk's text alongside the vector.
  You don't need a separate lookup — the text comes back with the result.
  This is the standard pattern for small-medium RAG (< 1M chunks).

ChromaDB metadata constraint worth knowing:
  Metadata values must be str, int, float, or bool.
  No lists, no nested dicts. If you try to store a list in metadata, it will crash.
  Flatten everything before inserting.
"""

import os
import chromadb
from chromadb.config import Settings


# Local persistent storage — ChromaDB writes to this folder
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "rag_chunks"


class VectorStore:
    def __init__(self, collection_name: str = COLLECTION_NAME, chroma_dir: str = CHROMA_DIR):
        """
        Creates (or loads) a persistent ChromaDB collection.

        PersistentClient means data survives between Python sessions —
        you index once, query many times without re-embedding.
        """
        chroma_dir = os.path.abspath(chroma_dir)
        os.makedirs(chroma_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection_name = collection_name

        # get_or_create: safe to call every time — won't duplicate if exists
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # cosine distance = 1 - cosine_similarity
            # since we L2-normalised in embedder, dot product == cosine sim
            # but ChromaDB's cosine metric handles this correctly regardless
            metadata={"hnsw:space": "cosine"}
        )

        count = self.collection.count()
        print(f"[VECTORSTORE] Collection '{collection_name}' loaded. Chunks stored: {count}")

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Inserts chunk dicts (from embedder.embed_documents) into ChromaDB.

        ChromaDB expects 4 parallel lists — same index = same chunk.
        We build those lists by iterating chunks once.

        Skips chunks that are already in the collection (by chunk_id).
        This means re-running index.py won't create duplicates.
        """
        if not chunks:
            print("[VECTORSTORE] No chunks to add.")
            return

        # Find which chunk_ids already exist to avoid duplicates
        existing_ids = set()
        existing = self.collection.get(include=[])   # fetch only ids, no content
        if existing["ids"]:
            existing_ids = set(existing["ids"])

        # Build the 4 parallel lists ChromaDB expects
        ids, embeddings, documents, metadatas = [], [], [], []

        for chunk in chunks:
            chunk_id = chunk["metadata"]["chunk_id"]

            if chunk_id in existing_ids:
                continue    # already indexed — skip

            if "embedding" not in chunk:
                print(f"[VECTORSTORE] Warning: chunk {chunk_id} has no embedding. Run embedder first.")
                continue

            # ChromaDB metadata: flatten everything, values must be primitives
            # chunk["metadata"] already has only str/int values from chunker.py
            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])

        if not ids:
            print("[VECTORSTORE] All chunks already indexed. Nothing to add.")
            return

        # ChromaDB batches inserts automatically — passing all at once is fine
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        print(f"[VECTORSTORE] Inserted {len(ids)} chunks. Total now: {self.collection.count()}")

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_doc_id: str = None,    # e.g. "what_is_rag" — search only that doc
    ) -> list[dict]:
        """
        Semantic search: finds top_k chunks closest to query_embedding.

        Returns a list of result dicts, sorted by relevance (best first):
        [
            {
                "chunk_id":  "what_is_rag__2",
                "text":      "chunk text...",
                "score":     0.87,           # cosine similarity (higher = better)
                "metadata":  {...},
                "rank":      1,              # 1 = best match
            },
            ...
        ]

        Why return this shape?
          The retriever.py will merge results from vectorstore + BM25.
          Having a consistent result shape makes merging straightforward.
        """
        where = {"doc_id": filter_doc_id} if filter_doc_id else None

        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),  # can't request more than exist
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        # ChromaDB returns lists-of-lists because you can query multiple embeddings at once
        # [0] because we sent one query embedding
        ids        = raw["ids"][0]
        documents  = raw["documents"][0]
        metadatas  = raw["metadatas"][0]
        distances  = raw["distances"][0]   # cosine distance: 0 = identical, 2 = opposite

        results = []
        for rank, (chunk_id, text, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            results.append({
                "chunk_id": chunk_id,
                "text":     text,
                "score":    round(1 - dist, 4),   # convert distance → similarity
                "metadata": meta,
                "rank":     rank,
            })

        return results

    def reset(self) -> None:
        """
        Deletes and recreates the collection — useful during development
        when you change chunking/embedding strategy and need a clean slate.
        """
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[VECTORSTORE] Collection '{self.collection_name}' reset. Now empty.")

    def count(self) -> int:
        return self.collection.count()


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from pipeline.loader import load_documents
    from pipeline.chunker import chunk_documents
    from pipeline.embedder import Embedder

    # Full pipeline: load → chunk → embed → store → query
    docs   = load_documents()
    chunks = chunk_documents(docs, chunk_size=512, overlap=50)

    embedder = Embedder()
    chunks   = embedder.embed_documents(chunks)

    store = VectorStore()
    store.add_chunks(chunks)

    # Test query
    query     = "Why does chunk size matter for retrieval quality?"
    query_vec = embedder.embed_query(query)
    results   = store.query(query_vec, top_k=3)

    print(f"\n── Semantic search results ───────────────────────")
    print(f"Query: '{query}'\n")
    for r in results:
        print(f"Rank {r['rank']}  score={r['score']}  id={r['chunk_id']}")
        print(f"  {r['text'][:120].replace(chr(10), ' ')}...")
        print()
