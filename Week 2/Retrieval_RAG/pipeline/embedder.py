"""
embedder.py — wraps SentenceTransformer for embedding chunks and queries.

Why SentenceTransformer over OpenAI embeddings here?
  - Free, runs locally on CPU, no API key needed for indexing
  - all-MiniLM-L6-v2 is small (80MB) but produces strong embeddings
    for semantic similarity tasks — benchmark standard for RAG demos
  - Same interface as OpenAI embeddings so you can swap later

Two separate methods matter:
  embed_documents() — used at INDEX time (many chunks, batch is faster)
  embed_query()     — used at QUERY time (single string, must match
                       the same vector space as documents)

They use the same model so vectors are in the same space —
this is the requirement for cosine similarity to be meaningful.
"""

from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "all-MiniLM-L6-v2"   # 80MB, 384-dim vectors, good speed/quality tradeoff


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        print(f"[EMBEDDER] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[EMBEDDER] Ready. Vector dimension: {self.dimension}")

    def embed_documents(self, chunks: list[dict]) -> list[dict]:
        """
        Embeds a list of chunk dicts (from chunker.py).
        Adds an "embedding" key to each chunk dict in-place.

        Uses batch encoding — much faster than embedding one at a time.

        Returns the same list with embeddings added.
        """
        texts = [chunk["text"] for chunk in chunks]

        print(f"[EMBEDDER] Embedding {len(texts)} chunks...")
        # batch_size=32 is safe for CPU — increase if you have more RAM
        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,      # ChromaDB expects plain lists/numpy
            normalize_embeddings=True,  # L2 normalize → cosine sim = dot product
        )

        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector.tolist()   # ChromaDB needs list, not np.array

        print(f"[EMBEDDER] Done. Each vector: {len(chunks[0]['embedding'])} dims")
        return chunks

    def embed_query(self, query: str) -> list[float]:
        """
        Embeds a single query string for retrieval.
        Must use the same model as embed_documents() — same vector space.
        """
        vector = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,  # must match index-time normalization
        )
        return vector.tolist()

    def similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """
        Cosine similarity between two vectors.
        Since we normalize at encode time, this is just dot product.
        Returns a float in [-1, 1]. Higher = more similar.

        Useful for manual inspection / debugging retrieval.
        """
        a = np.array(vec_a)
        b = np.array(vec_b)
        return float(np.dot(a, b))   # dot product of unit vectors = cosine sim


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pipeline.loader import load_documents
    from pipeline.chunker import chunk_documents

    docs = load_documents()
    chunks = chunk_documents(docs, chunk_size=512, overlap=50)

    embedder = Embedder()
    chunks = embedder.embed_documents(chunks)

    print("\n── Embedding preview ──────────────────────────────")
    sample = chunks[0]
    print(f"chunk_id  : {sample['metadata']['chunk_id']}")
    print(f"text      : {sample['text'][:80]}...")
    print(f"embedding : [{sample['embedding'][0]:.6f}, {sample['embedding'][1]:.6f}, ...]")
    print(f"dims      : {len(sample['embedding'])}")

    # Sanity check: similar texts should have high similarity
    print("\n── Similarity sanity check ────────────────────────")
    query = "What is retrieval augmented generation?"
    q_vec = embedder.embed_query(query)

    scores = [
        (chunk["metadata"]["chunk_id"], embedder.similarity(q_vec, chunk["embedding"]))
        for chunk in chunks
    ]
    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"Query: '{query}'")
    print("Top 3 most similar chunks:")
    for chunk_id, score in scores[:3]:
        print(f"  {score:.4f}  {chunk_id}")
