"""
reranker.py — cross-encoder reranking using SentenceTransformers.

Why rerank at all?
  Bi-encoder (what we use for embeddings):
    → encodes query and chunk SEPARATELY, compares vectors
    → fast: O(1) per query after indexing
    → weakness: loses interaction between query and chunk tokens

  Cross-encoder (what we use here):
    → sees query AND chunk TOGETHER in one forward pass
    → slower: O(n) per query, runs the model on each (query, chunk) pair
    → strength: much more accurate because it models token-level interaction

  Pattern in production:
    Retrieve K=20 candidates cheaply with bi-encoder
    → rerank to top N=4 accurately with cross-encoder
    → send only those N chunks to the LLM

  You cast a wide net cheap, then filter accurately expensive.
  This keeps latency acceptable while maximising precision.

Model choice:
  cross-encoder/ms-marco-MiniLM-L-6-v2
    - Trained on MS MARCO passage ranking dataset
    - Fast, small (66MB), strong for QA-style retrieval
    - Returns a raw logit score — higher = more relevant
    - SentenceTransformers provides this out of the box
"""

from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        print(f"[RERANKER] Loading model: {model_name}")
        self.model = CrossEncoder(model_name)
        print(f"[RERANKER] Ready.")

    def rerank(self, query: str, results: list[dict], top_n: int = 5) -> list[dict]:
        """
        Reranks a list of retrieval results using the cross-encoder.

        Args:
            query:   the original user query string
            results: list of result dicts from retriever.py
                     (must have "text", "chunk_id", "metadata")
            top_n:   how many to return after reranking

        Returns:
            top_n results sorted by cross-encoder score descending.
            Adds "rerank_score" and "original_rank" keys to each result.

        CrossEncoder.predict() takes a list of (query, passage) pairs
        and returns a numpy array of scores — one float per pair.
        SentenceTransformers handles batching internally.
        """
        if not results:
            return []

        # Build (query, chunk_text) pairs — this is what cross-encoder expects
        pairs = [(query, r["text"]) for r in results]

        # Single call — SentenceTransformers batches this internally
        scores = self.model.predict(pairs)

        # Attach scores and original rank before sorting
        for i, (result, score) in enumerate(zip(results, scores)):
            result["rerank_score"]  = round(float(score), 4)
            result["original_rank"] = result["rank"]   # rank before reranking

        # Sort by cross-encoder score descending
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        # Update rank to reflect new order
        for new_rank, result in enumerate(reranked[:top_n], start=1):
            result["rank"] = new_rank

        return reranked[:top_n]


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from pipeline.loader      import load_documents
    from pipeline.chunker     import chunk_documents
    from pipeline.embedder    import Embedder
    from pipeline.vectorstore import VectorStore
    from pipeline.bm25_store  import BM25Store
    from pipeline.retriever   import Retriever

    # Build full pipeline up to retrieval
    docs   = load_documents()
    chunks = chunk_documents(docs, chunk_size=512, overlap=50)

    embedder = Embedder()
    chunks   = embedder.embed_documents(chunks)

    vector_store = VectorStore()
    vector_store.add_chunks(chunks)

    bm25_store = BM25Store()
    bm25_store.build_index(chunks)

    retriever = Retriever(vector_store, bm25_store, embedder)
    reranker  = Reranker()

    query = "Why does chunk size affect retrieval quality?"

    # Get hybrid results — wider net before reranking
    candidates = retriever.hybrid(query, top_k=10, fetch_k=20)
    reranked   = reranker.rerank(query, candidates, top_n=4)

    print(f"\nQuery: '{query}'")
    print(f"\n── Before reranking (hybrid top-5) ───────────────")
    for r in candidates[:5]:
        print(f"  Rank {r['original_rank']}  score={r['score']}  {r['chunk_id']}")
        print(f"    {r['text'][:100].replace(chr(10), ' ')}...")

    print(f"\n── After reranking (top-4) ───────────────────────")
    for r in reranked:
        moved = r['original_rank'] - r['rank']
        arrow = f"↑{moved}" if moved > 0 else (f"↓{abs(moved)}" if moved < 0 else "─")
        print(f"  Rank {r['rank']} ({arrow})  rerank_score={r['rerank_score']}  {r['chunk_id']}")
        print(f"    {r['text'][:100].replace(chr(10), ' ')}...")

    # The arrow shows rank movement — chunks that moved UP were underranked
    # by embedding similarity but the cross-encoder found them more relevant.
    # That movement IS the value of reranking.
