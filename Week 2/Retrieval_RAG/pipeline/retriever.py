"""
retriever.py — three retrieval strategies built on VectorStore + BM25Store.

Strategy A: Semantic only     — pure vector similarity via ChromaDB
Strategy B: BM25 only         — pure keyword matching
Strategy C: Hybrid + RRF      — fuses A and B using Reciprocal Rank Fusion

All three return the same shape so the rest of the pipeline
(reranker, generator) doesn't need to know which strategy was used.

──────────────────────────────────────────────────────
What is RRF (Reciprocal Rank Fusion)?

Problem: vector scores and BM25 scores live on different scales.
  Vector: cosine similarity → always in [-1, 1]
  BM25:   unbounded float   → depends on query length, corpus size

You can't average them directly. But RANKS are always comparable:
  Rank 1 = best match, regardless of which system produced it.

RRF formula:
  rrf_score(chunk) = Σ  1 / (k + rank_in_list_i)
                    lists

  k = 60  (from the original paper — dampens the impact of top ranks)

Example with k=60:
  chunk appears at rank 1 in semantic list  → 1/(60+1) = 0.0164
  chunk appears at rank 3 in BM25 list      → 1/(60+3) = 0.0159
  combined rrf_score                         = 0.0323

  chunk missing from BM25 list entirely     → contributes 0 from that list
  so consistently-ranked chunks beat one-list wonders

This is why hybrid beats either individual system — a chunk that ranks
well in BOTH lists gets a much higher RRF score than one that ranks
well in only one.
──────────────────────────────────────────────────────
"""


RRF_K = 60   # standard value from the original RRF paper (Cormack et al., 2009)


def _rrf_merge(ranked_lists: list[list[dict]], top_k: int) -> list[dict]:
    """
    Merges multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: list of result lists, each already sorted best-first.
                      Each item must have "chunk_id", "text", "metadata".
        top_k:        how many results to return after fusion.

    Returns:
        Merged list sorted by RRF score descending, length = top_k.
    """
    rrf_scores = {}   # chunk_id → accumulated RRF score
    chunk_data  = {}  # chunk_id → {text, metadata} for reconstruction

    for result_list in ranked_lists:
        for rank, result in enumerate(result_list, start=1):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1 / (RRF_K + rank))

            # Store chunk data the first time we see this chunk_id
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = {
                    "text":     result["text"],
                    "metadata": result["metadata"],
                }

    # Sort by RRF score descending
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for rank, (chunk_id, score) in enumerate(sorted_chunks[:top_k], start=1):
        results.append({
            "chunk_id": chunk_id,
            "text":     chunk_data[chunk_id]["text"],
            "metadata": chunk_data[chunk_id]["metadata"],
            "score":    round(score, 6),
            "rank":     rank,
        })

    return results


class Retriever:
    def __init__(self, vector_store, bm25_store, embedder):
        """
        Args:
            vector_store: VectorStore instance (already indexed)
            bm25_store:   BM25Store instance (already indexed)
            embedder:     Embedder instance (for embedding queries)
        """
        self.vector_store = vector_store
        self.bm25_store   = bm25_store
        self.embedder     = embedder

    def semantic(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Strategy A: pure vector similarity.
        Embeds the query → finds closest chunk vectors in ChromaDB.

        Best for: paraphrases, conceptual queries, synonyms.
        Weakness: misses exact keyword matches if phrasing differs.
        """
        query_vec = self.embedder.embed_query(query)
        results   = self.vector_store.query(query_vec, top_k=top_k)

        print(f"[RETRIEVER:semantic] '{query[:60]}' → {len(results)} results")
        return results

    def bm25(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Strategy B: pure keyword matching via BM25.

        Best for: exact terms, acronyms, proper nouns (e.g. "HNSW", "BM25").
        Weakness: misses semantic similarity — "car" won't match "automobile".
        """
        results = self.bm25_store.query(query, top_k=top_k)

        print(f"[RETRIEVER:bm25]     '{query[:60]}' → {len(results)} results")
        return results

    def hybrid(self, query: str, top_k: int = 10, fetch_k: int = 20) -> list[dict]:
        """
        Strategy C: Hybrid search with RRF fusion.

        Fetches fetch_k candidates from each system (cast wide net),
        then fuses using RRF and returns top_k (narrow to best).

        fetch_k > top_k is intentional:
          Both systems may rank the same chunk highly for different reasons.
          Fetching more candidates before fusion gives RRF more signal to work with.
          Standard practice: fetch_k = 2-4x top_k.

        Best for: most production use cases — gets keyword precision AND
                  semantic recall in one pass.
        """
        query_vec      = self.embedder.embed_query(query)
        semantic_results = self.vector_store.query(query_vec, top_k=fetch_k)
        bm25_results     = self.bm25_store.query(query, top_k=fetch_k)

        fused = _rrf_merge([semantic_results, bm25_results], top_k=top_k)

        print(f"[RETRIEVER:hybrid]   '{query[:60]}' → {len(fused)} results (fused from {len(semantic_results)}+{len(bm25_results)})")
        return fused


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from pipeline.loader       import load_documents
    from pipeline.chunker      import chunk_documents
    from pipeline.embedder     import Embedder
    from pipeline.vectorstore  import VectorStore
    from pipeline.bm25_store   import BM25Store

    # Build full pipeline
    docs   = load_documents()
    chunks = chunk_documents(docs, chunk_size=512, overlap=50)

    embedder = Embedder()
    chunks   = embedder.embed_documents(chunks)

    vector_store = VectorStore()
    vector_store.add_chunks(chunks)

    bm25_store = BM25Store()
    bm25_store.build_index(chunks)

    retriever = Retriever(vector_store, bm25_store, embedder)

    # Run same query through all three strategies and compare
    query = "recursive chunking separators"
    top_k = 3

    sem_results    = retriever.semantic(query, top_k=top_k)
    bm25_results   = retriever.bm25(query, top_k=top_k)
    hybrid_results = retriever.hybrid(query, top_k=top_k, fetch_k=10)

    def print_results(label, results):
        print(f"\n── {label} ──────────────────────────────────")
        for r in results:
            print(f"  Rank {r['rank']}  score={r['score']}  {r['chunk_id']}")
            print(f"    {r['text'][:100].replace(chr(10), ' ')}...")

    print(f"\nQuery: '{query}'")
    print_results("Strategy A: Semantic", sem_results)
    print_results("Strategy B: BM25",     bm25_results)
    print_results("Strategy C: Hybrid",   hybrid_results)

    # Key observation to make:
    # A chunk that appears in both semantic AND bm25 lists will rank higher
    # in hybrid than in either individual list — that's RRF working correctly.
    semantic_ids = {r["chunk_id"] for r in sem_results}
    bm25_ids     = {r["chunk_id"] for r in bm25_results}
    overlap      = semantic_ids & bm25_ids
    print(f"\n── Overlap between semantic and BM25 top-{top_k} ──────")
    print(f"  Chunks in both lists: {overlap or 'none'}")
    print(f"  These chunks get RRF boost in hybrid strategy.")
