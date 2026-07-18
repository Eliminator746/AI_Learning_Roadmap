"""
bm25_store.py — BM25 keyword index for retrieval.

Why BM25 alongside a vector store?
  Vector search finds semantically similar chunks — great for paraphrases,
  synonyms, conceptual queries. But it can miss exact keyword matches.

  Example:
    Query:   "What is HNSW?"
    Vector:  finds chunks about "approximate nearest neighbour search" ✅
    BM25:    finds chunks containing the exact string "HNSW"           ✅

  BM25 excels when the query contains rare proper nouns, acronyms, or
  exact terms that don't embed well. Hybrid search (vector + BM25 via RRF)
  gets the best of both worlds — that's why we build both separately.

How BM25 works (intuition, not math):
  It scores each chunk based on:
    1. Term frequency  — how often query words appear in the chunk
    2. IDF             — rare words get higher weight than common words
    3. Document length — penalises long chunks to avoid length bias

  "BM25" stands for "Best Match 25" — it's the 25th iteration of a
  family of ranking functions, developed in the 1970s-90s. Still the
  standard baseline for keyword search in 2024 (used inside Elasticsearch).

Important: BM25 is NOT persistent like ChromaDB.
  It's an in-memory index built from your chunks list at runtime.
  This is fine for development — for production you'd serialise with pickle.
"""

import re
import os
import math
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """
    Lowercase + split on non-alphanumeric characters.
    This is the tokenizer used at both INDEX time and QUERY time.

    Critical: you must tokenize queries EXACTLY the same way you tokenized
    documents. Mismatch = tokens never match = zero scores for everything.

    "What is RAG?" → ["what", "is", "rag"]
    """
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return tokens


class BM25Store:
    """
    Pure Python BM25 implementation.
    Builds from a list of chunk dicts (same format as vectorstore input).

    k1 and b are standard BM25 tuning params:
      k1 = 1.5  controls term frequency saturation (higher = TF matters more)
      b  = 0.75 controls length normalisation (1.0 = full normalise, 0 = none)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b

        # Built by build_index()
        self.chunks: list[dict]       = []
        self.tokenized: list[list[str]] = []   # tokenized version of each chunk
        self.df: dict[str, int]       = {}     # document frequency per term
        self.idf: dict[str, float]    = {}     # inverse document frequency per term
        self.avgdl: float             = 0.0    # average document length in tokens
        self.n_docs: int              = 0

    def build_index(self, chunks: list[dict]) -> None:
        """
        Tokenizes all chunks and computes IDF for every term.
        Must be called before query().

        Two passes:
          Pass 1: tokenize each chunk, collect document frequencies
          Pass 2: compute IDF from document frequencies
        """
        self.chunks    = chunks
        self.n_docs    = len(chunks)
        self.tokenized = []
        self.df        = {}

        print(f"[BM25] Building index over {self.n_docs} chunks...")

        total_tokens = 0
        for chunk in chunks:
            tokens = _tokenize(chunk["text"])
            self.tokenized.append(tokens)
            total_tokens += len(tokens)

            # DF: how many documents contain this term (count each term once per doc)
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

        self.avgdl = total_tokens / self.n_docs if self.n_docs > 0 else 1

        # IDF formula (Robertson-Sparck Jones):
        # idf(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        # The +1 outside the log ensures IDF is always positive.
        # Terms appearing in every document get IDF ≈ 0 (they're useless for ranking).
        for term, df_val in self.df.items():
            self.idf[term] = math.log(
                (self.n_docs - df_val + 0.5) / (df_val + 0.5) + 1
            )

        print(f"[BM25] Index built. Vocab size: {len(self.df)} terms. Avg doc length: {self.avgdl:.1f} tokens")

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        """
        BM25 score for one (query, document) pair.

        For each query term:
          - Get IDF weight (0 if term not in vocab — unseen terms contribute nothing)
          - Get term frequency in this document
          - Apply BM25 saturation + length normalisation formula
          - Sum across all query terms
        """
        doc_len = len(doc_tokens)
        tf_map  = Counter(doc_tokens)   # term → count in this document
        score   = 0.0

        for term in query_tokens:
            if term not in self.idf:
                continue   # term not in any document — contributes 0

            tf  = tf_map.get(term, 0)
            idf = self.idf[term]

            # BM25 TF component with length normalisation:
            # numerator:   tf * (k1 + 1)  — boosts TF but with diminishing returns
            # denominator: tf + k1 * (1 - b + b * doc_len / avgdl)
            #              — longer docs are penalised (b controls how much)
            numerator   = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)

            score += idf * (numerator / denominator if denominator > 0 else 0)

        return score

    def query(self, query_text: str, top_k: int = 10) -> list[dict]:
        """
        Keyword search: scores all chunks against query_text, returns top_k.

        Returns same shape as VectorStore.query() so retriever.py
        can treat both outputs identically.

        [
            {
                "chunk_id": "what_is_rag__2",
                "text":     "chunk text...",
                "score":    4.23,        # BM25 score (no upper bound, higher = better)
                "metadata": {...},
                "rank":     1,
            },
            ...
        ]

        Note: BM25 scores are NOT comparable to cosine similarity scores.
          BM25: unbounded float, depends on query length and corpus size
          Cosine: always in [-1, 1]
          This is why RRF uses RANKS not scores when fusing the two lists.
        """
        if not self.chunks:
            raise RuntimeError("Index is empty. Call build_index() first.")

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []

        # Score every chunk — O(n_docs * query_len), fast enough for < 100k chunks
        scored = []
        for i, (chunk, doc_tokens) in enumerate(zip(self.chunks, self.tokenized)):
            score = self._score(query_tokens, doc_tokens)
            if score > 0:    # skip zero-score chunks entirely
                scored.append((score, i))

        # Sort descending by score, take top_k
        scored.sort(reverse=True)
        top = scored[:top_k]

        results = []
        for rank, (score, idx) in enumerate(top, start=1):
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["metadata"]["chunk_id"],
                "text":     chunk["text"],
                "score":    round(score, 4),
                "metadata": chunk["metadata"],
                "rank":     rank,
            })

        return results


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from pipeline.loader import load_documents
    from pipeline.chunker import chunk_documents

    docs   = load_documents()
    chunks = chunk_documents(docs, chunk_size=512, overlap=50)

    bm25 = BM25Store()
    bm25.build_index(chunks)

    # Test 1: specific keyword query — BM25 should nail this
    query1 = "recursive chunking separators"
    results1 = bm25.query(query1, top_k=3)
    print(f"\n── BM25 results ──────────────────────────────────")
    print(f"Query: '{query1}'\n")
    for r in results1:
        print(f"Rank {r['rank']}  score={r['score']}  id={r['chunk_id']}")
        print(f"  {r['text'][:120].replace(chr(10), ' ')}...")
        print()

    # Test 2: vague conceptual query — BM25 may struggle here
    # (vector search would do better — this sets up the hybrid story)
    query2 = "why retrieval matters"
    results2 = bm25.query(query2, top_k=3)
    print(f"── BM25 results ──────────────────────────────────")
    print(f"Query: '{query2}'\n")
    for r in results2:
        print(f"Rank {r['rank']}  score={r['score']}  id={r['chunk_id']}")
        print(f"  {r['text'][:120].replace(chr(10), ' ')}...")
        print()

    # Show IDF insight — common words have low IDF, rare words have high IDF
    print("── IDF spot check (higher = rarer = more discriminative) ──")
    check_terms = ["the", "chunking", "recursive", "bm25", "embedding", "is"]
    for term in check_terms:
        idf = bm25.idf.get(term, 0)
        print(f"  '{term}': IDF = {idf:.4f}")
