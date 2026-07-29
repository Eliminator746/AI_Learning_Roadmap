"""
bm25_store.py — BM25 keyword search using rank_bm25.

rank_bm25 implements BM25Okapi out of the box.
We just tokenize, build the index, and query it.
"""

import re
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """
    Lowercase + split on non-alphanumeric characters.
    Must be identical at index time and query time — mismatch = zero scores.
    "What is RAG?" → ["what", "is", "rag"]
    """
    return re.findall(r'\b[a-z0-9]+\b', text.lower())


class BM25Store:
    def __init__(self):
        self.chunks: list[dict] = []
        self.index: BM25Okapi  = None

    def build_index(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        tokenized   = [_tokenize(c["text"]) for c in chunks]
        self.index  = BM25Okapi(tokenized)
        print(f"[BM25] Index built over {len(chunks)} chunks.")

    def query(self, query_text: str, top_k: int = 10) -> list[dict]:
        if not self.index:
            raise RuntimeError("Call build_index() first.")

        query_tokens = _tokenize(query_text)
        scores       = self.index.get_scores(query_tokens)

        # Pair each score with its chunk index, sort descending, take top_k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            if score == 0:
                continue    # skip chunks with no keyword overlap
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["metadata"]["chunk_id"],
                "text":     chunk["text"],
                "score":    round(float(score), 4),
                "metadata": chunk["metadata"],
                "rank":     rank,
            })

        return results


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pipeline.loader  import load_documents
    from pipeline.chunker import chunk_documents

    chunks = chunk_documents(load_documents(), chunk_size=512, overlap=50)

    bm25 = BM25Store()
    bm25.build_index(chunks)

    for query in ["recursive chunking separators", "why retrieval matters"]:
        results = bm25.query(query, top_k=3)
        print(f"\nQuery: '{query}'")
        for r in results:
            print(f"  Rank {r['rank']}  score={r['score']}  {r['chunk_id']}")
            print(f"    {r['text'][:100].replace(chr(10), ' ')}...")