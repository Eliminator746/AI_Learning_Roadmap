"""
query.py — runs all retrieval strategies on a query and prints a comparison.

Usage:
    python query.py                          # interactive mode, prompts for query
    python query.py "your question here"     # single query from command line

Strategies compared:
    A  Semantic only
    B  BM25 only
    C  Hybrid + RRF
    D  Hybrid + RRF + Rerank
    E  HyDE + Semantic
    F  Multi-query + Hybrid + RRF
    G  Step-back + Hybrid + RRF
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.loader            import load_documents
from pipeline.chunker           import chunk_documents
from pipeline.embedder          import Embedder
from pipeline.vectorstore       import VectorStore
from pipeline.bm25_store        import BM25Store
from pipeline.retriever         import Retriever, _rrf_merge
from pipeline.reranker          import Reranker
from pipeline.query_transformer import hyde, multi_query, step_back
from pipeline.generator         import generate


# ── Initialise all components once ────────────────────────────
def load_pipeline(collection_name: str = "rag_chunks"):
    """
    Loads everything into memory.
    VectorStore loads from disk (ChromaDB persistent).
    BM25 is rebuilt from chunks each run — fast for small corpora.
    """
    print("[QUERY] Loading pipeline...")

    # Need raw chunks for BM25 (BM25 isn't persisted)
    documents = load_documents()
    chunks    = chunk_documents(documents, chunk_size=512, overlap=50)

    embedder     = Embedder()
    vector_store = VectorStore(collection_name=collection_name)
    bm25_store   = BM25Store()
    bm25_store.build_index(chunks)

    retriever = Retriever(vector_store, bm25_store, embedder)
    reranker  = Reranker()

    print(f"[QUERY] Pipeline ready. ChromaDB chunks: {vector_store.count()}\n")
    return embedder, retriever, reranker


# ── Run all strategies ─────────────────────────────────────────
def run_all_strategies(query: str, embedder, retriever, reranker) -> dict:
    results = {}

    # ── A: Semantic only ──────────────────────────────────────
    chunks_a = retriever.semantic(query, top_k=4)
    results["A: Semantic"] = {
        "chunks":  chunks_a,
        "answer":  generate(query, chunks_a),
    }

    # ── B: BM25 only ──────────────────────────────────────────
    chunks_b = retriever.bm25(query, top_k=4)
    results["B: BM25"] = {
        "chunks":  chunks_b,
        "answer":  generate(query, chunks_b),
    }

    # ── C: Hybrid + RRF ───────────────────────────────────────
    chunks_c = retriever.hybrid(query, top_k=4, fetch_k=12)
    results["C: Hybrid+RRF"] = {
        "chunks":  chunks_c,
        "answer":  generate(query, chunks_c),
    }

    # ── D: Hybrid + RRF + Rerank ──────────────────────────────
    candidates_d = retriever.hybrid(query, top_k=12, fetch_k=20)
    chunks_d     = reranker.rerank(query, candidates_d, top_n=4)
    results["D: Hybrid+Rerank"] = {
        "chunks":  chunks_d,
        "answer":  generate(query, chunks_d),
    }

    # ── E: HyDE + Semantic ────────────────────────────────────
    hyde_vec, hyde_text = hyde(query, embedder)
    # Query ChromaDB directly with the HyDE embedding
    chunks_e = retriever.vector_store.query(hyde_vec, top_k=4)
    results["E: HyDE+Semantic"] = {
        "chunks":    chunks_e,
        "hyde_text": hyde_text,
        "answer":    generate(query, chunks_e),
    }

    # ── F: Multi-query + Hybrid + RRF ─────────────────────────
    mq_queries   = multi_query(query, n=3)
    mq_all_results = []
    for q in mq_queries:
        mq_all_results.append(retriever.hybrid(q, top_k=8, fetch_k=15))
    chunks_f = _rrf_merge(mq_all_results, top_k=4)
    results["F: Multi-query"] = {
        "chunks":     chunks_f,
        "mq_queries": mq_queries,
        "answer":     generate(query, chunks_f),
    }

    # ── G: Step-back + Hybrid + RRF ───────────────────────────
    original, broader = step_back(query)
    chunks_orig   = retriever.hybrid(original, top_k=8, fetch_k=15)
    chunks_broad  = retriever.hybrid(broader,  top_k=8, fetch_k=15)
    chunks_g      = _rrf_merge([chunks_orig, chunks_broad], top_k=4)
    results["G: Step-back"] = {
        "chunks":         chunks_g,
        "broader_query":  broader,
        "answer":         generate(query, chunks_g),
    }

    return results


# ── Pretty print comparison ────────────────────────────────────
def print_comparison(query: str, results: dict):
    SEP = "=" * 60

    print(f"\n\n{SEP}")
    print("COMPARISON SUMMARY")
    print(SEP)
    print(f"Query: '{query}'\n")

    # Token + chunk table
    print(f"{'Strategy':<22} {'Chunks':>6} {'Prompt tok':>11} {'Answer tok':>11}")
    print("-" * 55)
    for strategy, data in results.items():
        ans = data["answer"]
        print(
            f"{strategy:<22}"
            f"{ans['chunks_used']:>6}"
            f"{ans['prompt_tokens']:>11}"
            f"{ans['answer_tokens']:>11}"
        )

    # Full answers per strategy
    for strategy, data in results.items():
        print(f"\n{SEP}")
        print(f"Strategy {strategy}")
        print(SEP)

        # Show extra info for transformer strategies
        if "hyde_text" in data:
            print(f"[HyDE generated]: {data['hyde_text'][:120]}...")
        if "mq_queries" in data:
            print(f"[Queries used]:")
            for i, q in enumerate(data["mq_queries"], 1):
                print(f"  {i}. {q}")
        if "broader_query" in data:
            print(f"[Step-back query]: {data['broader_query']}")

        print(f"\nChunks retrieved:")
        for c in data["chunks"]:
            print(f"  [{c['rank']}] {c['chunk_id']}  score={c.get('rerank_score', c['score'])}")

        print(f"\nAnswer:\n{data['answer']['answer']}")
        print(f"\nSources: {data['answer']['sources']}")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    # Get query from CLI arg or prompt interactively
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("\nEnter your question: ").strip()
        if not query:
            query = "Why does chunk size affect retrieval quality?"
            print(f"Using default: '{query}'")

    embedder, retriever, reranker = load_pipeline()
    results = run_all_strategies(query, embedder, retriever, reranker)
    print_comparison(query, results)
