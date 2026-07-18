"""
query_transformer.py — rewrites the user query BEFORE retrieval.

Why transform the query at all?
  The user's raw query is often a poor retrieval signal:
    - Too short:   "chunk size?"  → embedding is noisy, BM25 gets 1 term
    - Too specific: "what's the exact token overlap in recursive chunking"
                    → misses chunks that explain the concept without that phrasing
    - Wrong form:  queries are questions, documents are answers
                   → their embeddings naturally live in different vector regions

Three techniques, each solving a different failure mode:

  HyDE (Hypothetical Document Embedding)
    Problem:  query embeddings and document embeddings live in different
              vector spaces — a question embeds differently than an answer.
    Fix:      ask the LLM to generate a hypothetical answer first.
              Embed THAT instead of the raw question.
              A fake answer lives in the same vector region as real answers.
    Best for: factual QA where the answer form is predictable.

  Multi-Query
    Problem:  one phrasing might miss relevant chunks due to vocabulary mismatch.
    Fix:      generate N alternative phrasings of the same question.
              Retrieve for each, merge all result lists via RRF.
              More surface area = higher recall.
    Best for: broad questions, exploratory queries.

  Step-Back
    Problem:  query is too specific — retriever can't find a chunk that
              answers exactly that, but a broader chunk contains the answer.
    Fix:      ask the LLM "what broader topic does this question belong to?"
              Retrieve for that broader topic instead.
    Best for: narrow technical questions buried inside larger concepts.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Point at whichever free endpoint you're using
# Change base_url + api_key + MODEL to switch providers — nothing else changes
_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
)
MODEL = os.getenv("LLM_MODEL", "openrouter/free")


def _llm_call(prompt: str, max_tokens: int = 300) -> str:
    """Single LLM call — used by all three transformers."""
    r = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content.strip()


# ──────────────────────────────────────────────────────
# Technique 1: HyDE
# ──────────────────────────────────────────────────────

HYDE_PROMPT = """Write a short factual passage (3-5 sentences) that directly answers the question below.
Write as if you are an expert writing a knowledge base article.
Do not say "the answer is" — just write the passage directly.

Question: {query}

Passage:"""


def hyde(query: str, embedder) -> tuple[list[float], str]:
    """
    Generates a hypothetical answer, embeds it instead of the raw query.

    Returns:
        (embedding, hypothetical_text)
        Pass the embedding to vectorstore.query() instead of the raw query embedding.
        Return the hypothetical_text so you can log/debug what was generated.
    """
    print(f"[HYDE] Generating hypothetical answer for: '{query[:60]}'")
    hypothetical = _llm_call(HYDE_PROMPT.format(query=query))
    print(f"[HYDE] Generated: {hypothetical[:120]}...")

    # Embed the hypothetical answer — NOT the original query
    embedding = embedder.embed_query(hypothetical)
    return embedding, hypothetical


# ──────────────────────────────────────────────────────
# Technique 2: Multi-Query
# ──────────────────────────────────────────────────────

MULTI_QUERY_PROMPT = """Generate {n} different ways to ask the following question.
Each version should use different vocabulary and phrasing but mean the same thing.
Return ONLY the questions, one per line, no numbering, no explanation.

Original question: {query}

Alternative questions:"""


def multi_query(query: str, n: int = 3) -> list[str]:
    """
    Generates N alternative phrasings of the same query.

    The caller should:
      1. Retrieve results for each alternative query
      2. Merge all result lists using RRF (already in retriever._rrf_merge)

    Returns:
        List of alternative query strings (includes original at position 0).
    """
    print(f"[MULTI-QUERY] Generating {n} alternatives for: '{query[:60]}'")
    raw = _llm_call(MULTI_QUERY_PROMPT.format(query=query, n=n))

    alternatives = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and line.strip() != query
    ][:n]

    queries = [query] + alternatives   # original first, then alternatives
    print(f"[MULTI-QUERY] Generated {len(queries)} total queries:")
    for i, q in enumerate(queries):
        print(f"  [{i}] {q}")

    return queries


# ──────────────────────────────────────────────────────
# Technique 3: Step-Back
# ──────────────────────────────────────────────────────

STEP_BACK_PROMPT = """Your task is to make a question more general.
Given a specific question, generate a broader question that covers the underlying concept.

Examples:
  Specific: "What is the exact overlap value used in recursive chunking?"
  Broader:  "How does chunk overlap work in document splitting?"

  Specific: "What is HNSW's ef_construction parameter?"
  Broader:  "How do vector database indexes work?"

Now do the same for this question:
  Specific: "{query}"
  Broader:"""


def step_back(query: str) -> tuple[str, str]:
    """
    Generates a broader version of the query for retrieval.

    Returns:
        (original_query, broader_query)
        Retrieve for the broader_query — it catches chunks that explain
        the concept without using the specific terminology.
        The caller can retrieve for both and RRF merge the results.
    """
    print(f"[STEP-BACK] Abstracting: '{query[:60]}'")
    broader = _llm_call(STEP_BACK_PROMPT.format(query=query), max_tokens=80)
    # Strip any "Broader:" prefix the model might include
    broader = broader.replace("Broader:", "").strip().strip('"')
    print(f"[STEP-BACK] → '{broader}'")
    return query, broader


# ──────────────────────────────────────────────────────
# Convenience wrapper — runs all three on one query
# Useful for comparison in query.py
# ──────────────────────────────────────────────────────

def transform_all(query: str, embedder) -> dict:
    """
    Runs all three transformations and returns their outputs in one dict.
    Use this in query.py when you want to compare strategies.
    """
    hyde_embedding, hyde_text   = hyde(query, embedder)
    mq_queries                  = multi_query(query, n=3)
    original, broader           = step_back(query)

    return {
        "original":       query,
        "hyde_text":      hyde_text,
        "hyde_embedding": hyde_embedding,   # pass directly to vectorstore.query()
        "mq_queries":     mq_queries,       # retrieve for each, then RRF merge
        "step_back":      broader,          # retrieve for this broader query
    }


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pipeline.embedder import Embedder

    embedder = Embedder()
    query    = "What is the exact overlap value used in recursive chunking?"

    print("=" * 60)
    print(f"Original query: '{query}'")
    print("=" * 60)

    # HyDE
    print("\n── HyDE ───────────────────────────────────────────")
    embedding, hypo = hyde(query, embedder)
    print(f"Hypothetical doc: {hypo}")
    print(f"Embedding dims: {len(embedding)}")

    # Multi-query
    print("\n── Multi-Query ────────────────────────────────────")
    queries = multi_query(query, n=3)

    # Step-back
    print("\n── Step-Back ──────────────────────────────────────")
    orig, broad = step_back(query)
    print(f"Original : {orig}")
    print(f"Broader  : {broad}")
