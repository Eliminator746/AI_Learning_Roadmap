"""
generator.py — prompt engineering + LLM call to produce the final answer.

This is the "AG" part of RAG (Augmented Generation).
The retriever's job is done — we now have the top-N relevant chunks.
The generator's job: format them into a prompt and get an answer.

Prompt engineering principles applied here:
  1. Role + task in system prompt    — grounds the model's behaviour
  2. Numbered sources                — lets model cite [1], [2] naturally
  3. Explicit "only use context"     — reduces hallucination
  4. Fallback instruction            — tells model what to do if context
                                       doesn't contain the answer
                                       (say so, don't make something up)
  5. Query restated at the end       — keeps the question fresh after
                                       potentially long context block
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
)
MODEL = os.getenv("LLM_MODEL", "openrouter/free")


SYSTEM_PROMPT = """You are a precise question-answering assistant.
You will be given a question and a set of numbered context passages retrieved from a knowledge base.

Rules:
- Answer using ONLY the information in the provided context.
- If the context does not contain enough information, say: "I don't have enough context to answer this."
- Be concise and direct.
- You may reference sources as [1], [2], etc. when useful."""


RAG_PROMPT = """Context passages:
{context}

Question: {query}

Answer:"""


def _format_context(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into numbered passages for the prompt.

    Numbered format lets the model cite sources naturally ([1], [2])
    and makes it easy to trace which chunk produced which claim.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"].get("filename", "unknown")
        parts.append(f"[{i}] (source: {source})\n{chunk['text']}")
    return "\n\n".join(parts)


def generate(query: str, chunks: list[dict], max_tokens: int = 400) -> dict:
    """
    Generates a final answer from the query + retrieved chunks.

    Args:
        query:      original user question
        chunks:     list of retrieved chunk dicts (from retriever/reranker)
        max_tokens: cap on answer length

    Returns:
        {
            "answer":         "...",
            "sources":        ["chunking_strategies.txt", ...],
            "chunks_used":    3,
            "prompt_tokens":  412,
            "answer_tokens":  89,
        }

    Returning token counts here is important — this is where you measure
    the "cost" of each retrieval strategy when comparing them in query.py.
    """
    if not chunks:
        return {
            "answer":        "No relevant context was retrieved.",
            "sources":       [],
            "chunks_used":   0,
            "prompt_tokens": 0,
            "answer_tokens": 0,
        }

    context = _format_context(chunks)
    user_prompt = RAG_PROMPT.format(context=context, query=query)

    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,     # low temp for factual QA — we want consistent answers
        max_tokens=max_tokens,
    )

    answer = response.choices[0].message.content.strip()
    sources = list({c["metadata"].get("filename", "unknown") for c in chunks})

    return {
        "answer":        answer,
        "sources":       sources,
        "chunks_used":   len(chunks),
        "prompt_tokens": response.usage.prompt_tokens,
        "answer_tokens": response.usage.completion_tokens,
    }


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from pipeline.loader      import load_documents
    from pipeline.chunker     import chunk_documents
    from pipeline.embedder    import Embedder
    from pipeline.vectorstore import VectorStore
    from pipeline.bm25_store  import BM25Store
    from pipeline.retriever   import Retriever
    from pipeline.reranker    import Reranker

    # Full pipeline
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

    query = "Why does chunk size matter for retrieval quality?"

    # Strategy D: Hybrid + Rerank → Generate
    candidates = retriever.hybrid(query, top_k=10, fetch_k=20)
    top_chunks = reranker.rerank(query, candidates, top_n=4)
    result     = generate(query, top_chunks)

    print(f"\nQuery: '{query}'")
    print(f"\n── Answer ─────────────────────────────────────────")
    print(result["answer"])
    print(f"\n── Metadata ───────────────────────────────────────")
    print(f"Sources       : {result['sources']}")
    print(f"Chunks used   : {result['chunks_used']}")
    print(f"Prompt tokens : {result['prompt_tokens']}")
    print(f"Answer tokens : {result['answer_tokens']}")
