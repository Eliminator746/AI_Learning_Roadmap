"""
enricher.py — Contextual Retrieval (Anthropic, 2024).

Problem it solves:
  A chunk pulled out of a document loses its surrounding context.
  e.g. a chunk saying "The overlap value should be 10-15% of chunk size"
  gives no hint it's about RAG chunking — so its embedding is generic.

Fix:
  Before embedding, ask the LLM to write a 1-2 sentence context for
  each chunk using the FULL document as reference.
  Prepend that context to the chunk text before embedding.

  "This chunk is from a document about chunking strategies in RAG.
   It specifically discusses recommended overlap values for recursive chunking.
   [original chunk text follows...]"

  Now the embedding captures both the specific content AND the broader topic.
  Retrieval precision improves, especially for short or ambiguous chunks.

Cost tradeoff:
  One LLM call per chunk at index time. If you have 500 chunks → 500 LLM calls.
  Index time gets expensive, query time stays identical.
  Worth it for static knowledge bases. Not worth it for frequently updated docs.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
MODEL = os.getenv("LLM_MODEL", "openrouter/free")


ENRICHMENT_PROMPT = """Here is a document:
<document>
{full_document}
</document>

Here is a specific chunk from that document:
<chunk>
{chunk_text}
</chunk>

Write 1-2 sentences that situate this chunk within the broader document.
Explain what topic the document covers and what specific aspect this chunk addresses.
Reply with ONLY those 1-2 sentences. No preamble, no labels."""


def enrich_chunk(chunk: dict, full_document: str) -> dict:
    """
    Adds LLM-generated context to a single chunk.
    Prepends the context to chunk["text"] before embedding.
    Preserves the original text in metadata for display purposes.
    """
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": ENRICHMENT_PROMPT.format(
                full_document=full_document[:3000],  # cap to avoid token blowout on large docs
                chunk_text=chunk["text"],
            )
        }],
        temperature=0,
        max_tokens=120,
    )

    context = response.choices[0].message.content.strip()

    enriched_chunk = {
        # Prepend context to text — this is what gets embedded
        "text": f"{context}\n\n{chunk['text']}",
        "metadata": {
            **chunk["metadata"],
            "original_text":    chunk["text"],   # preserve for display
            "enrichment_context": context,
            "enriched":         True,
        }
    }
    # Carry over embedding if already computed (shouldn't be, but defensive)
    if "embedding" in chunk:
        enriched_chunk["embedding"] = chunk["embedding"]

    return enriched_chunk


def enrich_documents(documents: list[dict], chunks: list[dict], verbose: bool = True) -> list[dict]:
    """
    Enriches all chunks using their parent document as context.

    Matches each chunk back to its source document via doc_id,
    then calls enrich_chunk() for each one.

    Args:
        documents: raw document dicts from loader.py (needed for full text)
        chunks:    chunk dicts from chunker.py (not yet embedded)

    Returns:
        Enriched chunk list — same shape, ready to pass to embedder.
    """
    # Build doc_id → full text lookup so we don't search on every chunk
    doc_lookup = {doc["metadata"]["doc_id"]: doc["text"] for doc in documents}

    enriched = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        doc_id   = chunk["metadata"]["doc_id"]
        full_doc = doc_lookup.get(doc_id, "")

        if verbose:
            print(f"[ENRICHER] {i+1}/{total}  {chunk['metadata']['chunk_id']}")

        enriched_chunk = enrich_chunk(chunk, full_doc)
        enriched.append(enriched_chunk)

    print(f"\n[ENRICHER] Done. {len(enriched)} chunks enriched.")
    return enriched


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pipeline.loader  import load_documents
    from pipeline.chunker import chunk_documents

    docs   = load_documents()
    chunks = chunk_documents(docs, chunk_size=512, overlap=50)

    # Enrich just the first 2 chunks so the test is fast
    sample   = chunks[:2]
    enriched = enrich_documents(docs, sample)

    print("\n── Before vs After ────────────────────────────────")
    for orig, enr in zip(sample, enriched):
        print(f"\nChunk: {orig['metadata']['chunk_id']}")
        print(f"BEFORE: {orig['text'][:120]}...")
        print(f"AFTER : {enr['text'][:220]}...")
        print(f"Context added: {enr['metadata']['enrichment_context']}")
