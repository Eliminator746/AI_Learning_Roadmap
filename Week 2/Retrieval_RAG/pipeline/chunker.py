"""
chunker.py — recursive chunking with overlap.

Why recursive over fixed?
  Fixed chunking splits every N chars blindly — often mid-sentence.
  Recursive chunking tries natural boundaries first:
    paragraph (\n\n) → line (\n) → word ( ) → character
  This keeps semantic units intact, which produces better embeddings.

Why overlap?
  Without overlap, a sentence that crosses a chunk boundary is split
  across two chunks — neither chunk is retrievable for a query about
  that sentence. Overlap duplicates the boundary region so it always
  lives fully inside at least one chunk.

Output shape (one dict per chunk):
{
    "text":     "chunk text...",
    "metadata": {
        ...everything from the parent document's metadata...,
        "chunk_index":  3,          # position in this document
        "chunk_total":  12,         # total chunks for this document
        "char_start":   1024,       # where this chunk starts in original text
        "chunk_id":     "what_is_rag__3",   # unique ID for ChromaDB
    }
}
"""


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """
    Internal: tries each separator in order, returns pieces <= chunk_size.
    If a piece is still too large after the current separator, recurses
    with the next separator in the list.
    """
    if not separators:
        # No separators left — hard split by character as last resort
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep = separators[0]
    remaining_seps = separators[1:]

    splits = text.split(sep)
    chunks = []
    current = ""

    for split in splits:
        candidate = (current + sep + split).strip() if current else split.strip()

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Save what we have so far
            if current:
                chunks.append(current)

            # Is this single split itself too large? Recurse deeper.
            if len(split) > chunk_size:
                sub_chunks = _recursive_split(split, remaining_seps, chunk_size)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = split.strip()

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]   # drop empty strings


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """
    Prepends the last `overlap` chars of chunk[i-1] to chunk[i].
    This duplicates the boundary region so it's fully inside both chunks.
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]   # last `overlap` chars of previous chunk
        overlapped.append(tail + " " + chunks[i])

    return overlapped


def chunk_document(
    document: dict,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[dict]:
    """
    Takes ONE document dict (from loader.py) and returns a list of chunk dicts.

    Args:
        document:   output dict from loader.load_documents()
        chunk_size: max characters per chunk (not tokens — tokens ≈ chars/4)
        overlap:    chars to repeat from end of previous chunk into start of next

    Returns:
        list of chunk dicts, each with text + enriched metadata
    """
    text = document["text"]
    parent_metadata = document["metadata"]

    # Order matters: try largest natural boundaries first
    separators = ["\n\n", "\n", ". ", " ", ""]

    raw_chunks = _recursive_split(text, separators, chunk_size)
    overlapped_chunks = _apply_overlap(raw_chunks, overlap)

    total = len(overlapped_chunks)
    result = []

    # Track approximate char position in original text for metadata
    char_cursor = 0

    for i, chunk_text in enumerate(overlapped_chunks):
        chunk_id = f"{parent_metadata['doc_id']}__{i}"

        chunk = {
            "text": chunk_text,
            "metadata": {
                **parent_metadata,           # inherit all parent metadata
                "chunk_index": i,
                "chunk_total": total,
                "char_start":  char_cursor,
                "chunk_id":    chunk_id,
                "chunk_size_used": chunk_size,
                "overlap_used":    overlap,
            }
        }
        result.append(chunk)

        # Advance cursor (approximate — overlap means this isn't exact)
        char_cursor += len(chunk_text) - overlap

    return result


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[dict]:
    """Chunk a list of documents. Thin wrapper over chunk_document."""
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        print(f"[CHUNKER] {doc['metadata']['filename']} → {len(chunks)} chunks")

    print(f"\n[CHUNKER] Total chunks: {len(all_chunks)}")
    return all_chunks


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pipeline.loader import load_documents

    docs = load_documents()
    chunks = chunk_documents(docs, chunk_size=512, overlap=50)

    print("\n── Chunk preview ─────────────────────────────────")
    for chunk in chunks[:3]:
        m = chunk["metadata"]
        print(f"\nchunk_id    : {m['chunk_id']}")
        print(f"chunk_index : {m['chunk_index']} / {m['chunk_total'] - 1}")
        print(f"char_start  : {m['char_start']}")
        print(f"char_count  : {len(chunk['text'])}")
        print(f"text preview: {chunk['text'][:120].replace(chr(10), ' ')}...")

    # Show that overlap is working — chunk[1] should start with tail of chunk[0]
    if len(chunks) >= 2:
        print("\n── Overlap check ──────────────────────────────────")
        tail_of_0 = chunks[0]["text"][-50:]
        start_of_1 = chunks[1]["text"][:60]
        print(f"End of chunk 0  : ...{tail_of_0}")
        print(f"Start of chunk 1: {start_of_1}...")
        print(f"Overlap present : {tail_of_0.strip() in chunks[1]['text']}")
