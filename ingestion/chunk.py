def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks by word count.

    Why word-based and not character-based?
    Word counts approximate token counts well enough for our chunk_size range
    (300-500 tokens), and it's simpler than running a full tokenizer.

    Args:
        text: Raw extracted text
        chunk_size: Target words per chunk (default 400)
        overlap: Words shared between adjacent chunks (default 50)

    Returns:
        List of text chunk strings
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Move forward by (chunk_size - overlap) so next chunk
        # re-includes the last `overlap` words of this chunk
        start += chunk_size - overlap

    return chunks


def chunk_document(doc: dict, chunk_size: int = 400, overlap: int = 50) -> list[dict]:
    """
    Chunk a document dict from extract.py and attach metadata to each chunk.

    Args:
        doc: { "text": str, "source_type": str, "filename": str }

    Returns:
        List of chunk dicts with text + metadata
    """
    raw_chunks = chunk_text(doc["text"], chunk_size, overlap)

    return [
        {
            "text": chunk,
            "metadata": {
                "filename": doc["filename"],
                "source_type": doc["source_type"],
                "chunk_index": i,
            },
        }
        for i, chunk in enumerate(raw_chunks)
    ]