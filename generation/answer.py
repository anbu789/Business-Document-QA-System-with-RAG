"""
generation/answer.py
--------------------
Phase 4 — Answer Generation

Takes the top-5 reranked chunks from Phase 3 and generates a grounded,
cited answer using the Gemini API.

Three responsibilities:
  1. build_context()     — format chunks into numbered [Source N] passages
  2. generate_answer()   — call Gemini with system prompt + context + question
  3. parse_sources()     — detect which [Source N] labels appear in the answer
"""

import os
import re
from typing import Optional
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "gemini-3.1-flash-lite"

SYSTEM_PROMPT = """You are a precise business document assistant.

Your job is to answer questions using ONLY the numbered source passages provided below.

Rules you must follow:
- Base every claim in your answer on the provided passages.
- When you use information from a passage, cite it inline as [Source N].
- If the answer cannot be found in the passages, say exactly: "I could not find this information in the provided documents."
- Do not add facts, numbers, dates, or names that are not in the passages.
- Be concise and direct. Do not repeat information unnecessarily.
- If multiple sources support the same point, cite all of them, e.g. [Source 1][Source 3].
"""

# ---------------------------------------------------------------------------
# Step 1: Build context string from retrieved chunks
# ---------------------------------------------------------------------------

def build_context(chunks: list[dict]) -> str:
    """
    Format a list of reranked chunks into a numbered context block.

    Each chunk dict is expected to have:
      - "text"      : the raw chunk text
      - "metadata"  : dict with at least "filename" and "source_type"

    Returns a string like:
      [Source 1] invoices.csv (csv)
      invoice_id: INV-2024-004, client: Nexus Solutions ...

      [Source 2] service_contract_nexus_solutions.pdf (pdf)
      SERVICE AGREEMENT between TechVentures Inc. ...
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        filename    = chunk["metadata"].get("filename", "unknown")
        source_type = chunk["metadata"].get("source_type", "unknown")
        text        = chunk["text"].strip()

        header = f"[Source {i}] {filename} ({source_type})"
        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Step 2: Call Gemini and get a grounded answer
# ---------------------------------------------------------------------------

def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Generate a grounded answer for `query` using the provided chunks.

    Args:
        query  : the user's question (string)
        chunks : list of reranked chunk dicts from Phase 3

    Returns a dict:
        {
            "answer"       : str   — the LLM's response
            "sources_used" : list  — e.g. ["invoices.csv (chunk 0)", ...]
            "context"      : str   — the formatted context sent to the LLM
            "query"        : str   — the original query (for reference)
        }
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment. Check your .env file.")

    # Build the numbered context block
    context = build_context(chunks)

    # Compose the user message: context passages + the question
    user_message = f"""Here are the relevant passages from the business documents:

{context}

Question: {query}

Answer (cite sources inline as [Source N]):"""

    # Call Gemini
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,        # deterministic — we want grounded answers, not creative ones
            max_output_tokens=1024,
        ),
    )

    answer_text = response.text.strip()

    # Parse which source numbers appear in the answer
    sources_used = parse_sources(answer_text, chunks)

    return {
        "query"        : query,
        "answer"       : answer_text,
        "sources_used" : sources_used,
        "context"      : context,
    }


# ---------------------------------------------------------------------------
# Step 3: Parse which [Source N] labels the LLM cited in its answer
# ---------------------------------------------------------------------------

def parse_sources(answer_text: str, chunks: list[dict]) -> list[str]:
    """
    Scan the answer text for [Source N] citations and return a human-readable
    list of which source files were actually used.

    Example:
      answer_text = "Nexus Solutions has overdue invoices [Source 1][Source 3]."
      returns     = ["invoices.csv (chunk 0)", "employees.csv (chunk 1)"]
    """
    # Find all [Source N] patterns, e.g. [Source 1], [Source 3]
    cited_numbers = re.findall(r'\[Source (\d+)\]', answer_text)
    cited_indices = sorted(set(int(n) - 1 for n in cited_numbers))  # convert to 0-based

    sources_used = []
    for idx in cited_indices:
        if 0 <= idx < len(chunks):
            meta     = chunks[idx]["metadata"]
            filename = meta.get("filename", "unknown")
            chunk_i  = meta.get("chunk_index", "?")
            sources_used.append(f"{filename} (chunk {chunk_i})")

    return sources_used
