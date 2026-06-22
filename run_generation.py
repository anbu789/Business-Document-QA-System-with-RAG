"""
run_generation.py
-----------------
Phase 4 end-to-end runner.

Wires the full pipeline together:
  query → hybrid retrieval → cross-encoder rerank → scope guard → Gemini answer

The scope guard sits between reranking and generation. It uses the
cross-encoder's own top score (already computed during reranking — no
extra cost) to decide if the query is even plausibly about the business
documents. Only borderline cases spend an extra Gemini call on a yes/no
classification. Clearly off-topic queries (e.g. "write me DFS/BFS code")
never reach generate_answer() at all.

Run from the project root with venv activated:
  python -m run_generation

Prerequisites:
  - Ingestion must have been run first (chroma_store/ and bm25_index.pkl must exist)
  - GOOGLE_API_KEY must be set in .env
"""

from retrieval.hybrid import hybrid_search
from retrieval.rerank import rerank
from generation.answer import generate_answer
from guardrail.scope_check import is_in_scope

# ---------------------------------------------------------------------------
# Sample queries — same cross-document questions from Phase 2, plus one
# deliberately off-topic query to demonstrate the scope guard in action.
# Remove the last one once you're done eyeballing the guard's behavior.
# ---------------------------------------------------------------------------

QUERIES = [
    "Which invoices from Nexus Solutions are overdue?",
    "Who signed the contract with Nexus Solutions and what is their role?",
    "What are the payment terms in the service contract?",
    "What did the top sales representatives achieve in Q1 2024?",
    "Write Python code for depth-first search and breadth-first search.",
]

SCOPE_REFUSAL_MESSAGE = (
    "This question doesn't relate to the available business documents "
    "(invoices, contracts, employees, sales data). Please ask a question "
    "about those instead."
)


def run_pipeline(query: str) -> None:
    """Run the full RAG pipeline for a single query and print the result."""

    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    # Stage 1 — Hybrid retrieval (BM25 + ChromaDB, merged with RRF)
    print("\n[1/4] Running hybrid retrieval...")
    candidates = hybrid_search(query, top_k=10)
    print(f"      → {len(candidates)} candidates retrieved")

    # Stage 2 — Cross-encoder reranking → top 5
    print("[2/4] Reranking with cross-encoder...")
    top_chunks = rerank(query, candidates, top_k=5)
    print(f"      → Top {len(top_chunks)} chunks selected")

    # Stage 3 — Scope guard: block off-topic queries before they reach Gemini
    print("[3/4] Checking query scope...")
    in_scope, reason = is_in_scope(query, top_chunks)
    print(f"      → In scope: {in_scope}  ({reason})")

    if not in_scope:
        print("\n" + "-" * 70)
        print("ANSWER:")
        print("-" * 70)
        print(SCOPE_REFUSAL_MESSAGE)
        print("\n(Gemini was not called — blocked by the scope guard.)")
        return

    # Stage 4 — Answer generation with Gemini
    print("[4/4] Generating answer with Gemini...")
    result = generate_answer(query, top_chunks)

    # Print the answer
    print("\n" + "-" * 70)
    print("ANSWER:")
    print("-" * 70)
    print(result["answer"])

    # Print which sources were cited
    print("\nSOURCES CITED:")
    if result["sources_used"]:
        for src in result["sources_used"]:
            print(f"  • {src}")
    else:
        print("  (no sources explicitly cited in answer)")

    # Print the context that was sent to the LLM (optional — useful for debugging)
    print("\n" + "-" * 70)
    print("CONTEXT SENT TO LLM:")
    print("-" * 70)
    # Truncate long contexts to keep terminal output readable
    context_preview = result["context"]
    if len(context_preview) > 1200:
        context_preview = context_preview[:1200] + "\n... [truncated for display]"
    print(context_preview)


def main():
    print("\n" + "=" * 70)
    print("  Business Document QA System — Phase 4: Answer Generation")
    print("=" * 70)
    print(f"Running {len(QUERIES)} queries through the full pipeline.\n")

    for query in QUERIES:
        run_pipeline(query)

    print("\n" + "=" * 70)
    print("Phase 4 complete. Full RAG pipeline is working end to end.")
    print("=" * 70)


if __name__ == "__main__":
    main()
