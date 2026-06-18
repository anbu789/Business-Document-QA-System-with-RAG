"""
retrieval/run_retrieval.py
--------------------------
End-to-end test runner for the Phase 3 retrieval pipeline.

Runs several sample queries that exercise different retrieval strengths:
  - Semantic queries  → dense search does the heavy lifting
  - Keyword queries   → BM25 does the heavy lifting
  - Cross-document    → requires chunks from multiple files to surface together

Run from the project root (venv activated):
    python -m retrieval.run_retrieval

Expected output: top-5 reranked chunks per query, with source filename and score.
"""

from retrieval.hybrid import hybrid_search
from retrieval.rerank  import retrieve_and_rerank

# ── Test queries ───────────────────────────────────────────────────────────────

TEST_QUERIES = [
    # Keyword-heavy — BM25 should surface invoice chunks; cross-encoder confirms
    "Which invoices from Nexus Solutions are overdue?",

    # Semantic — dense search should find payment terms from the PDF
    "What are the payment terms in the service contract?",

    # Cross-document — answer requires both employees.csv and sales_report CSV
    "What did Michael Torres sell in Q1 2024?",

    # Exact entity — tests BM25's exact-match strength
    "Who is James Okafor and what is his role?",
]

TOP_K_HYBRID  = 20   # candidates fetched from hybrid search (before reranking)
TOP_K_FINAL   = 5    # chunks returned to LLM after reranking


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_pipeline(query: str) -> list[dict]:
    """Run hybrid search → rerank for a single query. Returns top-5 chunks."""

    # Stage 1: Hybrid search (BM25 + dense → RRF merge)
    candidates = hybrid_search(query, top_k=TOP_K_HYBRID)

    # Stage 2: Cross-encoder reranking
    final_chunks = retrieve_and_rerank(query, candidates, top_k=TOP_K_FINAL)

    return final_chunks


def print_results(query: str, results: list[dict]) -> None:
    """Pretty-print the final reranked chunks."""
    divider = "─" * 70
    print(f"\n{divider}")
    print(f"QUERY: {query}")
    print(divider)

    if not results:
        print("  No results returned.")
        return

    for i, chunk in enumerate(results, 1):
        filename   = chunk["metadata"].get("filename",    "unknown")
        src_type   = chunk["metadata"].get("source_type", "unknown")
        chunk_idx  = chunk["metadata"].get("chunk_index",  "?")
        score      = chunk["rerank_score"]
        preview    = chunk["text"][:300].replace("\n", " ").strip()

        print(f"\n  [{i}] Score: {score:+.4f}")
        print(f"       Source: {filename}  (type={src_type}, chunk={chunk_idx})")
        print(f"       Text:   {preview} ...")

    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Phase 3 — Hybrid Retrieval Pipeline Test")
    print("=" * 70)
    print(f"\n  Config: hybrid top_k={TOP_K_HYBRID}, rerank top_k={TOP_K_FINAL}")
    print(f"  Queries: {len(TEST_QUERIES)}\n")

    for query in TEST_QUERIES:
        results = run_pipeline(query)
        print_results(query, results)

    print("=" * 70)
    print("  Phase 3 retrieval pipeline test complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
