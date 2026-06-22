"""
guardrail/calibrate_threshold.py

Empirically determine UPPER_THRESHOLD and LOWER_THRESHOLD for the scope guard
by running known-relevant and known-irrelevant queries through the existing
Phase 3 retrieval + rerank pipeline and inspecting the top cross-encoder score.

Why this exists: cross-encoder scores are raw, uncalibrated logits whose range
depends on the specific model + corpus. Rather than guessing threshold numbers,
we look at where relevant vs. irrelevant queries actually land and set the
boundaries from real data.

Run from project root (after ingestion has been run):
    python -m guardrail.calibrate_threshold
"""

from retrieval.hybrid import hybrid_search
from retrieval.rerank import rerank, load_reranker

# Known-relevant queries — same ones validated in Phase 3 / Phase 4, PLUS
# additional naturally-phrased questions covering sales and employee data,
# added after "What did the top sales representatives achieve in Q1 2024?"
# was incorrectly blocked in the first calibration pass. The original set
# skewed heavily toward the Nexus contract/invoice (narrative PDF text) and
# under-tested the CSV-formatted chunks (sales report, employees), which the
# cross-encoder appears to score lower even when topically correct.
RELEVANT_QUERIES = [
    "Which invoices from Nexus Solutions are overdue?",
    "Who signed the contract with Nexus Solutions and what is their role?",
    "What did Michael Torres sell in Q1 2024?",
    "What is the total contract value with Nexus Solutions and what are the payment terms?",
    "What did the top sales representatives achieve in Q1 2024?",
    "Which employees work in the Engineering department?",
    "What was the total revenue in the sales report?",
]

# Known-irrelevant queries — nothing in the business documents relates to these.
# These SHOULD score low. Mix of code requests, general knowledge, and creative asks
# to cover the range of off-topic queries a real user might try.
IRRELEVANT_QUERIES = [
    "Write Python code for depth-first search and breadth-first search.",
    "What is the capital of France?",
    "Write a haiku about the ocean.",
    "Explain how a binary search tree works.",
    "What's the weather like today?",
]


def get_top_score(query: str, reranker, top_k_rerank: int = 5) -> float:
    """
    Run a query through hybrid search + rerank and return the cross-encoder
    score of the single best-ranked chunk — this is the signal the scope
    guard will threshold against.

    Uses hybrid_search()'s own defaults (dense_k=10, sparse_k=10, merged
    top_k=20) so calibration sees the same candidate pool the real pipeline
    will use at query time. Takes a pre-loaded reranker so the ~85MB model
    is loaded once for the whole script, not once per query.
    """
    candidates = hybrid_search(query)
    reranked = rerank(query, candidates, top_k=top_k_rerank, reranker=reranker)

    if not reranked:
        # No candidates returned at all — treat as maximally irrelevant
        return float("-inf")

    return reranked[0]["rerank_score"]


def run_calibration():
    print("=" * 70)
    print("SCOPE GUARD CALIBRATION")
    print("=" * 70)

    print("\n[Setup] Loading cross-encoder once for the whole run...")
    reranker = load_reranker()

    print("\n--- RELEVANT queries (should score HIGH) ---\n")
    relevant_scores = []
    for q in RELEVANT_QUERIES:
        score = get_top_score(q, reranker)
        relevant_scores.append(score)
        print(f"  {score:+.4f}   {q}")

    print("\n--- IRRELEVANT queries (should score LOW) ---\n")
    irrelevant_scores = []
    for q in IRRELEVANT_QUERIES:
        score = get_top_score(q, reranker)
        irrelevant_scores.append(score)
        print(f"  {score:+.4f}   {q}")

    relevant_floor = min(relevant_scores)
    relevant_ceiling = max(relevant_scores)
    irrelevant_floor = min(irrelevant_scores)
    irrelevant_ceiling = max(irrelevant_scores)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Relevant queries   -> min: {relevant_floor:+.4f}   max: {relevant_ceiling:+.4f}")
    print(f"Irrelevant queries -> min: {irrelevant_floor:+.4f}   max: {irrelevant_ceiling:+.4f}")

    if irrelevant_ceiling < relevant_floor:
        gap = relevant_floor - irrelevant_ceiling
        upper = irrelevant_ceiling + gap * 0.66
        lower = irrelevant_ceiling + gap * 0.33

        print(f"\n✅ Clean separation found. Gap = {gap:.4f}")
        print("\nSuggested thresholds for guardrail/scope_check.py:")
        print(f"  UPPER_THRESHOLD = {upper:.4f}   # score >= this  -> confidently RELEVANT")
        print(f"  LOWER_THRESHOLD = {lower:.4f}   # score <= this  -> confidently IRRELEVANT")
        print(f"  Borderline zone: {lower:.4f} to {upper:.4f}  -> LLM classifier fallback")
    else:
        overlap = relevant_floor - irrelevant_ceiling
        print(f"\n⚠️  WARNING: score ranges OVERLAP by {abs(overlap):.4f}.")
        print("   No clean line separates relevant from irrelevant queries.")
        print("   Options:")
        print("   - Add more relevant/irrelevant test queries to tighten the picture")
        print("   - Widen the borderline (LLM fallback) zone to cover the overlap")
        print("   - Inspect the per-query scores above and set thresholds by hand")

    print("\nNext step: copy the threshold values into guardrail/scope_check.py")
    print("=" * 70)


if __name__ == "__main__":
    run_calibration()
