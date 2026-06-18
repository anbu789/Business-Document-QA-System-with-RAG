"""
retrieval/rerank.py
-------------------
Cross-encoder reranking stage.

Takes the hybrid search shortlist (~20 candidates) and scores each
(query, chunk) pair jointly using a cross-encoder model. Returns the
top-K chunks with precise relevance scores for the LLM.

Why cross-encoders instead of just using bi-encoder scores?
- Bi-encoders (used in hybrid.py) encode query and document separately.
  Fast, but approximate — they can't model word-level interactions.
- Cross-encoders read both query and document together in one forward pass.
  Slower, but far more accurate — they see exactly how query terms relate
  to document terms.
- Solution: bi-encoder for high-recall retrieval (fast, run over all chunks),
  cross-encoder for high-precision reranking (slow, run only on shortlist).
  This is the exact architecture used by Google, Bing, and most production
  search systems.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on MS MARCO (Microsoft's large-scale passage ranking dataset,
  built from real Bing search queries).
- Outputs a single relevance score per (query, passage) pair.
- Higher score = more relevant. No fixed scale — relative ordering matters.
"""

from sentence_transformers.cross_encoder import CrossEncoder

# ── Configuration ──────────────────────────────────────────────────────────────

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K     = 5


# ── Model loader ───────────────────────────────────────────────────────────────

def load_reranker(model_name: str = RERANK_MODEL_NAME) -> CrossEncoder:
    """
    Load the cross-encoder model.
    Downloads from HuggingFace on first run (~85MB), then cached locally.
    """
    return CrossEncoder(model_name)


# ── Reranking ──────────────────────────────────────────────────────────────────

def rerank(query: str,
           candidates: list[dict],
           top_k: int = DEFAULT_TOP_K,
           reranker: CrossEncoder = None) -> list[dict]:
    """
    Score each (query, candidate_text) pair with the cross-encoder,
    then return the top_k candidates sorted by score descending.

    Args:
        query:      The user's original question
        candidates: List of dicts from hybrid_search — each has 'text' and 'metadata'
        top_k:      How many reranked results to return (sent to LLM in Phase 4)
        reranker:   Optional pre-loaded CrossEncoder (avoids reloading model in loops)

    Returns:
        List of dicts: { text, metadata, rerank_score }, sorted best-first
    """
    if not candidates:
        return []

    # Load model if not passed in
    if reranker is None:
        print("[Rerank] Loading cross-encoder model...")
        reranker = load_reranker()

    # Build (query, passage) pairs — this is the input format CrossEncoder expects
    pairs = [(query, candidate["text"]) for candidate in candidates]

    print(f"[Rerank] Scoring {len(pairs)} candidates with cross-encoder...")

    # Score all pairs in one batch — returns a list of floats
    scores = reranker.predict(pairs)

    # Attach scores to candidates
    scored_candidates = []
    for candidate, score in zip(candidates, scores):
        scored_candidates.append({
            "text":         candidate["text"],
            "metadata":     candidate["metadata"],
            "rerank_score": float(score),
        })

    # Sort by cross-encoder score, best first
    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Return top_k
    top_results = scored_candidates[:top_k]

    print(f"[Rerank] Returning top {len(top_results)} chunks.")

    return top_results


# ── Public convenience function ────────────────────────────────────────────────

def retrieve_and_rerank(query: str,
                        candidates: list[dict],
                        top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Convenience wrapper: loads the cross-encoder and reranks in one call.
    Use this from run_retrieval.py and later from generation/answer.py.

    Args:
        query:      User's question
        candidates: Output of hybrid_search()
        top_k:      Final number of chunks to return to the LLM

    Returns:
        Top-k reranked chunks as list of dicts: { text, metadata, rerank_score }
    """
    reranker = load_reranker()
    return rerank(query, candidates, top_k=top_k, reranker=reranker)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test with dummy candidates
    dummy_candidates = [
        {
            "text": "Invoice INV-2024-004 for Nexus Solutions, amount $32,000, status: Overdue.",
            "metadata": {"filename": "invoices.csv", "source_type": "csv"},
        },
        {
            "text": "Employee James Okafor, Engineering Manager, joined 2021.",
            "metadata": {"filename": "employees.csv", "source_type": "csv"},
        },
        {
            "text": "Service contract with Nexus Solutions. Payment terms: Net-30. Total value $120,000.",
            "metadata": {"filename": "service_contract_nexus_solutions.pdf", "source_type": "pdf"},
        },
    ]

    query = "Which invoices from Nexus Solutions are overdue?"
    results = retrieve_and_rerank(query, dummy_candidates, top_k=2)

    print("\n── Reranked results ──")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score={r['rerank_score']:.4f}  source={r['metadata']['filename']}")
        print(f"     {r['text'][:200]}")
