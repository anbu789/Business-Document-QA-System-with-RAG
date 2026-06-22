"""
guardrail/scope_check.py
-------------------------
Scope guard — decides whether a query is in-scope (about the business
documents) BEFORE generation runs, so off-topic questions never reach
the full Gemini answer call.

Hybrid approach, thresholds calibrated empirically in
guardrail/calibrate_threshold.py against this project's actual corpus:

    rerank_score >= UPPER_THRESHOLD  -> confidently RELEVANT   -> skip LLM, proceed
    rerank_score <= LOWER_THRESHOLD  -> confidently IRRELEVANT -> block, no LLM call
    otherwise (borderline)           -> ask Gemini a focused yes/no question

Calibration data (12 queries across 2 runs — see calibrate_threshold.py):
    relevant queries:    -7.51 to +3.89  (PDF narrative scores high, CSV rows score lower)
    irrelevant queries:  -11.35 to -11.19 (tight cluster, consistent across all 5 queries)
    gap:                 3.68 (relevant floor -7.51 vs irrelevant ceiling -11.19)

Key finding: abstract phrasing against CSV-formatted chunks scores lower
than specific phrasing against PDF narrative text, even when topically
correct. UPPER_THRESHOLD = -7.0 ensures abstract CSV-backed queries go to
the LLM fallback rather than scraping past the score gate on thin margins.
"""

import os
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────

MODEL_NAME = "gemini-3.1-flash-lite"   # same model verified working in Phase 4

UPPER_THRESHOLD = -7.0    # score >= -7.0  → pass directly (all calibrated relevant queries)
LOWER_THRESHOLD = -10.0   # score <= -10.0 → block directly (irrelevant cluster at -11.2 to -11.4)

# Calibration history (see guardrail/calibrate_threshold.py):
#   Pass 1 (4 relevant queries, all PDF/contract-narrative): clean 12-point gap.
#   Pass 2 (+3 CSV-sourced queries: sales reps, employees, sales revenue):
#     revealed CSV row-dump chunks score consistently lower than narrative
#     PDF text even when topically correct (-7.51 to -5.24 vs -0.68 to +3.89).
#     LOWER_THRESHOLD widened from -7.0 to -10.0 so these land in the LLM
#     fallback zone instead of being auto-blocked. Irrelevant cluster stayed
#     a tight -11.35 to -11.19 across both passes — safety margin preserved.

CLASSIFIER_PROMPT = """You are a strict scope classifier for a business document Q&A system.

The system only answers questions using these documents: invoices, service contracts, employee records, and sales reports.

Given the user's question below, respond with EXACTLY one word:
"yes" if the question could plausibly be answered from invoices, contracts, employee records, or sales reports.
"no" if it is unrelated — general knowledge, coding help, creative writing, small talk, or anything not about these business documents.

Question: {query}

Answer (yes or no only):"""


# ── LLM fallback classifier (borderline cases only) ─────────────────────────

def _llm_classify(query: str) -> bool:
    """
    Ask Gemini a focused yes/no question to resolve a borderline scope case.
    Only called when the retrieval score gate can't decide on its own.

    Fails OPEN: if the API call errors (quota, network, etc.), the query is
    treated as in-scope rather than blocked. Rationale: a transient API
    failure shouldn't silently deny a legitimate user, and the strict
    grounding prompt in generate_answer() is still there as a backstop —
    worst case, Gemini correctly says "I could not find this information."
    If you'd rather fail closed (block on classifier error), flip the
    return value in the except block below.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment. Check your .env file.")

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=CLASSIFIER_PROMPT.format(query=query),
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=5,   # only need "yes" or "no"
            ),
        )
        verdict = response.text.strip().lower()
        return verdict.startswith("yes")
    except Exception as e:
        print(f"[ScopeGuard] Classifier call failed ({e}) — failing open (treating as in-scope).")
        return True


# ── Public API ────────────────────────────────────────────────────────────

def is_in_scope(query: str, reranked_chunks: list[dict]) -> tuple[bool, str]:
    """
    Decide whether a query is in-scope for the business document QA system.

    Args:
        query            : the user's original question
        reranked_chunks  : output of retrieval.rerank.rerank() — each dict
                            must have a 'rerank_score' key (Phase 3 format)

    Returns:
        (in_scope, reason) — reason is a short string useful for logging
        and for Phase 5 RAGAS test cases that check the guard fired correctly.
    """
    if not reranked_chunks:
        return False, "no_candidates"

    top_score = reranked_chunks[0]["rerank_score"]

    if top_score >= UPPER_THRESHOLD:
        return True, f"score_relevant ({top_score:.4f} >= {UPPER_THRESHOLD})"

    if top_score <= LOWER_THRESHOLD:
        return False, f"score_irrelevant ({top_score:.4f} <= {LOWER_THRESHOLD})"

    # Borderline — the free signal couldn't decide, spend one LLM call
    print(f"[ScopeGuard] Borderline score ({top_score:.4f}) — asking Gemini to classify...")
    in_scope = _llm_classify(query)
    reason = "llm_classified_relevant" if in_scope else "llm_classified_irrelevant"
    return in_scope, reason


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    from retrieval.hybrid import hybrid_search
    from retrieval.rerank import rerank, load_reranker

    test_queries = [
        "Which invoices from Nexus Solutions are overdue?",        # should pass (score gate)
        "Write Python code for depth-first search and breadth-first search.",  # should block (score gate)
    ]

    reranker = load_reranker()

    for q in test_queries:
        candidates = hybrid_search(q)
        reranked = rerank(q, candidates, top_k=5, reranker=reranker)
        in_scope, reason = is_in_scope(q, reranked)
        print(f"\nQuery: {q}")
        print(f"  In scope: {in_scope}  ({reason})")
