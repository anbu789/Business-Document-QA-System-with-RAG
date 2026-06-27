"""
evaluation/ragas_eval.py
-------------------------
Phase 5 — RAGAS-style evaluation of the full RAG pipeline.

Implements the four standard RAGAS metrics directly using Gemini, rather
than routing through the ragas library's async executor (which has a
Python 3.14 incompatibility with asyncio.timeout()).

Why implement the metrics ourselves?
  ragas.evaluate() internally asks an LLM judge the same questions we
  implement below. By writing the prompts ourselves we get full control,
  no dependency on ragas's executor internals, and a clearer demonstration
  of understanding each metric at its core — which is what matters for
  interviews.

Metric implementations:
  Faithfulness      — asks Gemini whether any answer claim contradicts context
  Answer Relevancy  — asks Gemini how on-topic the answer is for the question
  Context Precision — asks Gemini how many retrieved chunks are actually useful
  Context Recall    — asks Gemini how much of the ground truth is in the context

Run from project root with venv activated:
  python -m evaluation.ragas_eval
"""

# ── Patch MUST be first — before any ragas import ─────────────────────────────
# ragas 0.2.15 unconditionally imports ChatVertexAI from
# langchain_community.chat_models.vertexai at module load time, even when
# Vertex AI is never used. Stubbing it here satisfies the import.
import sys, types as _types
_stub = _types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _stub
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import json
import time
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

from retrieval.hybrid import hybrid_search
from retrieval.rerank import rerank, load_reranker
from guardrail.scope_check import is_in_scope
from generation.answer import generate_answer
from evaluation.test_set import TEST_CASES

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_NAME  = "gemini-3.1-flash-lite"   # same verified working model from Phase 4
TEST_SLEEP  = 3    # seconds between pipeline runs
SCORE_SLEEP = 5    # seconds between individual metric scoring calls


# ── Gemini client ─────────────────────────────────────────────────────────────

def get_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")
    return genai.Client(api_key=api_key)


def call_gemini(client, prompt: str, max_retries: int = 3) -> str:
    """
    Single Gemini call with retry on quota errors.
    Parses the retryDelay from the 429 response and waits accordingly.
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                # Parse suggested retry delay from error message if present
                delay_match = re.search(r"retry in (\d+)", err, re.IGNORECASE)
                wait = int(delay_match.group(1)) + 5 if delay_match else 45
                print(f"    [quota] 429 hit — waiting {wait}s before retry "
                      f"(attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise   # non-quota errors propagate immediately
    raise RuntimeError(f"Gemini call failed after {max_retries} retries")


def extract_score(text: str) -> float:
    """
    Extract a 0-1 float from a Gemini response.
    Tries JSON first, then regex on bare numbers.
    Returns nan if nothing parseable found.
    """
    # Try JSON block
    try:
        clean = re.sub(r"```(?:json)?|```", "", text).strip()
        data = json.loads(clean)
        if "score" in data:
            return float(data["score"])
    except Exception:
        pass

    # Fallback: find first float/int in 0-1 range
    numbers = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    if numbers:
        return float(numbers[0])

    return float("nan")


# ── Metric implementations ────────────────────────────────────────────────────

def score_faithfulness(client, question: str, answer: str, contexts: list[str]) -> float:
    """
    Faithfulness — does the answer contain only claims supported by the context?

    Method: ask Gemini to identify claims in the answer, then check each
    against the context. Score = supported_claims / total_claims.
    Low score = hallucination detected.
    """
    context_block = "\n\n".join(f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts))
    prompt = f"""You are evaluating whether an answer is faithful to its source context.

Question: {question}

Retrieved context:
{context_block}

Answer to evaluate:
{answer}

Task: Identify each factual claim in the answer. For each claim, check whether
it is directly supported by the retrieved context above.

Return ONLY a JSON object like this:
{{"score": 0.85, "supported": 5, "total": 6, "reason": "one claim about X not in context"}}

Score must be between 0.0 and 1.0 where:
  1.0 = all claims are supported by the context
  0.0 = no claims are supported (complete hallucination)"""

    try:
        response = call_gemini(client, prompt)
        time.sleep(SCORE_SLEEP)
        return extract_score(response)
    except Exception as e:
        print(f"    [faithfulness error] {e}")
        return float("nan")


def score_answer_relevancy(client, question: str, answer: str) -> float:
    """
    Answer Relevancy — is the answer actually addressing the question?

    Method: ask Gemini to rate how directly the answer responds to the question.
    Low score = off-topic, evasive, or incomplete answers.
    Note: this metric does NOT require ground truth.
    """
    prompt = f"""You are evaluating whether an answer is relevant to the question asked.

Question: {question}

Answer: {answer}

Task: Rate how directly and completely the answer addresses the question.
Do not judge factual correctness — only relevance and focus.

A score of 1.0 means the answer directly and completely addresses the question.
A score of 0.0 means the answer is entirely off-topic or refuses to engage.
A grounding refusal ("I could not find this information") still scores around
0.5-0.7 if it is directly responding to the question (just without an answer).

Return ONLY a JSON object like this:
{{"score": 0.9, "reason": "answer directly addresses the payment terms asked"}}

Score must be between 0.0 and 1.0."""

    try:
        response = call_gemini(client, prompt)
        time.sleep(SCORE_SLEEP)
        return extract_score(response)
    except Exception as e:
        print(f"    [answer_relevancy error] {e}")
        return float("nan")


def score_context_precision(client, question: str, contexts: list[str]) -> float:
    """
    Context Precision — are the retrieved chunks actually useful for this question?

    Method: ask Gemini to rate each retrieved chunk on whether it is relevant
    to answering the question. Score = relevant_chunks / total_chunks.
    Low score = retrieval returned noisy/off-topic chunks.
    """
    chunk_list = "\n\n".join(
        f"[Chunk {i+1}]\n{c[:400]}{'...' if len(c) > 400 else ''}"
        for i, c in enumerate(contexts)
    )
    prompt = f"""You are evaluating whether retrieved context chunks are relevant to a question.

Question: {question}

Retrieved chunks:
{chunk_list}

Task: For each chunk, decide if it contains information that would be useful
for answering the question (yes/no). Then compute the proportion of useful chunks.

Return ONLY a JSON object like this:
{{"score": 0.6, "useful_chunks": [1, 3], "total_chunks": 5,
  "reason": "chunks 1 and 3 contain relevant invoice data; others are about employees"}}

Score must be between 0.0 and 1.0 where:
  1.0 = all retrieved chunks are relevant to the question
  0.0 = no retrieved chunks are relevant"""

    try:
        response = call_gemini(client, prompt)
        time.sleep(SCORE_SLEEP)
        return extract_score(response)
    except Exception as e:
        print(f"    [context_precision error] {e}")
        return float("nan")


def score_context_recall(client, question: str, contexts: list[str], ground_truth: str) -> float:
    """
    Context Recall — did retrieval find all the information needed to answer?

    Method: ask Gemini whether the retrieved contexts contain enough information
    to construct the ground truth answer. Score = 1 if all key facts are present,
    0 if critical facts are missing.
    Low score = retrieval missed important chunks.
    """
    context_block = "\n\n".join(f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts))
    prompt = f"""You are evaluating whether retrieved context contains enough information
to construct a reference answer.

Question: {question}

Reference answer (ground truth):
{ground_truth}

Retrieved context:
{context_block}

Task: Identify the key facts/claims in the reference answer. Check whether each
key fact is present in the retrieved context. Score = proportion of key facts
that appear in the retrieved context.

Return ONLY a JSON object like this:
{{"score": 0.8, "found": 4, "total_key_facts": 5,
  "missing": "the late payment penalty rate was not in the retrieved context"}}

Score must be between 0.0 and 1.0 where:
  1.0 = all key facts from the reference answer appear in the retrieved context
  0.0 = no key facts from the reference answer appear in the context"""

    try:
        response = call_gemini(client, prompt)
        time.sleep(SCORE_SLEEP)
        return extract_score(response)
    except Exception as e:
        print(f"    [context_recall error] {e}")
        return float("nan")


# ── Data collection loop ──────────────────────────────────────────────────────

def collect_pipeline_outputs(reranker) -> tuple[list[dict], list[dict]]:
    """
    Run every test case through the real pipeline and collect
    question, answer, contexts, ground_truth for scoring.
    """
    rows         = []
    skipped_rows = []

    print(f"\nCollecting pipeline outputs for {len(TEST_CASES)} test cases...\n")

    for i, case in enumerate(TEST_CASES, 1):
        question     = case["question"]
        ground_truth = case["ground_truth"]
        category     = case["category"]

        print(f"[{i}/{len(TEST_CASES)}] {category.upper()}: {question[:65]}...")

        candidates = hybrid_search(question)
        top_chunks = rerank(question, candidates, top_k=5, reranker=reranker)
        in_scope, reason = is_in_scope(question, top_chunks)

        if not in_scope:
            print(f"  ⚠️  BLOCKED ({reason}) — skipping\n")
            skipped_rows.append({"question": question, "reason": reason, "category": category})
            time.sleep(TEST_SLEEP)
            continue

        result   = generate_answer(question, top_chunks)
        contexts = [chunk["text"] for chunk in top_chunks]

        rows.append({
            "question":     question,
            "answer":       result["answer"],
            "contexts":     contexts,
            "ground_truth": ground_truth,
            "category":     category,
        })

        print(f"  ✅ Collected ({len(result['answer'])} chars, {len(contexts)} contexts)\n")
        time.sleep(TEST_SLEEP)

    return rows, skipped_rows


# ── Score all rows with all metrics ──────────────────────────────────────────

def score_all(rows: list[dict], client) -> list[dict]:
    """
    Run all 4 metrics on every row and attach scores.
    """
    print(f"\nScoring {len(rows)} rows across 4 metrics...\n")

    for i, row in enumerate(rows, 1):
        q  = row["question"]
        a  = row["answer"]
        c  = row["contexts"]
        gt = row["ground_truth"]

        print(f"  [{i}/{len(rows)}] Scoring: {q[:55]}...")

        row["faithfulness"]      = score_faithfulness(client, q, a, c)
        row["answer_relevancy"]  = score_answer_relevancy(client, q, a)
        row["context_precision"] = score_context_precision(client, q, c)
        row["context_recall"]    = score_context_recall(client, q, c, gt)

        print(f"    faith={row['faithfulness']:.2f}  "
              f"relevancy={row['answer_relevancy']:.2f}  "
              f"precision={row['context_precision']:.2f}  "
              f"recall={row['context_recall']:.2f}\n")

    return rows


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(rows: list[dict], skipped_rows: list[dict]) -> None:
    import math

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print("\n" + "=" * 70)
    print("RAGAS-STYLE EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nRows evaluated : {len(rows)}")
    print(f"Rows skipped   : {len(skipped_rows)} (scope guard)")

    print("\n── Aggregate scores (0–1, higher is better) ──\n")
    for m in metrics:
        valid = [r[m] for r in rows if not math.isnan(r[m])]
        mean  = sum(valid) / len(valid) if valid else float("nan")
        print(f"  {m:<25}  {mean:.4f}")

    print("\n── Per-row scores ──\n")
    header = f"  {'Category':<14} {'Question':<42}  faith  relev  prec  recall"
    print(header)
    print("  " + "-" * 75)
    for row in rows:
        q = row["question"][:40]
        print(f"  {row['category']:<14} {q:<42}"
              f"  {row['faithfulness']:>5.2f}"
              f"  {row['answer_relevancy']:>5.2f}"
              f"  {row['context_precision']:>5.2f}"
              f"  {row['context_recall']:>6.2f}")

    if skipped_rows:
        print(f"\n── Skipped by scope guard ──\n")
        for s in skipped_rows:
            print(f"  [{s['category']}] {s['question'][:60]}")

    print("\n── What the scores mean ──\n")
    print("  Faithfulness      — LLM grounding check: are all claims in the context?")
    print("  Answer Relevancy  — is the answer on-topic for the question?")
    print("  Context Precision — are retrieved chunks actually relevant to the query?")
    print("  Context Recall    — did retrieval find all facts needed for the answer?")
    print()
    print("  Note: neg_evidence case (overdue Nexus invoices) is expected to score")
    print("  low on Context Recall — system says 'I could not find this' when the")
    print("  correct answer ('none are overdue') IS in the context. Known limitation.")
    print("=" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  Business Document QA System — Phase 5: RAGAS Evaluation")
    print("=" * 70)

    print("\n[Setup] Loading cross-encoder reranker...")
    reranker = load_reranker()

    print("[Setup] Initialising Gemini client...")
    client = get_client()

    rows, skipped_rows = collect_pipeline_outputs(reranker)

    if not rows:
        print("\n❌ No rows collected.")
        return

    rows = score_all(rows, client)
    print_report(rows, skipped_rows)


if __name__ == "__main__":
    main()
