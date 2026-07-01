"""
api/main.py
-----------
FastAPI backend for the Business Document QA System.

Wraps the 4-stage RAG pipeline — hybrid retrieval -> cross-encoder rerank ->
scope guard -> grounded generation — behind a single POST /query endpoint.

Model loading strategy:
    All heavy resources (embedding model, cross-encoder, ChromaDB collection,
    BM25 index) are loaded ONCE at startup via the lifespan context manager
    and cached on app.state. They are NOT reloaded per request.

    This matters because the CLI runner scripts (run_generation.py,
    run_retrieval.py) reload everything from scratch on every run — fine for
    a one-off script, but if an API handler did the same thing, every single
    HTTP request would pay the full model-load cost (disk read + weights
    into memory) before doing any actual work. Loading once at process
    startup and reusing across requests is the standard pattern for serving
    ML models behind an API.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from retrieval.hybrid import (
    hybrid_search,
    load_embedding_model,
    load_chroma_collection,
    load_bm25_index,
)
from retrieval.rerank import rerank, load_reranker
from guardrail.scope_check import is_in_scope
from generation.answer import generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")


# ── Request / response schemas ───────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's natural language question")


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources_used: list[str]
    in_scope: bool
    scope_reason: str
    rerank_score: Optional[float] = None


# ── Lifespan: load all models/indexes ONCE at process startup ───────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models and indexes (embedding model, cross-encoder, "
                "ChromaDB, BM25)... this happens once at startup.")

    app.state.embed_model = load_embedding_model()
    app.state.collection = load_chroma_collection()
    bm25, texts, metadatas = load_bm25_index()
    app.state.bm25_data = (bm25, texts, metadatas)
    app.state.reranker = load_reranker()

    logger.info("Startup complete — all resources cached on app.state.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Business Document QA System",
    description="RAG-powered Q&A over invoices, contracts, employee records, "
                 "and sales reports.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Rate limiting ────────────────────────────────────────────────────────
# Protects the Gemini free-tier quota from being drained by a single
# runaway client (script, retry loop, or malicious traffic). Limits are
# per-IP. Adjust QUERY_RATE_LIMIT via env var if you need it looser/tighter
# for a specific deployment.

QUERY_RATE_LIMIT = os.getenv("QUERY_RATE_LIMIT", "10/minute")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────────────────
# Streamlit (localhost:8501) and FastAPI (localhost:8000) are different
# origins even when running on the same machine — browsers block
# cross-origin requests by default, so the API must explicitly allow the
# frontend's origin. Configurable via env var for when the frontend URL
# changes (e.g. after deployment).

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — used by deployment platforms and for a quick manual check."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
@limiter.limit(QUERY_RATE_LIMIT)
def query(request: Request, body: QueryRequest):
    """
    Run the full 4-stage RAG pipeline on a user question:
        1. hybrid_search() — BM25 + ChromaDB + RRF merge
        2. rerank()        — cross-encoder, top-5 chunks
        3. is_in_scope()   — score gate + LLM fallback
        4. generate_answer() — grounded Gemini answer (only if in scope)

    Rate limited per-IP (see QUERY_RATE_LIMIT) to protect the Gemini
    free-tier quota from a single runaway or malicious client.
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Stage 1 — hybrid retrieval, using cached embed_model/collection/bm25_data
        candidates = hybrid_search(
            question,
            embed_model=app.state.embed_model,
            collection=app.state.collection,
            bm25_data=app.state.bm25_data,
        )

        if not candidates:
            return QueryResponse(
                query=question,
                answer="I could not find this information in the provided documents.",
                sources_used=[],
                in_scope=False,
                scope_reason="no_candidates",
                rerank_score=None,
            )

        # Stage 2 — cross-encoder reranking, using cached reranker
        reranked = rerank(question, candidates, top_k=5, reranker=app.state.reranker)

        top_score = reranked[0]["rerank_score"]

        # Stage 3 — scope guard (score gate + LLM fallback for borderline cases)
        in_scope, reason = is_in_scope(question, reranked)

        if not in_scope:
            return QueryResponse(
                query=question,
                answer=(
                    "This question doesn't appear to be about the business "
                    "documents (invoices, contracts, employee records, or "
                    "sales reports). Please ask something related to those "
                    "documents."
                ),
                sources_used=[],
                in_scope=False,
                scope_reason=reason,
                rerank_score=top_score,
            )

        # Stage 4 — grounded answer generation (Gemini call)
        result = generate_answer(question, reranked)

        return QueryResponse(
            query=question,
            answer=result["answer"],
            sources_used=result["sources_used"],
            in_scope=True,
            scope_reason=reason,
            rerank_score=top_score,
        )

    except HTTPException:
        raise
    except Exception:
        # Covers Gemini quota errors, network issues, or any unexpected bug.
        # The full exception is logged server-side for debugging, but NOT
        # included in the response — error details (stack traces, internal
        # config hints) should never reach the client. The client gets a
        # clean, generic message instead.
        logger.exception("Error handling /query")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable. Please try again shortly.",
        )
