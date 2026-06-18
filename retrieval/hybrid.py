"""
retrieval/hybrid.py
-------------------
Hybrid retrieval: queries ChromaDB (dense) + BM25 (sparse),
then merges results using Reciprocal Rank Fusion (RRF).

Returns a ranked list of candidate chunks ready for reranking.
"""

import pickle
import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────────────────────

CHROMA_PATH      = "chroma_store"
COLLECTION_NAME  = "documents"
BM25_PATH        = "bm25_index.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# RRF smoothing constant — standard value from the original RRF paper (Cormack 2009)
# Higher k reduces the impact of top-ranked documents; 60 is the industry default
RRF_K = 60


# ── Index loaders ──────────────────────────────────────────────────────────────

def load_chroma_collection(chroma_path: str = CHROMA_PATH,
                            collection_name: str = COLLECTION_NAME):
    """Load the ChromaDB persistent collection."""
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_collection(collection_name)


def load_bm25_index(bm25_path: str = BM25_PATH):
    """
    Load the BM25 index from disk.
    Returns: (bm25_model, texts_list, metadatas_list)
    """
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["texts"], data["metadatas"]


def load_embedding_model(model_name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    """Load the sentence-transformers embedding model."""
    return SentenceTransformer(model_name)


# ── Search functions ───────────────────────────────────────────────────────────

def dense_search(query: str,
                 collection,
                 embed_model: SentenceTransformer,
                 top_k: int = 10) -> list[dict]:
    """
    Query ChromaDB using a dense vector embedding of the query.

    ChromaDB returns distances (lower = more similar). We convert to a
    similarity score (1 - distance) for consistent direction with BM25.

    Returns list of dicts: { id, text, metadata, score }
    """
    query_vector = embed_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    candidates = []
    for i in range(len(results["ids"][0])):
        candidates.append({
            "id":       results["ids"][0][i],
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score":    1 - results["distances"][0][i],   # convert distance → similarity
        })

    return candidates  # already sorted best-first by ChromaDB


def sparse_search(query: str,
                  bm25_model,
                  texts: list[str],
                  metadatas: list[dict],
                  top_k: int = 10) -> list[dict]:
    """
    Query the BM25 index using tokenized query words.

    BM25 scores based on term frequency × inverse document frequency.
    Higher score = better keyword match.

    Returns list of dicts: { id, text, metadata, score }, sorted best-first.
    """
    tokenized_query = query.lower().split()
    scores = bm25_model.get_scores(tokenized_query)

    # Pair each chunk with its BM25 score and sort descending
    scored = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )

    candidates = []
    for idx, score in scored[:top_k]:
        candidates.append({
            "id":       f"bm25_{idx}",   # BM25 has no built-in IDs; use index position
            "text":     texts[idx],
            "metadata": metadatas[idx],
            "score":    float(score),
        })

    return candidates


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def reciprocal_rank_fusion(dense_results: list[dict],
                           sparse_results: list[dict],
                           k: int = RRF_K) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    RRF formula for each chunk:
        rrf_score = Σ  1 / (k + rank)
    where the sum is over each list the chunk appears in.

    Why RRF instead of score normalization?
    - BM25 and cosine similarity live on completely different scales.
    - Normalizing across different distributions is fragile and sensitive to outliers.
    - RRF only uses rank position, which is scale-invariant and robust.
    - Used in production by Elasticsearch 8.x, Weaviate, and Pinecone.

    Chunks are matched across lists by their text content (since BM25 uses
    positional IDs and ChromaDB uses UUID-style IDs).

    Returns deduplicated list sorted by rrf_score descending.
    """
    rrf_scores = {}   # text → running rrf score
    chunk_data  = {}  # text → { text, metadata } (to reconstruct output)

    # Score dense results
    for rank, item in enumerate(dense_results):
        text = item["text"]
        rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (k + rank + 1)
        chunk_data[text] = {"text": item["text"], "metadata": item["metadata"]}

    # Score sparse results (adds to existing score if chunk already seen)
    for rank, item in enumerate(sparse_results):
        text = item["text"]
        rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (k + rank + 1)
        chunk_data[text] = {"text": item["text"], "metadata": item["metadata"]}

    # Build final sorted list
    merged = []
    for text, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        merged.append({
            "text":      chunk_data[text]["text"],
            "metadata":  chunk_data[text]["metadata"],
            "rrf_score": rrf_score,
        })

    return merged


# ── Public API ─────────────────────────────────────────────────────────────────

def hybrid_search(query: str,
                  top_k: int = 20,
                  dense_k: int = 10,
                  sparse_k: int = 10) -> list[dict]:
    """
    Full hybrid search pipeline:
        1. Dense search  → top dense_k results from ChromaDB
        2. Sparse search → top sparse_k results from BM25
        3. RRF merge     → single ranked list, deduplicated

    Args:
        query:    User's natural language question
        top_k:    Max candidates to return after merging (passed to reranker)
        dense_k:  How many results to fetch from ChromaDB
        sparse_k: How many results to fetch from BM25

    Returns:
        List of dicts: { text, metadata, rrf_score }, sorted best-first
    """
    print(f"\n[Hybrid] Query: '{query}'")

    # Load indexes and model
    embed_model = load_embedding_model()
    collection  = load_chroma_collection()
    bm25, texts, metadatas = load_bm25_index()

    # Run both searches
    dense_results  = dense_search(query, collection, embed_model, top_k=dense_k)
    sparse_results = sparse_search(query, bm25, texts, metadatas, top_k=sparse_k)

    print(f"[Hybrid] Dense results:  {len(dense_results)}")
    print(f"[Hybrid] Sparse results: {len(sparse_results)}")

    # Merge with RRF
    merged = reciprocal_rank_fusion(dense_results, sparse_results)
    merged = merged[:top_k]

    print(f"[Hybrid] Merged candidates after RRF: {len(merged)}")

    return merged


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = hybrid_search("Which invoices from Nexus Solutions are overdue?")
    print("\n── Top candidates ──")
    for i, r in enumerate(results, 1):
        src = r["metadata"].get("filename", "unknown")
        print(f"\n[{i}] RRF={r['rrf_score']:.4f}  source={src}")
        print(f"     {r['text'][:200].strip()} ...")
