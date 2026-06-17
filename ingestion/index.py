import os
import pickle
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # suppresses another common warning

# Paths
CHROMA_DIR = "chroma_store"
BM25_PATH = "bm25_index.pkl"

# Load embedding model once at module level (avoid reloading on every call)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB client + collection
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name="documents")


def index_chunks(chunks: list[dict]) -> None:
    """
    Embed chunks and store in ChromaDB. Also builds/updates BM25 index.

    Args:
        chunks: List of { "text": str, "metadata": dict } from chunk_document()
    """
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Generate unique IDs: filename + chunk_index
    ids = [
        f"{m['filename']}_chunk_{m['chunk_index']}"
        for m in metadatas
    ]

    # Embed all chunks in one batch (faster than one-by-one)
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    # Store in ChromaDB
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"[ChromaDB] Indexed {len(chunks)} chunks.")

    # Build BM25 index from tokenized texts
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)

    # Save BM25 index + raw texts to disk (needed for retrieval)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "texts": texts, "metadatas": metadatas}, f)
    print(f"[BM25] Index saved to {BM25_PATH}.")


def load_bm25() -> tuple:
    """Load BM25 index from disk. Returns (bm25, texts, metadatas)."""
    if not os.path.exists(BM25_PATH):
        raise FileNotFoundError("BM25 index not found. Run ingestion first.")
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["texts"], data["metadatas"]