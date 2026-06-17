"""
Run this script to ingest all documents in the data/ folder.
Usage: python -m ingestion.run_ingestion
"""
from pathlib import Path
from ingestion.extract import extract
from ingestion.chunk import chunk_document
from ingestion.index import index_chunks


DATA_DIR = "data"


def run():
    data_path = Path(DATA_DIR)
    files = list(data_path.glob("*.pdf")) + list(data_path.glob("*.csv"))

    if not files:
        print(f"No PDF or CSV files found in {DATA_DIR}/")
        return

    all_chunks = []
    for filepath in files:
        print(f"\nProcessing: {filepath.name}")
        doc = extract(str(filepath))
        chunks = chunk_document(doc)
        print(f"  → {len(chunks)} chunks created")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks to index: {len(all_chunks)}")
    index_chunks(all_chunks)
    print("\n Ingestion complete.")


if __name__ == "__main__":
    run()