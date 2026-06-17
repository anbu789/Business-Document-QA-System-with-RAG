import csv
import pdfplumber
from pathlib import Path


def extract_pdf(filepath: str) -> str:
    """Extract text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_csv(filepath: str) -> str:
    """Convert CSV rows into readable text blocks."""
    lines = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Turn each row into "key: value, key: value" format
            row_text = ", ".join(f"{k}: {v}" for k, v in row.items())
            lines.append(row_text)
    return "\n".join(lines)


def extract(filepath: str) -> dict:
    """
    Auto-detect file type and extract text.
    Returns: { "text": str, "source_type": str, "filename": str }
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == ".pdf":
        text = extract_pdf(filepath)
        source_type = "pdf"
    elif ext == ".csv":
        text = extract_csv(filepath)
        source_type = "csv"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return {
        "text": text,
        "source_type": source_type,
        "filename": path.name,
    }