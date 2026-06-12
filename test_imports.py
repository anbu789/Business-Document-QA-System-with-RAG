# add this to test_imports.py
from dotenv import load_dotenv
import os

load_dotenv()

print("Gemini key loaded:", bool(os.getenv("GOOGLE_API_KEY")))
print("HuggingFace token loaded:", bool(os.getenv("HUGGINGFACE_TOKEN")))