import sys
from io import BytesIO
from pathlib import Path
import chromadb

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm import embed
from services.pdf_processor import extract_text_from_pdf, chunk_text_with_metadata


PDF_PATH = PROJECT_ROOT / "contract.pdf"  
CHROMA_PATH = BASE_DIR / "chroma_local_db"
COLLECTION_NAME = "contract_local"
import json
from pypdf import PdfReader

reader = PdfReader(str(PDF_PATH))
pages_store = {}
for page_num, page_obj in enumerate(reader.pages, start=1):
    pages_store[f"{PDF_PATH.name}::{page_num}"] = (page_obj.extract_text() or "").strip()

PAGES_STORE_PATH = BASE_DIR / "pages_store.json"
with PAGES_STORE_PATH.open("w", encoding="utf-8") as pf:
    json.dump(pages_store, pf, ensure_ascii=False, indent=2)

def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

    with PDF_PATH.open("rb") as f:
        pdf_bytes = BytesIO(f.read())

    full_text, page_boundaries = extract_text_from_pdf(pdf_bytes)
    chunks = chunk_text_with_metadata(
        full_text,
        page_boundaries,
        filename=PDF_PATH.name,
    )

    if not chunks:
        raise ValueError("No chunks produced from PDF.")

    raw_docs = [c["document"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"contract_p{m['page_number']}_{i}" for i, m in enumerate(metadatas)]

  
    prefixed_docs = [f"search_document: {doc}" for doc in raw_docs]
    vectors = embed(prefixed_docs)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
  
    existing = [col.name for col in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)
    collection.add(
        ids=ids,
        documents=raw_docs,
        metadatas=metadatas,
        embeddings=vectors,
    )

    print(f"Indexed {len(raw_docs)} chunks into '{COLLECTION_NAME}' at {CHROMA_PATH}")


if __name__ == "__main__":
    main()