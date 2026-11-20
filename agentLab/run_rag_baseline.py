import csv
import time
from pathlib import Path
import chromadb

from llm import chat, embed

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "testset.csv"
OUTPUT_CSV = BASE_DIR / "results_rag.csv"
CHROMA_PATH = BASE_DIR / "chroma_local_db"
COLLECTION_NAME = "contract_local"
TOP_K = 3

INSTRUCTION = "Answer only from the text below. If the answer is not there, say NOT FOUND."


def is_correct(row: dict, answer: str) -> bool:
    a = answer.strip().lower()
    expected = row["expected_answer"].strip().lower()
    if row["type"] == "not_in_text":
        return "not found" in a
    return expected in a


def check_page_hit(expected_page: str, retrieved_pages: list[str]) -> bool:
    exp = expected_page.strip()
    if not exp or exp.upper() == "N/A":
        return False
    # Match exact string or any individual page number in a range like "1-2"
    expected_tokens = {p.strip() for p in exp.replace("-", ",").split(",") if p.strip()}
    retrieved_tokens = set()
    for rp in retrieved_pages:
        for tok in str(rp).replace("-", ",").split(","):
            if tok.strip():
                retrieved_tokens.add(tok.strip())
    return bool(expected_tokens & retrieved_tokens)


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)

    rows_out = []
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            status = (row.get("status") or "").strip().lower()
            if status == "skipped":
                rows_out.append({
                    "id": row["id"],
                    "question": row["question"],
                    "expected_answer": row["expected_answer"],
                    "model_answer": "SKIPPED",
                    "correct": "",
                    "retrieved_pages": "",
                    "page_hit": "",
                    "failure_type": "skipped",
                    "source_file": row["source_file"],
                    "page": row["page"],
                    "type": row["type"],
                    "seconds": 0.0,
                })
                continue

            start = time.perf_counter()
            q_vec = embed([f"search_query: {row['question']}"])[0]
            res = collection.query(query_embeddings=[q_vec], n_results=TOP_K)

            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            ret_pages = [str(m.get("page_number", "")) for m in metas]
            page_hit = check_page_hit(row["page"], ret_pages)

            context_text = "\n\n---\n\n".join(docs)
            prompt = f"{INSTRUCTION}\n\nText:\n{context_text}\n\nQuestion: {row['question']}"

            model_answer = chat([{"role": "user", "content": prompt}], temperature=0).strip()
            elapsed = round(time.perf_counter() - start, 2)
            correct = is_correct(row, model_answer)

            if correct:
                failure_type = "none"
            elif not page_hit:
                failure_type = "retrieval_miss"
            else:
                failure_type = "model_failure"

            rows_out.append({
                "id": row["id"],
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "model_answer": model_answer,
                "correct": correct,
                "retrieved_pages": "|".join(ret_pages),
                "page_hit": page_hit,
                "failure_type": failure_type,
                "source_file": row["source_file"],
                "page": row["page"],
                "type": row["type"],
                "seconds": elapsed,
            })

    fieldnames = [
        "id",
        "question",
        "expected_answer",
        "model_answer",
        "correct",
        "retrieved_pages",
        "page_hit",
        "failure_type",
        "source_file",
        "page",
        "type",
        "seconds",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Saved {len(rows_out)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()