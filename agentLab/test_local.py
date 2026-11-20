import csv
import time
from pathlib import Path
from llm import chat

BASE_DIR = Path(__file__).parent
INPUT_CSV = BASE_DIR / "testset.csv"
OUTPUT_CSV = BASE_DIR / "results.csv"


SAMPLE_CONTEXT = """
Law Firm: Pk-Legal and Associates.
Subject: Legal Notice for Breach of Contract.
1. Background: On [Date of Contract], our client and you entered into a legally binding agreement regarding [brief description of the agreement, e.g., supply of goods/services, real estate transaction, partnership, etc.].
2. Breach of Contract: Your failure to comply with the terms of the contract constitutes a material breach, causing our client significant loss and damages. Despite our client's repeated attempts to communicate with you and resolve the issue amicably, you have failed to take any action to rectify the breach or compensate our client for the losses incurred.
3. Relief Demanded: We urge you to take immediate action to fulfill your contractual obligations within [number of days, typically 7-15 days] from the date of receipt of this notice. Failing to comply will compel our client to initiate appropriate legal proceedings against you, including but not limited to filing a suit for breach of contract, seeking damages, and any other relief deemed appropriate under Pakistani law. You will be held responsible for all costs incurred, including legal fees and court charges.
4. Consequences of Non-Compliance: If you fail to comply with this notice within the stipulated time, we will proceed to file a lawsuit in the competent court of law without any further notice. You will bear the full responsibility for all legal costs and damages resulting from this breach.
Important Notes: 1. Proof of Delivery: Ensure this legal notice is sent through a reliable method that provides proof of delivery, such as registered mail or courier. 2. Record Keeping: Keep copies of all related documents, including the contract, communications, and the proof of delivery, for future reference in legal proceedings.
"""

INSTRUCTION = 'Answer only from the text below. If the answer is not there, say NOT FOUND.'


def is_correct(row: dict, answer: str) -> bool:
    a = answer.strip().lower()
    expected = row["expected_answer"].strip().lower()
    if row["type"] == "not_in_text":
        return "not found" in a
    return expected in a


def main():
    rows_out = []
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("status") or "").strip().lower()
            if status == "skipped":
                rows_out.append({
                    "id": row["id"],
                    "question": row["question"],
                    "expected_answer": row["expected_answer"],
                    "model_answer": "SKIPPED",
                    "correct": "",
                    "source_file": row["source_file"],
                    "page": row["page"],
                    "type": row["type"],
                    "seconds": 0.0,
                })
                continue

            prompt = f"{INSTRUCTION}\n\nText:\n{SAMPLE_CONTEXT}\n\nQuestion: {row['question']}"
            start = time.perf_counter()
            model_answer = chat([{"role": "user", "content": prompt}], temperature=0).strip()
            elapsed = round(time.perf_counter() - start, 2)

            rows_out.append({
                "id": row["id"],
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "model_answer": model_answer,
                "correct": is_correct(row, model_answer),
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