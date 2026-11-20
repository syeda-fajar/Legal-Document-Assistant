import csv
import time
from pathlib import Path
from llm import chat

BASE_DIR = Path(__file__).parent
INPUT_CSV = BASE_DIR / "testset.csv"

with INPUT_CSV.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row["id"] in {"18", "19", "20"}:
            start = time.perf_counter()
            ans = chat([{"role": "user", "content": row["question"]}], temperature=0).strip()
            elapsed = round(time.perf_counter() - start, 2)
            print(f"Row {row['id']} ({elapsed}s) | Expected: {row['expected_answer']}")
            print(f"Model: {ans}\n{'-'*50}")