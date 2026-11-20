import csv
import json
import time
from pathlib import Path
from llm import chat
from tools import TOOLS

BASE_DIR = Path(__file__).parent
INPUT_CSV = BASE_DIR / "testset.csv"

SYSTEM_PROMPT = """You are a precise assistant with two tools:
1. calculator: evaluates math expressions using only numbers and + - * / ( ).
   JSON format: {"tool": "calculator", "args": {"expression": "80000 * 1.15"}}
2. date_add: adds days to a date (use year 2025 for non-leap years).
   JSON format: {"tool": "date_add", "args": {"start_date": "2025-02-24", "days": 15}}

Reply with ONLY valid JSON. Do not write any text outside the JSON.
Either call a tool:
{"tool": "<tool_name>", "args": {...}}
Or give the final answer:
{"answer": "<final answer>"}"""


def parse_model_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def run_tool_agent(question: str) -> tuple[str, int]:
    """Returns (final_answer, broken_json_count)."""
    broken_json_count = 0
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _step in range(3):
        raw = chat(messages, temperature=0)

        parsed = None
        for attempt in range(2): 
            try:
                parsed = parse_model_json(raw)
                break
            except Exception as exc:
                broken_json_count += 1
                if attempt == 0:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": f"Invalid JSON ({exc}). Reply ONLY with valid JSON matching the schema.",
                    })
                    raw = chat(messages, temperature=0)
                else:
                    return f"ERROR_BROKEN_JSON: {raw}", broken_json_count

        if "answer" in parsed:
            return str(parsed["answer"]).strip(), broken_json_count

        tool_name = parsed.get("tool")
        args = parsed.get("args")

        if tool_name not in TOOLS or not isinstance(args, dict):
            return f"ERROR_INVALID_TOOL: {parsed}", broken_json_count

        try:
            tool_result = TOOLS[tool_name](**args)
        except Exception as exc:
            tool_result = f"TOOL_ERROR: {exc}"

        messages.append({"role": "assistant", "content": json.dumps(parsed)})
        messages.append({
            "role": "user",
            "content": f"Tool '{tool_name}' returned: {tool_result}. Now reply with ONLY {{\"answer\": \"...\"}}.",
        })

    return "ERROR_MAX_STEPS", broken_json_count


def main():
    calc_ids = {"17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27"}
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["id"] in calc_ids]

    total_broken = 0
    for row in rows:
        # 1) Plain baseline (no tools)
        plain_ans = chat([{"role": "user", "content": row["question"]}], temperature=0).strip()

        # 2) With tools
        start = time.perf_counter()
        tool_ans, broken = run_tool_agent(row["question"])
        elapsed = round(time.perf_counter() - start, 2)
        total_broken += broken

        print(f"Row {row['id']} | Expected: {row['expected_answer']}")
        print(f"  Plain : {plain_ans.splitlines()[-1]}")
        print(f"  Tool  : {tool_ans} ({elapsed}s, broken_json={broken})")
        print("-" * 60)

    print(f"Total broken JSON events across run: {total_broken}")


if __name__ == "__main__":
    main()