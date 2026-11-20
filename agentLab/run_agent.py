import csv
import json
import re
import time
from pathlib import Path
from llm import chat
from tools import TOOLS

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "testset.csv"
OUTPUT_CSV = BASE_DIR / "results_agent.csv"
TRACES_FILE = BASE_DIR / "traces.jsonl"
MAX_STEPS = 5

SYSTEM_PROMPT = """You are a legal and math assistant. You have 4 tools and must reply ONLY with valid JSON.

TOOLS:
1. search_docs: Search indexed legal documents for relevant passages.
   Format: {"tool": "search_docs", "args": {"query": "<search_terms>"}}
2. get_page: Read the full text of a specific page when a search snippet is incomplete.
   Format: {"tool": "get_page", "args": {"source_file": "<file.pdf>", "page": <page_number>}}
3. calculator: Evaluate arithmetic using numbers and + - * / ( ) only. Only use when the question asks for a calculation.
   Format: {"tool": "calculator", "args": {"expression": "<math_expression>"}}
4. date_add: Add a whole integer number of days to a date. Only use when the question asks to add days to a date.
   Format: {"tool": "date_add", "args": {"start_date": "YYYY-MM-DD", "days": <integer_days>}}

RULES:
- Questions about the contract or document content -> call search_docs first.
- Questions needing arithmetic or percentages -> call calculator using numbers from the question.
- Questions adding days to a date -> call date_add using a date and integer days from the question.
- Never copy placeholder tokens or invent numbers/dates not in the question or tool results.
- If search_docs finds no relevant answer in the document -> reply with NOT FOUND.
- When ready to give the final answer, include the cited files and pages (or [] if no document was used).

EXAMPLES:
Tool call:
{"tool": "search_docs", "args": {"query": "<search_terms>"}}

Final answer with document citation:
{"answer": "<concise_answer>", "sources": [{"file": "<file.pdf>", "page": "1"}]}

Final answer without document (math/date or not found):
{"answer": "NOT FOUND", "sources": []}"""

STOPWORDS = {
    "the", "a", "an", "of", "in", "and", "or", "to", "for",
    "is", "are", "what", "which", "how", "on", "with", "by", "from",
}
PAGE_RE = re.compile(r"^\d+(?:-\d+)?$")


def query_tokens(q: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (q or "").lower())
    return {w for w in words if w not in STOPWORDS}


def is_near_duplicate_search(cur_q: str, past_queries: list[set[str]]) -> bool:
    if len(past_queries) >= 3:
        return True
    cur = query_tokens(cur_q)
    if not cur:
        return False
    for prev in past_queries:
        if not prev:
            continue
        sim = len(cur & prev) / len(cur | prev)
        if sim >= 0.6:
            return True
    return False


def clean_sources(raw) -> list[dict]:
    if not isinstance(raw, list):
        return []
    cleaned = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        file_val = str(item.get("file") or "").strip()
        page_val = item.get("page")
        if not file_val or file_val.lower() in {"[]", "null", "none"}:
            continue
        if "tool" in file_val.lower() or "args" in file_val.lower():
            continue
        if isinstance(page_val, int) and not isinstance(page_val, bool):
            page_str = str(page_val)
        elif isinstance(page_val, str) and PAGE_RE.fullmatch(page_val.strip()):
            page_str = page_val.strip()
        else:
            continue
        key = (file_val, page_str)
        if key not in seen:
            seen.add(key)
            cleaned.append({"file": file_val, "page": page_str})
    return cleaned


def parse_json_obj(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("JSON root must be an object")
    return obj


def is_correct(row: dict, answer: str) -> bool:
    a = answer.strip().lower()
    expected = row["expected_answer"].strip().lower()
    if row["type"] == "not_in_text":
        return "not found" in a
    return expected in a


def call_tool_safely(tool_name: str, args: dict, question: str) -> dict:
    fn = TOOLS[tool_name]
    if tool_name in {"calculator", "date_add"}:
        try:
            return fn(question=question, **args)
        except TypeError:
            return fn(**args)
    return fn(**args)


def run_agent_question(question: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps_trace = []
    broken_json_count = 0
    seen_exact_calls: set[tuple[str, str]] = set()
    past_search_queries: list[set[str]] = []

    for step_num in range(1, MAX_STEPS + 1):
        raw = chat(messages, temperature=0)
        parsed = None

        for attempt in range(2):
            try:
                parsed = parse_json_obj(raw)
                break
            except Exception as exc:
                broken_json_count += 1
                if attempt == 0:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": f"Invalid JSON ({exc}). Reply ONLY with valid JSON.",
                    })
                    raw = chat(messages, temperature=0)
                else:
                    return {
                        "answer": "ERROR_BROKEN_JSON",
                        "sources": [],
                        "steps": steps_trace,
                        "num_steps": step_num,
                        "broken_json": broken_json_count,
                    }

        if "answer" in parsed:
            return {
                "answer": str(parsed["answer"]).strip(),
                "sources": clean_sources(parsed.get("sources", [])),
                "steps": steps_trace,
                "num_steps": step_num,
                "broken_json": broken_json_count,
            }

        tool_name = parsed.get("tool")
        args = parsed.get("args")

        if tool_name not in TOOLS:
            tool_result = {"error": f"Unknown tool '{tool_name}'. Allowed: {list(TOOLS.keys())}"}
        elif not isinstance(args, dict):
            tool_result = {"error": "'args' must be a JSON object."}
        else:
            sig = (tool_name, json.dumps(args, sort_keys=True, ensure_ascii=False))
            if sig in seen_exact_calls:
                tool_result = {
                    "error": "Duplicate tool call blocked. Answer now from existing tool results or reply NOT FOUND."
                }
            elif tool_name == "search_docs" and is_near_duplicate_search(
                str(args.get("query", "")), past_search_queries
            ):
                tool_result = {
                    "error": "Too many or too similar search_docs calls. Answer now from retrieved results or reply NOT FOUND."
                }
            else:
                seen_exact_calls.add(sig)
                if tool_name == "search_docs":
                    past_search_queries.append(query_tokens(str(args.get("query", ""))))
                try:
                    tool_result = call_tool_safely(tool_name, args, question)
                except Exception as exc:
                    tool_result = {"error": str(exc)}

        steps_trace.append({
            "step": step_num,
            "tool": tool_name,
            "args": args,
            "result": tool_result,
        })

        messages.append({"role": "assistant", "content": json.dumps(parsed)})
        messages.append({
            "role": "user",
            "content": f"Tool result: {json.dumps(tool_result)}",
        })

    return {
        "answer": "ERROR_MAX_STEPS",
        "sources": [],
        "steps": steps_trace,
        "num_steps": MAX_STEPS,
        "broken_json": broken_json_count,
    }


def main():
    rows_out = []
    if TRACES_FILE.exists():
        TRACES_FILE.unlink()

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            status = (row.get("status") or "").strip().lower()
            if status == "skipped":
                rows_out.append({
                    "id": row["id"],
                    "question": row["question"],
                    "expected_answer": row["expected_answer"],
                    "model_answer": "SKIPPED",
                    "sources": "",
                    "auto_correct": "",
                    "num_steps": 0,
                    "broken_json": 0,
                    "type": row["type"],
                    "seconds": 0.0,
                })
                continue

            start = time.perf_counter()
            outcome = run_agent_question(row["question"])
            elapsed = round(time.perf_counter() - start, 2)

            trace_record = {
                "id": row["id"],
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "steps": outcome["steps"],
                "num_steps": outcome["num_steps"],
                "broken_json": outcome["broken_json"],
                "final_answer": outcome["answer"],
                "sources": outcome["sources"],
            }
            with TRACES_FILE.open("a", encoding="utf-8") as tf:
                tf.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

            rows_out.append({
                "id": row["id"],
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "model_answer": outcome["answer"],
                "sources": json.dumps(outcome["sources"], ensure_ascii=False),
                "auto_correct": is_correct(row, outcome["answer"]),
                "num_steps": outcome["num_steps"],
                "broken_json": outcome["broken_json"],
                "type": row["type"],
                "seconds": elapsed,
            })

    fieldnames = [
        "id",
        "question",
        "expected_answer",
        "model_answer",
        "sources",
        "auto_correct",
        "num_steps",
        "broken_json",
        "type",
        "seconds",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    active = [r for r in rows_out if r["model_answer"] != "SKIPPED"]
    avg_steps = round(sum(r["num_steps"] for r in active) / len(active), 2)
    auto_hits = sum(1 for r in active if r["auto_correct"] is True)
    print(f"Saved {len(rows_out)} rows to {OUTPUT_CSV} and traces to {TRACES_FILE}")
    print(f"Auto correct: {auto_hits}/{len(active)} | Average steps: {avg_steps}")


if __name__ == "__main__":
    main()