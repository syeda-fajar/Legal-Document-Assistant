import ast
import json
import operator
from datetime import datetime, timedelta
from pathlib import Path
import chromadb
from llm import embed

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_local_db"
COLLECTION_NAME = "contract_local"
PAGES_STORE_PATH = BASE_DIR / "pages_store.json"

ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

import re

PLACEHOLDER_RE = re.compile(r"(<[^>]+>|YYYY-MM-DD)", re.IGNORECASE)

def _has_placeholder(val) -> bool:
    if isinstance(val, str):
        return bool(PLACEHOLDER_RE.search(val))
    if isinstance(val, dict):
        return any(_has_placeholder(v) for v in val.values())
    if isinstance(val, list):
        return any(_has_placeholder(v) for v in val)
    return False

def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("Division by zero")
        return ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARYOPS:
        return ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Disallowed expression element: {type(node).__name__}")


def calculator(expression: str) -> dict:
    if not isinstance(expression, str) or not expression.strip():
        return {"error": "Expression must be a non-empty string"}
    try:
        clean = expression.replace(",", "").strip()
        tree = ast.parse(clean, mode="eval")
        result = _eval_node(tree)
        val = str(int(result)) if isinstance(result, float) and result.is_integer() else str(round(result, 6))
        return {"result": val}
    except Exception as exc:
        return {"error": str(exc)}


def date_add(start_date: str, days: int) -> dict:
    try:
        days_int = int(days)
        dt = datetime.strptime(str(start_date).strip(), "%Y-%m-%d")
        new_dt = dt + timedelta(days=days_int)
        return {
            "iso_date": new_dt.strftime("%Y-%m-%d"),
            "formatted_date": new_dt.strftime("%B %d").replace(" 0", " "),
        }
    except Exception as exc:
        return {"error": str(exc)}


def search_docs(query: str) -> dict:
    if not isinstance(query, str) or not query.strip():
        return {"results": [], "message": "Query must be a non-empty string."}
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    q_vec = embed([f"search_query: {query.strip()}"])[0]
    res = collection.query(query_embeddings=[q_vec], n_results=3)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    if not docs:
        return {"results": [], "message": "No matching passages found."}

    hits = []
    for doc, meta in zip(docs, metas):
        hits.append({
            "file": meta.get("source_filename", "contract.pdf"),
            "page": meta.get("page_number", "1"),
            "text": doc,
        })
    return {"results": hits}


def get_page(source_file: str, page: int) -> dict:
    if not PAGES_STORE_PATH.exists():
        return {"error": "pages_store.json not found. Run index_local.py first."}
    try:
        page_int = int(page)
    except Exception:
        return {"error": "page must be an integer"}

    with PAGES_STORE_PATH.open("r", encoding="utf-8") as f:
        pages_store = json.load(f)

    key = f"{str(source_file).strip()}::{page_int}"
    if key not in pages_store:
        return {"error": f"Page {page_int} of '{source_file}' not found."}
    return {"file": str(source_file).strip(), "page": page_int, "text": pages_store[key]}


TOOLS = {
    "search_docs": search_docs,
    "get_page": get_page,
    "calculator": calculator,
    "date_add": date_add,
}
