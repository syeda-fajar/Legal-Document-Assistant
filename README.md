# Legal Assistant RAG API + Local Agent Lab

This project has two parts:

1. **RAG API (working):** A FastAPI service that answers questions about uploaded PDFs, with page citations. It uses ChromaDB for search and the Gemini API for embeddings and answers.
2. **Agent Lab (experiment, measured):** A tool-using agent that I wrote myself. It runs on a small local model (Llama 3.2 3B via Ollama) and uses parts of the RAG API as tools. I measured it against the fixed RAG pipeline on a hand-checked test set. The agent is **not yet part of the API**.

The goal of this project is to learn how LLM agents work, measure them honestly, and build them as a real backend feature step by step.

---

## Status

| Part | Status |
|---|---|
| `POST /upload/`, `POST /ask/` (Gemini) | Working |
| Local model and local embeddings (Ollama) | Working, used in Agent Lab |
| Test set and results | Done for one small document (see Results) |
| Agent loop with 4 tools and saved traces | Working, weaker than fixed RAG on document questions |
| `POST /agent/ask/` in the API | Planned (next step) |
| Switch between Gemini and local model | Planned |
| Planning, answer checking, tests, Docker | Planned |

---

## Features (RAG API)

- **Dynamic Collection Isolation:** Uploads files and routes search queries into separate database collections at runtime.
- **Idempotent Storage Updates:** Deletes existing chunks of a file in a collection before indexing a new version, so there are no duplicates.
- **Stateful Multi-Turn Conversations:** Keeps per-session chat history with a sliding window for follow-up questions.
- **Automated Query Rewriting:** Uses the model to turn follow-up questions ("that violation") into standalone queries before search.
- **Multi-Page Span Citation Mapping:** Detects chunks that cross page boundaries and stores page ranges as strings (for example `"1-2"`), so citations are never cut.
- **Validation Layer:** Rejects non-PDF files and PDFs with no extractable text (scanned images without OCR).
- **HTTP Error Mapping:** Maps database states and API rate limits to `404`, `503` and `502`.
- **Persistent Logging:** Writes milestones and full tracebacks to the console and to `backend_errors.log`.
- **API Key Authentication:** `X-API-Key` header checked with `secrets.compare_digest`.
- **Router-Level Dependencies:** Security is enforced for a whole route group, so no route is left open by mistake.

---

## Agent Lab (experiment)

**Idea:** In the fixed RAG pipeline, my code decides every step (rewrite, search once, answer). In an agent, the model decides what to do next: search, read a page, calculate, or answer. My code checks every request and runs the tools. The model never runs anything itself.

**How it works:**

1. The model gets the question and a system prompt that lists the tools.
2. It replies with JSON: either a tool call or a final answer with sources.
3. My code validates the JSON and the arguments, runs the tool, and sends the result back to the model.
4. This repeats until the model answers or the step limit (5) is reached.
5. Every run is saved as one line in `traces.jsonl` (question, each tool call, result, final answer).

**Tools**

| Tool | What it does |
|---|---|
| `search_docs(query)` | Searches the indexed document and returns chunks with file and page |
| `get_page(source_file, page)` | Returns the full clean text of one page (from a page map made at indexing time) |
| `calculator(expression)` | Safe arithmetic with `ast` (numbers and `+ - * / ( )` only) |
| `date_add(start_date, days)` | Adds days to a `YYYY-MM-DD` date |

**Guards in code:** JSON validation with one retry, step limit, blocking of repeated identical tool calls, argument checks (for example `days` must be a whole number), and cleaning of the `sources` field.

---

## Experiments and Results

**Setup**

- Model: `llama3.2:3b` via Ollama, temperature 0
- Embeddings: `nomic-embed-text` (local), stored in a separate ChromaDB collection (`contract_local`, 8 chunks, top 3 chunks used)
- Document: one 2-page legal notice template (`contract.pdf`)
- Test set: `testset.csv` with 27 questions (26 used, 1 skipped because it needs two documents)
  - 7 in the text, 2 across two places, 5 not in the text, 12 calculation or date questions
- All answers were checked **by hand**. One run per question.

**Results**

| | Fixed RAG | Agent v1 | Agent v2 |
|---|---|---|---|
| All 26 questions | 17 | 17 (+1 partly correct) | 15 |
| Calculation / date (12) | 4 | 12 | 10 |
| Document questions (14) | 13 | 5 | 5 |
| Not in the text (5) | 5 | 1 | 0 |
| Average steps | 1 search | 3.0 | 2.5 |

Agent v2 added prompt changes and code guards to fix problems found in v1. Several things changed at once, so I cannot say which change caused which result.

**What I found**

- Tools fixed most calculation and date errors. The plain model often got dates wrong (for example, it forgot that some days were already used up in the month).
- The 3B agent was **worse** than fixed RAG on document questions.
- In v1 the model copied the example arguments from my system prompt (junk tool calls), repeated the same search until the step limit, and wrote answers that were not in the document.
- Prompt changes and code guards reduced the number of steps, but did not reduce invented answers.
- For example, for "What is the exact date the original contract was signed?" (not in the document), one version answered "January 31, 1970."

**Retrieval check:** the correct page was in the top 3 for 9 of 9 questions that have a page. This number is weak, because the document has only 2 pages.

**Limits of this test:** one small document, which is a template with placeholders such as `PKR [amount]`. 26 questions, one run each. No fine-tuning. Question 2 is ambiguous because of the template. These results do not say how other models or other documents behave.

**Related work:** Shen et al., *Small LLMs Are Weak Tool Learners: A Multi-LLM Agent* (EMNLP 2024), also reports weak tool use in small models and uses separate planner, caller and summarizer roles. My experiment is much smaller and uses prompts only.

---

## System Architecture

The app separates the transport layer (API routes) from the service layer (vector search and text extraction).

### RAG API flow

1. **Client request** with document, session or query parameters.
2. **FastAPI layer** validates file types and payloads and returns HTTP status codes.
3. **Service layer** resolves history, rewrites follow-up questions, chunks text, and queries ChromaDB.
4. **Gemini layer** (Google GenAI SDK) creates embeddings, rewrites queries and writes answers.

### Agent Lab flow

```text
question
   |
   v
 model (Llama 3.2 3B, Ollama)
   |  returns JSON: tool call or final answer
   v
 my code: validate -> run tool -> add result to the conversation
   |                 (search_docs, get_page, calculator, date_add)
   v
 repeat (max 5 steps) -> answer + sources -> trace saved
```

---

## Key Engineering Decisions

### 1. ChromaDB instead of PostgreSQL pgvector
ChromaDB is an embedded vector store. It needs no separate database server and gives collection management out of the box, which is good for a fast prototype. I did not need to set up schemas or index settings (`HNSW`/`IVFFlat`).

### 2. One collection per document set
Each document set gets its own ChromaDB collection instead of one big collection with metadata filters. A query in one collection cannot return text from another.

### 3. Delete-before-write indexing
Before indexing a PDF, the service deletes old records for that filename in the collection. Re-uploading the same file does not create duplicates.

### 4. Service layer raises normal Python exceptions
Service functions raise exceptions like `ValueError`. Only the route functions turn them into `HTTPException`. This keeps the RAG service independent from FastAPI.

### 5. API key check with `secrets.compare_digest()`
The check takes constant time, so the key cannot be guessed with timing attacks.

### 6. Page spans with interval overlap
A chunk can cross a page break. The chunker checks interval overlap (`pb["start"] < end and pb["end"] > start`) against every page and keeps all pages it touches.

### 7. Page ranges stored as strings
ChromaDB metadata accepts only scalar values, not lists. Page spans are stored as strings like `"1"` or `"1-2"`.

### 8. Session memory and standalone query rewriting
Each session has its own sliding window (`ConversationMemory`). Follow-up questions are rewritten into standalone queries, and search uses only the rewritten query.

### 9. Agent experiments run on a small local model
I use a 3B model through Ollama. It runs on my own laptop (8 GB RAM, no paid API), I can repeat tests as often as I want, and a small model makes the failures easy to see. The result is not a statement about larger models.

### 10. Agent loop written by hand, no framework
I wrote the loop myself so that I understand what a framework would hide: JSON parsing, validation, step limits and traces.

### 11. One collection per embedding model
Vectors from different embedding models cannot be mixed. The local experiments use their own collection (`contract_local`) with `nomic-embed-text`, separate from the Gemini collections.

### 12. Page map for `get_page`
The chunks overlap and some are marked `"1-2"`, so joining chunks gives repeated text. At indexing time I save the clean text of each page in `pages_store.json` (file name -> page -> text). `get_page` reads from this map.

---

## Known Limitations

- **No user accounts:** one shared `X-API-Key`. Anyone with the key can reach every collection.
- **No OCR:** scanned PDFs have no text, so they are rejected.
- **Single-node storage:** ChromaDB runs embedded with files on local disk. It cannot scale across several servers.
- **Memory is in RAM:** session history is lost when the server restarts.
- **Agent is weaker than fixed RAG on document questions** with a 3B model, and it still makes up answers when the document has none. No answer-checking step exists yet.
- **Small test set:** one 2-page template document, 26 questions, one run each, hand-checked. The "two documents" question type is not tested yet.
- **Weak auto-scoring:** my script's exact-match score is lower than the real score, so the table above uses hand grading.
- **Agent is not in the API yet.** The model switch (Gemini or local) is also not done.

---

## Technology Stack

- **API:** FastAPI (ASGI)
- **Vector database:** ChromaDB (local files)
- **RAG API models:** Google GenAI SDK (`gemini-embedding-2`, `gemini-2.5-flash`)
- **Agent Lab models:** Ollama with `llama3.2:3b` and `nomic-embed-text`
- **Environment:** Anaconda / Python virtual environments

---

## Project Structure

```text
AiProject/
├── main.py                # FastAPI routes, payload schemas, HTTP error handling
├── try_chunks.py          # Test script for chunk boundaries and multi-page spans
├── test.py                # Unit tests for session memory and isolation
├── backend_errors.log     # Error tracebacks (auto-generated)
├── requirements.txt       # Dependencies
├── services/
│   ├── __init__.py
│   ├── memory.py          # Session memory and query rewriter
│   ├── pdf_processor.py   # Text extraction, chunking, page overlap mapping
│   ├── vector_store.py    # ChromaDB collections and embedding pipeline
│   └── search_engine.py   # Retrieval and question answering
└── agentLab/              # Agent experiments (does not change the API)
    ├── llm.py             # chat() and embed() for local Ollama models
    ├── index_local.py     # Indexes a PDF with local embeddings and builds the page map
    ├── tools.py           # search_docs, get_page, calculator, date_add
    ├── tool_loop.py       # First tool-calling loop (calculator, date)
    ├── run_agent.py       # Agent loop with 4 tools, step limit and guards
    ├── run_tests.py       # Fixed RAG baseline on the test set
    ├── testset.csv        # Test questions with expected answers
    ├── results_rag.csv    # Fixed RAG results
    ├── traces.jsonl       # Saved agent runs
    └── pages_store.json   # Clean text per page (made at indexing)
```

---

## Installation and Setup

### 1. Activate the environment

```bash
conda activate your_env
```

### 2. Gemini key (for the RAG API)

```DOS
set GEMINI_API_KEY=your_actual_api_key_here
```

```PowerShell
$env:GEMINI_API_KEY="your_actual_api_key_here"
```

### 3. Local models (for Agent Lab)

Install Ollama, then download the models while you have internet:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

---

## API Endpoints

### 1. Upload and index a document

- **Endpoint:** `POST /upload/`
- **Security:** `X-API-Key` header
- **Query parameter:** `collection_name`
- **Payload:** `multipart/form-data` with one PDF

Responses: `401` (bad key), `415` (not a PDF), `400` (no extractable text), `500` (embedding or database error).

### 2. Ask a question (multi-turn RAG)

- **Endpoint:** `POST /ask/`
- **Security:** `X-API-Key` header

```json
{
  "question": "What are the penalties specified for that violation?",
  "collection_name": "corporate_contracts_2026",
  "filename": "employment_agreement.pdf",
  "session_id": "dispute_session_1"
}
```

Responses: `401` (bad key), `404` (collection not found), `503` (Gemini free-tier quota used up). When the answer is not in the document, the API returns `found: false` or an empty sources list, so it does not give false citations and does not add the failed answer to the session memory.

### 3. Agent question (planned, not implemented yet)

`POST /agent/ask/` will run the agent loop and return the answer, the sources, the number of steps and a trace id. It will use the same `X-API-Key` check and the same error handling.

---

## Local Development and Testing

Chunking and multi-page tests:

```bash
python try_chunks.py
```

Memory and session tests:

```bash
python -m unittest test.py
```

Agent Lab (Ollama must be running):

```bash
cd agentLab
python index_local.py     # index the PDF with local embeddings
python run_tests.py       # fixed RAG baseline on the test set
python run_agent.py       # agent run, writes traces.jsonl
```

Start the API:

```bash
uvicorn main:app --reload
```

Open the interactive docs at `http://127.0.0.1:8000/docs`.

---

## Roadmap

1. `POST /agent/ask/` with schemas, `X-API-Key`, error handling and trace logging.
2. Switch between Gemini and local model with environment variables.
3. Planning step for multi-part questions, compared on the same test set.
4. Check each claim against the retrieved text before answering (to reduce invented answers).
5. More documents in the test set, including questions that need two files.
6. A few automated tests, a Dockerfile and settings from environment variables.
