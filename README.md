

# Legal Assistant RAG API

A multi-tenant Retrieval-Augmented Generation (RAG) pipeline for legal and document analysis built with FastAPI. The system allows users to isolate documents into dynamic database collections and query them using the Gemini API, featuring structured HTTP error handling, timing-attack-safe authentication, session-based conversation memory, automated query rewriting, and precise multi-page citation mapping.

---

## Features

- **Dynamic Collection Isolation:** Uploads files and routes vector search queries into completely isolated database collections dynamically at runtime.
- **Idempotent Storage Updates:** Prevents data duplication by explicitly clearing existing document chunks within a targeted collection before indexing fresh versions of the same file.
- **Stateful Multi-Turn Conversations:** Maintains per-session chat history via sliding windows, enabling seamless context-aware follow-up questions across distinct client sessions.
- **Automated Query Rewriting:** Uses the Gemini model to dynamically translate dependent follow-up questions (using pronouns or contextual references like "that violation") into clean, standalone queries prior to vector retrieval.
- **Multi-Page Span Citation Mapping:** Uses interval overlap analysis to detect chunks crossing page boundaries, storing safe string ranges (e.g., `"1-2"`) to ensure legal users never encounter truncated or missing page citations.
- **Validation Layer:** Rejects invalid media types (non-PDF files) and detects files that yield zero extractable text (such as scanned images missing an OCR layer) before hitting background embedding routines.
- **Production Error Mapping:** Translates database states and third-party API rate limits directly into standard HTTP web responses (`404 Not Found`, `503 Service Unavailable`, `502 Bad Gateway`).
- **Persistent Diagnostic Logging:** Routes system milestones and full multi-line tracebacks concurrently to the terminal console and an internal `backend_errors.log` file for offline auditing.
- **API Key Header Authentication:** Secures all production endpoints using an `X-API-Key` header verified via timing-attack-safe string comparisons (`secrets.compare_digest`).
- **Router-Level Dependency Injection:** Enforces global security automatically across route groups via `APIRouter` dependencies, eliminating manual per-route checks and preventing accidental exposure.

---

## System Architecture

The application enforces a separation of concerns between the transport layer (API routing) and the service layer (vector processing and text extraction).

### Application Flow
1. **Client Request:** The client initiates an API call containing document, session, or query parameters.
2. **FastAPI Layer:** Validates MIME types, verifies incoming payload schemas, coordinates execution blocks, and returns web-standard HTTP status codes.
3. **Service Layer:** Resolves conversation history, rewrites ambiguous follow-up questions into standalone queries, partitions raw text, and executes context queries or updates against ChromaDB collections.
4. **Third-Party Service Layer:** Communicates with the Google GenAI SDK to generate high-density embeddings, perform query rewrites, and synthesize context responses.

---

## Key Engineering Decisions

### 1. Vector Native Engine (ChromaDB vs. PostgreSQL pgvector)
For the current iteration of this system, I selected ChromaDB over a relational engine with vector extensions like PostgreSQL. ChromaDB operates as an embedded vector store, removing external database server configurations and network round-trip overhead for localized data storage. It provides native document collection management out of the box, allowing rapid prototype development without configuring intensive relational schemas or index tuning fields (`HNSW`/`IVFFlat`) manually.

### 2. Multi-Tenant Collection Isolation (Dynamic vs. Single Base Collection)
Instead of streaming all document segments into a single global vector store and relying on meta-filtering parameters, the architecture dynamically creates and isolates data into distinct ChromaDB collections at runtime. This prevents context cross-contamination during semantic searches and ensures that queries executed within one document domain have no possibility of returning unrelated text data from another client file partition.

### 3. Idempotent Ingestion Pipelines (Delete-on-Index Pattern)
To ensure the embedding system remains clean and predictable across repeated uploads of the same file, the pipeline uses a delete-before-write process. Before parsing and chunking an incoming PDF file, the service explicitly drops any vector records tagged with that specific filename inside the target collection. This approach guarantees idempotency, preventing stale chunk fragmentation and duplicate text weights from corrupting context lookups.

### 4. Transport-Layer Exception Consolidation (Endpoint vs. Service Level Error Handling)
I intentionally designed the service layer functions to throw standard Python exceptions (such as `ValueError`) rather than embedding FastAPI-specific `HTTPException` dependencies directly within database operations. This adheres to the separation of concerns principle. It leaves the core RAG service completely framework-agnostic, while consolidating all web-transport translations, payload logging, and HTTP status code tracking inside the router endpoints.

### 5. API Key Security & Timing-Attack Mitigation
To protect endpoints from unauthorized resource exhaustion and data scraping without introducing heavy user-session state, the API relies on header-based token authentication (`X-API-Key`). Key verification uses Python's `secrets.compare_digest()`. This guarantees constant-time execution, preventing malicious actors from performing side-channel timing attacks to guess the secret key character-by-character, while gracefully handling Unicode inputs.

### 6. Multi-Page Span Metadata Mapping via Interval Overlap
Standard text chunkers suffer from "single-page blindness"—if a text chunk crosses a page break (e.g., starting on Page 1 and ending on Page 2), saving only the starting page breaks legal citation integrity. The chunking service evaluates interval intersections (`pb["start"] < end and pb["end"] > start`) across all page boundaries, capturing every page touched by the text window.

### 7. ChromaDB Scalar-Safe Page Range Encoding
Because vector databases like ChromaDB strictly enforce scalar types (`str`, `int`, `float`, `bool`) in metadata dictionaries and reject raw Python lists, multi-page spans are dynamically formatted into clean, searchable string ranges (e.g., `"1"` for single-page chunks or `"1-2"` for cross-page spans). This satisfies vector database schema rules while keeping citation strings completely readable by downstream answering modules.

### 8. Stateful Sliding-Window Memory & Standalone Query Rewriting
To support natural follow-up conversations without polluting context or breaking vector search caches, the system tracks conversation history via isolated, session-keyed sliding windows (`ConversationMemory`). When a user asks a context-dependent question, the rewriter uses Gemini to expand pronouns and dependent phrases into a standalone query. Cache keys and vector lookups are bound exclusively to this rewritten standalone query, ensuring high semantic accuracy and preventing cross-user session leaks.

---

## Known Limitations & Engineering Trade-offs

- **Absence of Cryptographic Access Control:** The current system lacks user authentication and role-based validation. All generated collections are publicly accessible via the API endpoints if the valid `X-API-Key` header is provided.
- **Lack of Optical Character Recognition (OCR):** Document parsing depends entirely on standard structural text extraction strings. PDFs consisting of raw scanned images yield zero characters, prompting a termination guard rather than falling back to an OCR conversion step.
- **Single-Node Local Storage Model:** ChromaDB runs in embedded memory mode with file persistence bound to the host filesystem. This layout cannot scale horizontally across multiple container nodes or distributed cloud environments without transitioning to an independent, client-server database deployment.
- **In-Memory Session Volatility:** Conversation memory is stored ephemerally in server RAM using Python `defaultdict`. Restarting the backend process clears active session histories.
- **Static API Key Model:** Authentication relies on a single shared administrative API key (`X-API-Key`) loaded from environment variables rather than a multi-tenant user database.

---

## Core Technology Stack

- **Framework:** FastAPI (Asynchronous ASGI setup)
- **Vector Database:** ChromaDB (Local storage persistence)
- **AI Engine & Embeddings:** Client-based Google GenAI SDK (`gemini-embedding-2`, `gemini-2.5-flash`)
- **Environment Management:** Anaconda / Python Virtual Environments

---

## Project Structure

```text
AiProject/
├── main.py                # FastAPI routers, payload schemas, and HTTP error handling
├── try_chunks.py          # Standalone test suite for chunking boundaries and multi-page validation
├── backend_errors.log     # Persistent internal server error tracebacks (auto-generated)
├── requirements.txt       # Project package dependencies
└── services/
    ├── __init__.py
    ├── memory.py          # Session-based conversation memory & query rewriter
    ├── pdf_processor.py   # Text extraction, interval overlap mapping, and document parsing logic
    └── vector_store.py    # ChromaDB collection manipulation and embedding pipeline
    └── search_engine.py   # RAG retrieval and question-answering logic

```

---

## Installation & Setup

### 1. Activate Environment

Ensure your virtual environment is configured and active before running the application:

```bash
conda activate your_env

```

### 2. Configure Your API Environment Variable

The modern Google GenAI SDK automatically looks for the `GEMINI_API_KEY` environment variable. Set it in your terminal session before launching the backend server:

```DOS
set GEMINI_API_KEY=your_actual_api_key_here

```

```PowerShell
$env:GEMINI_API_KEY="your_actual_api_key_here"

```

---

## API Endpoints Reference

### 1. Upload and Index Document

* **Endpoint:** `POST /upload/`
* **Security:** Requires `X-API-Key` header.
* **Query Parameters:** `collection_name` (string)
* **Payload:** `Multipart/Form-Data` containing a single PDF file.

**Validation Guards:**

* Returns `401 Unauthorized` if the `X-API-Key` header is missing or incorrect.
* Returns `415 Unsupported Media Type` if the file type is not explicitly `application/pdf`.
* Returns `400 Bad Request` if the PDF contains no extractable text string (e.g., blank or raw scanned images without OCR).
* Returns `500 Internal Server Error` if internal embedding processes or vector database writes fail.

### 2. Query Document Context (Multi-Turn Stateful RAG)

* **Endpoint:** `POST /ask/`
* **Security:** Requires `X-API-Key` header.
* **JSON Request Body Schema:**

```json
{
  "question": "What are the penalties specified for that violation?",
  "collection_name": "corporate_contracts_2026",
  "filename": "employment_agreement.pdf",
  "session_id": "dispute_session_1"
}

```

**Validation Guards & Response Rules:**

* Returns `401 Unauthorized` if authentication fails.
* Returns `404 Not Found` if the targeted collection does not exist on disk.
* Returns `503 Service Unavailable` if your Gemini API free-tier daily quota (`ClientError`) is exhausted.
* **Negative RAG Handling (`found: false`):** When semantic search or model synthesis determines the document does not contain the answer, the pipeline returns an empty sources array or `found: false` status to prevent false legal citations and avoids poisoning conversation memory.

---

## Local Development & Testing

### Running the Verification Suites

To execute the chunking and multi-page validation test suite offline, run:

```bash
python try_chunks.py

```

To execute the memory window and session isolation unit tests, run:

```bash
python -m unittest test.py

```

### Launching the Development Server

To launch the local Uvicorn development server with real-time reload tracking active, run:

```bash
uvicorn main:app --reload

```

Once the application initializes, access the interactive documentation page directly at:

`http://127.0.0.1:8000/docs`

```

```