# Legal Assistant RAG API

A multi-tenant Retrieval-Augmented Generation (RAG) pipeline for legal and document analysis built with FastAPI. The system allows users to isolate documents into dynamic database collections and query them using the Gemini API, featuring structured HTTP error handling and validation layers.

---

## Features

* **Dynamic Collection Isolation:** Uploads files and routes vector search queries into completely isolated database collections dynamically at runtime.
* **Idempotent Storage Updates:** Prevents data duplication by explicitly clearing existing document chunks within a targeted collection before indexing fresh versions of the same file.
* **Validation Layer:** Rejects invalid media types (non-PDF files) and detects files that yield zero extractable text (such as scanned images missing an OCR layer) before hitting background embedding routines.
* **Production Error Mapping:** Translates database states and third-party API rate limits directly into standard HTTP web responses (`404 Not Found`, `503 Service Unavailable`, `502 Bad Gateway`).
* **Persistent Diagnostic Logging:** Routes system milestones and full multi-line tracebacks concurrently to the terminal console and an internal `backend_errors.log` file for offline auditing.

---

## System Architecture

The application enforces a separation of concerns between the transport layer (API routing) and the service layer (vector processing and text extraction).

### Application Flow
1. **Client Request:** The client initiates an API call containing document or query parameters.
2. **FastAPI Layer:** Validates MIME types, verifies incoming payload schemas, coordinates execution blocks, and returns web-standard HTTP status codes.
3. **Service Layer:** Handles file stream conversions, partitions raw text into clean chunks, and executes context queries or updates against ChromaDB collections.
4. **Third-Party Service Layer:** Communicates with the Google GenAI SDK to generate high-density embeddings and synthesize context responses.

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

---

## Known Limitations & Engineering Trade-offs

* **Absence of Cryptographic Access Control:** The current system lacks user authentication and role-based validation. All generated collections are publicly accessible via the API endpoints, meaning any client can query or overwrite data if they know the collection name.
* **Lack of Optical Character Recognition (OCR):** Document parsing depends entirely on standard structural text extraction strings. PDFs consisting of raw scanned images yield zero characters, prompting a termination guard rather than falling back to an OCR conversion step.
* **Single-Node Local Storage Model:** ChromaDB runs in embedded memory mode with file persistence bound to the host filesystem. This layout cannot scale horizontally across multiple container nodes or distributed cloud environments without transitioning to an independent, client-server database deployment.
* **Stateless Conversational Execution:** The `/ask/` endpoint runs statelessly and lacks an underlying session database or caching history. The model processes each incoming question independently, preventing multi-turn conversational follow-ups or reference tracking.

---

## Core Technology Stack

* **Framework:** FastAPI (Asynchronous ASGI setup)
* **Vector Database:** ChromaDB (Local storage persistence)
* **AI Engine & Embeddings:** Client-based Google GenAI SDK (`gemini-embedding-2`, `gemini-2.5-flash`)
* **Environment Management:** Anaconda / Python Virtual Environments

---

## Project Structure

```text
AiProject/
├── main.py                  # FastAPI routers, payload schemas, and HTTP error handling
├── backend_errors.log       # Persistent internal server error tracebacks (auto-generated)
├── requirements.txt         # Project package dependencies
└── services/
    ├── __init__.py
    ├── pdf_processor.py     # Text extraction and document parsing logic
    └── vector_store.py      # ChromaDB collection manipulation and embedding pipeline
```
## Installation & Setup

### 1. Activate Environment
Ensure your virtual environment is configured and active before running the application:
```bash
conda activate your_env
```
### 2. Configure Your API Environment Variable
The modern Google GenAI SDK automatically looks for the GEMINI_API_KEY environment variable. Set it in your terminal session before launching the backend server:
```DOS
set GEMINI_API_KEY=your_actual_api_key_here
```
``` PowerShell
$env:GEMINI_API_KEY="your_actual_api_key_here"
```
## API Endpoints Reference
### 1. Upload and Index Document
- **Endpoint:** `POST /upload/`

- **Query Parameters:** `collection_name` (string)

- **Payload:** `Multipart/Form-Data` containing a single PDF file.

**Validation Guards:**

  - Returns `415 Unsupported Media Type` if the file type is not explicitly `application/pdf`.

  - Returns `400 Bad Request` if the PDF contains no extractable text string (e.g., blank or raw scanned images without OCR).

  - Returns `500 Internal Server Error` if the internal embedding processes or vector database writes fail.
### 2. Query Document Context
- **Endpoint:** `POST /ask/`

- **JSON Request Body Schema:**
```JSON
{
  "question": "What are the termination liabilities in section 4?",
  "collection_name": "corporate_contracts_2026",
  "filename": "employment_agreement.pdf"
}
```

**Validation Guards:**

   - Returns `404 Not Found` if the targeted collection does not exist on disk (intercepted when ChromaDB throws a `ValueError`).

   - Returns `503 Service Unavailable` if your Gemini API free-tier daily quota (`ClientError`) is exhausted.

## Local Development Execution
To launch the local Uvicorn development server with real-time reload tracking active, run:

```Bash
uvicorn main:app --reload
```
Once the application initializes, access the interactive documentation page directly at:

http://127.0.0.1:8000/docs

You can use this portal to test sample multipart upload payloads and structured query routes directly from your browser.