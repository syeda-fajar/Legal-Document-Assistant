import os
import secrets
from typing import Annotated
from io import BytesIO
import logging
import sys
from schema import QuestionRequest, RAGAnswerResponse, AskResponseWrapper
from services.memory import memory_store, rewrite_question

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Security, Depends, APIRouter
from fastapi.security.api_key import APIKeyHeader

from services.pdf_processor import extract_text_from_pdf, chunk_text_with_metadata
from services.vector_store import store_chunk_in_db
from services.search_engine import answer_document_question
from schema import QuestionRequest
from google.genai import errors

from fastapi.security import APIKeyHeader


log_format = "%(asctime)s - [%(levelname)s] - %(filename)s - %(message)s"
logging.basicConfig(
    level=logging.INFO, 
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),        
        logging.FileHandler("backend_errors.log")   
    ]
)
logger = logging.getLogger(__name__)

API_KEY_NAME = "X-API-Key"
VALID_API_KEY = "my_api_key"


api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def verify_api_key(incoming_key: str = Security(api_key_header)):
    if not secrets.compare_digest(incoming_key.encode("utf-8"), VALID_API_KEY.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key provided."
        )
    return incoming_key

app = FastAPI(title="Legal RAG API")


router = APIRouter(
    dependencies=[Depends(verify_api_key)])

@router.post("/upload/")
async def upload_file(file: Annotated[UploadFile, File()], collection_name: str):
    if file.content_type != "application/pdf":
        logger.warning(f"Rejected upload attempt. Invalid MIME type: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Only pdf are allowed."
        )
        
    pdf_file = await file.read()
    pdf_stream = BytesIO(pdf_file)
    
    full_text, page_boundaries = extract_text_from_pdf(pdf_stream)
    
    if not full_text or not full_text.strip():
        logger.warning(f"File '{file.filename}' processed but yielded zero extractable text.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded PDF contains no extractable text. Please ensure the document is not completely empty or corrupted."
        )
        
    processed_chunks = chunk_text_with_metadata(
        text=full_text,
        page_boundries=page_boundaries,
        filename=file.filename
    )
    
    try:

        store_data = store_chunk_in_db(
            chunks_with_metadata=processed_chunks, 
            collection_name=collection_name, 
            source_filename=file.filename
        )
        return {
            "status": "success",
            "filename": file.filename,
            "total_pages": len(page_boundaries),
            "chunks_indexed": len(processed_chunks),
            "message": f"Successfully parsed and stored {len(processed_chunks)} chunks with page tracking into the vector store."
        }
    except Exception as e:
        logger.error(f"Database ingestion crashed for file '{file.filename}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating embeddings or storing data in the vector database."
        )

@router.post("/ask/", response_model=AskResponseWrapper)
async def user_query(request: QuestionRequest):
    try:
        
        history = memory_store.get_history(request.session_id)
        
       
        standalone_query = rewrite_question(history, request.question)
        
        answer_result = await answer_document_question(
            standalone_query, 
            request.collection_name, 
            source_filename=request.filename
        )
        
    
        answer_text = answer_result.get("answer") if isinstance(answer_result, dict) else str(answer_result)
        is_found = answer_result.get("found", True) if isinstance(answer_result, dict) else True

        if request.session_id and is_found:
            memory_store.add_turn(request.session_id, request.question, answer_text)
        
        return {
            "status": "success",
            "filename": request.filename,
            "question": request.question,
            "rewritten_question": standalone_query,
            "answer": answer_result
        }
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="collection dont exist")
    except errors.ClientError as e:
        logger.warning(f"429 RESOURCE_EXHAUSTED quota limits {str(e)}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service Unavailable / Quota Exhausted")
    except errors.APIError as e:
        logger.warning(f"Bad gateway an api error {str(e)}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Bad Gateway")
    except Exception as e:
        logger.error(f"Unhandled system crash: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong internally. Please contact support."
        )

app.include_router(router)