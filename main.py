from fastapi import FastAPI,File,UploadFile,HTTPException,status
from io import BytesIO
from typing import Annotated
from services.pdf_processor import extract_text_from_pdf,chunk_text
from services.vector_store import store_chunk_in_db
from services.search_engine import answer_document_question
from schema import QuestionRequest
from google.genai import errors
import logging
import sys
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
app=FastAPI()

@app.get("/")
def startup():
    return{"messgae":"you did it bro"}

@app.post("/upload/")
async def upload_file(file:Annotated[UploadFile, File()],collection_name :str):
    
    if file.content_type != "application/pdf" :
        logger.warning(f"Rejected upload attempt. Invalid MIME type: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Only pdf are allowed."
        )
    pdf_file = await file.read()
    pdf_stream =BytesIO(pdf_file)
    extract_text = extract_text_from_pdf(pdf_stream)
    if not extract_text or not extract_text.strip():
        logger.warning(f"File '{file.filename}' processed but yielded zero extractable text.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The uploaded PDF contains no extractable text. Please ensure the document is not completely empty, corrupted."
        )
    processed_chunk = chunk_text(extract_text)
    try:
        store_data = store_chunk_in_db(processed_chunk,collection_name,source_filename=file.filename)
        return {
        "status": "success",
        "filename": file.filename,
        "chunks_indexed": len(processed_chunk),
        "message": f"Successfully parsed and stored {len(processed_chunk)} chunks into the vector store."
    }
    except Exception as e:
        logger.error(f"Database ingestion crashed for file '{file.filename}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating embeddings or storing data in the vector database."
        )

@app.post("/ask/")
async def user_query(request:QuestionRequest):
    try:
      answer = await answer_document_question(request.question,request.collection_name,source_filename=request.filename)
      return {
      "status": "success",
      "filename": request.filename,
      "question": request.question,
      "answer": answer
}
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND ,detail="collection dont exist")
    except errors.ClientError as e:
        logger.warning(f"429 RESOURCE_EXHAUSTED quota limits{str(e)}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,detail="Service Unavailable / Quota Exhausted")
    except errors.APIError as e:
        logger.warning(f"Bad gateway an api error {str(e)}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,detail="Bad Gateway")
    except Exception as e:
        logger.error(f"Unhandled system crash: {str(e)}", exc_info=True)
        raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Something went wrong internally. Please contact support."
    )