from pydantic import BaseModel
from typing import List, Optional
class QuestionRequest(BaseModel):
    question:str
    filename:str
    collection_name:str
    session_id: Optional[str] = None
class GeminiAnswerResponse(BaseModel):
    answer: str
    found: bool

class RAGAnswerResponse(BaseModel):
    answer: str
    found: bool
    sources: List[str] = []
    top_distance: float = 0.0

class AskResponseWrapper(BaseModel):
    status: str
    filename: str
    question: str
    answer: RAGAnswerResponse
    
