from pydantic import BaseModel
class QuestionRequest(BaseModel):
    question:str
    filename:str
    collection_name:str


