import os
import logging
from collections import defaultdict
from google import genai

logger = logging.getLogger(__name__)


MAX_TURNS = int(os.getenv("MAX_TURNS", 4))

class ConversationMemory:
    def __init__(self, max_turns: int = MAX_TURNS):
        self.sessions = defaultdict(list)
        self.max_messages = max_turns * 2

    def get_history(self, session_id: str | None) -> list[dict]:
        if not session_id:
            return []
        return self.sessions.get(session_id, [])[-self.max_messages:]

    def add_turn(self, session_id: str | None, user_question: str, assistant_answer: str):
        if not session_id:
            return
        
        self.sessions[session_id].append({"role": "user", "content": user_question})
        self.sessions[session_id].append({"role": "model", "content": assistant_answer})
        self.sessions[session_id] = self.sessions[session_id][-self.max_messages:]

memory_store = ConversationMemory()

def rewrite_question(history: list[dict], question: str) -> str:
    if not history:
        return question

    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    
    prompt = f"""Given the following conversation history, rewrite the user's latest follow-up question into a clear, standalone question that includes all necessary context. 

Rules:
- Output ONLY the rewritten question text.
- Do not answer the question.
- If the question is already standalone, return it unchanged.

Conversation History:
{history_text}

Latest Question: {question}
Standalone Question:"""

    try:
        
        from main import client  
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Question rewrite failed, falling back to original question: {str(e)}")
        return question