from google.genai import types
import chromadb
import uuid
from google import genai
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)
load_dotenv()

client = genai.Client()

Db_client = chromadb.PersistentClient(path="./legal_db")


cache_collection = Db_client.get_or_create_collection(
    name="answer_cache", 
    metadata={"hnsw:space": "cosine"}
)

async def answer_document_question(question: str, collection_name: str, source_filename: str) -> str:
  
    response = await client.aio.models.embed_content(
        model="gemini-embedding-2", 
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")

    )
    question_vector = response.embeddings[0].values

    
    cached_results = cache_collection.query(
        query_embeddings=[question_vector],
        n_results=1,
        where={"source_filename": source_filename}
    )
    
    
    if cached_results and cached_results.get("distances") and len(cached_results["distances"][0]) > 0:
        best_distance = cached_results["distances"][0][0]
        logger.debug("Cache evaluation distance: %.4f (Target <= 0.15)", best_distance)
        
        if best_distance <= 0.15:
            logger.info(" Cache Hit: Found semantic match with distance %.4f. Bypassing Gemini.", best_distance)
            return cached_results["documents"][0][0]
        logger.info("Cache Miss: Item exists in index but distance is too high (%.4f).", best_distance)
    else:
        logger.info("Cache Miss: No matching records found in cache index.")

    logger.info("Retrieving context fragments from document collection: %s", collection_name)
    collection = Db_client.get_collection(collection_name)
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=5,
        where={"source_filename": source_filename}
    )
    
    if not results or not results.get('documents') or len(results['documents'][0]) == 0:
        return "I cannot find the answer to that question because no relevant context was found for this document layout."
        
    retrieved_chunks = results['documents'][0]
    logger.debug("Successfully pulled %d chunks from primary vector store.", len(retrieved_chunks))
    context_text = "\n---\n".join(retrieved_chunks)
    
    prompt = f"""
    You are an expert legal and document analysis assistant. 
    Answer the user's question markdown format using ONLY the verified document context provided below.
    If the answer cannot be found or inferred from the context, say exactly: "I cannot find the answer to that question in the provided document." Do not make up facts.
    
    VERIFIED DOCUMENT CONTEXT:
    {context_text}
    
    USER QUESTION:
    {question}
    
    YOUR ANSWER:
    """

    print("Gemini is analyzing the retrieved context chunks...")
    ai_response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    generated_answer = ai_response.text
    logger.info("Caching new query-response set in 'answer_cache' collection.")
    cache_collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[question_vector],
        metadatas=[{"source_filename": source_filename}],
        documents=[generated_answer]
    )
    
    return generated_answer