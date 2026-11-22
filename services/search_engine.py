from google.genai import types
import chromadb
import uuid
from google import genai
from dotenv import load_dotenv
import logging
from fastapi import HTTPException, status
from schema import RAGAnswerResponse, GeminiAnswerResponse

logger = logging.getLogger(__name__)
load_dotenv()

client = genai.Client()

Db_client = chromadb.PersistentClient(path="./legal_db")

cache_collection = Db_client.get_or_create_collection(
    name="answer_cache", 
    metadata={"hnsw:space": "cosine"}
)

async def answer_document_question(question: str, collection_name: str, source_filename: str) -> RAGAnswerResponse:
    
    response = await client.aio.models.embed_content(
        model="gemini-embedding-2", 
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    question_vector = response.embeddings[0].values

    cached_results = cache_collection.query(
        query_embeddings=[question_vector],
        n_results=1,
        where={"source_filename": source_filename},
        include=["documents", "distances", "metadatas"]
    )
    
    if cached_results and cached_results.get("distances") and len(cached_results["distances"][0]) > 0:
        best_distance = cached_results["distances"][0][0]
        logger.debug("Cache evaluation distance: %.4f (Target <= 0.15)", best_distance)
        
        if best_distance <= 0.15:
            cached_doc = cached_results["documents"][0][0]
            cached_id = cached_results["ids"][0][0]
            try:
                parsed_cache = RAGAnswerResponse.model_validate_json(cached_doc)
                logger.info("Cache Hit: Found semantic match with distance %.4f. Bypassing Gemini.", best_distance)
                return parsed_cache
            except Exception as e:
                logger.warning("Cache Hit, but stored document is legacy plain text or invalid JSON (%s). Purging entry.", e)
                cache_collection.delete(ids=[cached_id])
        else:
            logger.info("Cache Miss: Item exists in index but distance is too high (%.4f).", best_distance)
    else:
        logger.info("Cache Miss: No matching records found in cache index.")

    logger.info("Retrieving context fragments from document collection: %s", collection_name)
    try:
        collection = Db_client.get_or_create_collection(name=collection_name)
    except Exception as e:
        logger.error("Collection not found: %s", collection_name)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document collection '{collection_name}' not found. Please upload the document first."
        )

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=5,
        where={"source_filename": source_filename},
        include=["documents", "metadatas", "distances"]
    )
    
    if not results or not results.get('documents') or len(results['documents'][0]) == 0:
        return RAGAnswerResponse(
            answer="I cannot find the answer to that question because no relevant context was found for this document layout.",
            found=False,
            sources=[],
            top_distance=1.0
        )
        
    retrieved_chunks = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0] if results.get('metadatas') else []
    retrieved_distances = results['distances'][0] if results.get('distances') else [0.0]

    # Format sources with Filename + Page Number (e.g., "contract.pdf, Page 3")
    formatted_sources = []
    seen_sources = set()
    context_parts = []

    for doc, meta in zip(retrieved_chunks, retrieved_metadatas):
        fname = meta.get("source_filename", source_filename)
        page_num = meta.get("page_number")
        
   
        source_label = f"{fname}, Page {page_num}" if page_num is not None else fname
        if source_label not in seen_sources:
            seen_sources.add(source_label)
            formatted_sources.append(source_label)
            
       
        header = f"[{source_label}]"
        context_parts.append(f"{header}\n{doc}")

    best_chunk_distance = float(retrieved_distances[0]) if retrieved_distances else 0.0

    logger.debug("Successfully pulled %d chunks from primary vector store. Best distance: %.4f", len(retrieved_chunks), best_chunk_distance)
    context_text = "\n---\n".join(context_parts)
    
    prompt = f"""
    You are an expert legal and document analysis assistant. 
    Answer the user's question in markdown format using ONLY the verified document context provided below.
    Each context block is labeled with its source file and page number.
    If the answer cannot be found or inferred from the context, set found to false and provide a polite explanation. Do not make up facts.
    
    VERIFIED DOCUMENT CONTEXT:
    {context_text}
    
    USER QUESTION:
    {question}
    """

    try:
        print("Gemini is analyzing the retrieved context chunks...")
        ai_response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiAnswerResponse
            )
        )

        gemini_data = GeminiAnswerResponse.model_validate_json(ai_response.text)
        
    except Exception as e:
        logger.error("Failed to generate or validate structured response from Gemini: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI model returned an invalid structure or failed to respond. Please try again."
        )

    parsed_response = RAGAnswerResponse(
        answer=gemini_data.answer,
        found=gemini_data.found,
        sources=formatted_sources,
        top_distance=best_chunk_distance
    )

    if parsed_response.found:
        final_json_to_cache = parsed_response.model_dump_json()
        logger.info("Caching new query-response set in 'answer_cache' collection.")
        cache_collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[question_vector],
            metadatas=[{"source_filename": source_filename}],
            documents=[final_json_to_cache]
        )
    else:
        logger.info("Skipping cache write for negative result (found: false).")

    return parsed_response