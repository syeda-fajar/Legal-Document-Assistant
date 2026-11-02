import os
import chromadb
from google import genai
from google.genai import types
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

def store_chunk_in_db(chunks_with_metadata: list[dict], collection_name: str, source_filename: str):
    collection = Db_client.get_or_create_collection(name=collection_name)
    
    existing_record = collection.get(where={"source_filename": source_filename})
    if existing_record and existing_record.get('ids'):
        logger.info(f"'{source_filename}' already exists in the database. Purging old chunks for an accurate overwrite...")
        collection.delete(where={"source_filename": source_filename})
        try:
            cache_collection.delete(where={"source_filename": source_filename})
            logger.info(f"Successfully invalidated and cleared stale cache history for file: {source_filename}")
        except Exception as e:
            logger.debug(f"No semantic cache entries found to invalidate for {source_filename}: {str(e)}")
            
    logger.info(f"Embedding and storing fresh {len(chunks_with_metadata)} chunks...") 

  
    documents = [item["document"] for item in chunks_with_metadata]
    metadata_list = [item["metadata"] for item in chunks_with_metadata]
    
    embedding_list = []
    for doc in documents:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=doc,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        embedding_list.append(response.embeddings[0].values)
    
    ids_list = [f"{source_filename}_chunk_{idx}" for idx in range(len(chunks_with_metadata))]
    
    collection.add(documents=documents, embeddings=embedding_list, ids=ids_list, metadatas=metadata_list)
    logger.info(f"Successfully stored {len(chunks_with_metadata)} chunks in the vector database.")