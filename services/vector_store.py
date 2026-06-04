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

def store_chunk_in_db(chunks: list, collection_name: str, source_filename: str):
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
            
    logger.info(f"Embedding and storing fresh {len(chunks)} chunks...") 

    
    embedding_list = []
    for chunk in chunks:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=chunk,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        
        embedding_list.append(response.embeddings[0].values)
    
    
    ids_list = [f"{source_filename}_chunk_{idx}" for idx in range(len(chunks))]
    metadata_list = [{"source_filename": source_filename} for _ in range(len(chunks))]

    
    collection.add(documents=chunks, embeddings=embedding_list, ids=ids_list, metadatas=metadata_list)
    logger.info(f"Successfully stored {len(chunks)} chunks in the vector database.")