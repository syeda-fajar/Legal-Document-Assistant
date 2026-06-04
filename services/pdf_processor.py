from pypdf import PdfReader
from io import BytesIO
import os
def extract_text_from_pdf(pdf_stream: BytesIO) -> str:
    reader = PdfReader(pdf_stream)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:  
            full_text += text
    return full_text

def chunk_text(text, chunk_size=500, chunk_overlap=50):
  chunks = []
  start =0
  while start <len(text):
     if start + chunk_size >= len(text):
        chunks.append(text[start:]) 
        break
     end = start + chunk_size
     space_index = text.rfind(" ",start,end)
     if space_index != -1:
        end =space_index

     chunk = text[start:end]
     chunks.append(chunk)
     start = end - chunk_overlap
  
  return chunks


