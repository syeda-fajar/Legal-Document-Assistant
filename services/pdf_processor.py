from io import BytesIO
from pypdf import PdfReader

def extract_text_from_pdf(pdf_stream: BytesIO) -> tuple[str, list[dict]]:
    """
    Extracts text from a PDF, joins pages with \n\n to prevent glued words,
    and returns both the global text string and page boundary markers.
    """
    reader = PdfReader(pdf_stream)
    full_text = ""
    page_boundries = []
    
    current_idx = 0
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            start_idx = current_idx
            full_text += text + "\n\n"
            end_idx = len(full_text)
            
            page_boundries.append({
                "page_number": page_num,
                "start": start_idx,
                "end": end_idx
            })
            current_idx = end_idx
            
    return full_text, page_boundries


def get_page_number(start_idx: int, page_boundries: list[dict]) -> int:
    """Finds which page number a chunk's starting index falls into."""
    for boundary in page_boundries:
        if boundary["start"] <= start_idx < boundary["end"]:
            return boundary["page_number"]
    return page_boundries[-1]["page_number"] if page_boundries else 1


def chunk_text_with_metadata(text: str, page_boundries: list[dict], filename: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """
    Chunks global text while mapping each chunk's start index back to its page boundary.
    """
    chunks_with_meta = []
    start = 0
    
    if not text:
        return []

    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            end = text_length
        else:
            space_index = text.rfind(" ", start, end)
            if space_index != -1 and space_index > start:
                end = space_index

        chunk = text[start:end]
        chunk_start = start
        
        page_num = get_page_number(chunk_start, page_boundries)
        
        chunks_with_meta.append({
            "document": chunk,
            "metadata": {
                "source_filename": filename,
                "page_number": page_num
            }
        })
        
        if end == text_length:
            break
            
        start = end - chunk_overlap
        
    return chunks_with_meta