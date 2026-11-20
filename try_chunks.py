from services.pdf_processor import chunk_text_with_metadata

# Original test cases
cases = {
    "no_spaces": "x" * 5000,
    "space_at_10": "a" * 10 + " " + "b" * 5000,
    "one_space_2000": "c" * 1000 + " " + "d" * 999,
}

for name, text in cases.items():
    chunks = chunk_text_with_metadata(text, [], "test.pdf")
    num_chunks = len(chunks)
    first_chunk_len = len(chunks[0]["document"]) if chunks else 0
    print(f"Case: {name}")
    print(f"  - Number of chunks: {num_chunks}")
    print(f"  - First chunk length: {first_chunk_len}")
    
    for idx, item in enumerate(chunks):
        doc_text = item["document"]
        assert doc_text.strip(), f"Error: Empty or whitespace-only chunk found at index {idx} in case '{name}'"
    
    if chunks:
        last_chunk = chunks[-1]["document"]
        assert text.endswith(last_chunk), f"Error: Last chunk in case '{name}' does not match the end of the text."
    
    print(f"  - Status: PASSED (All chunks valid & reaches text end)\n")

# Fourth Case: Fake Boundary Multi-Page Test
print("Case: fake_boundary_multipage")
boundaries = [
    {"page_number": 1, "start": 0, "end": 300},
    {"page_number": 2, "start": 300, "end": 600},
]
text_fake = "word " * 120  # 600 characters total
chunks_fake = chunk_text_with_metadata(text_fake, boundaries, "test.pdf")

if chunks_fake:
    print(f"  - First chunk metadata: {chunks_fake[0]['metadata']}")
    print(f"  - First chunk length: {len(chunks_fake[0]['document'])}")
else:
    print("  - No chunks generated.")