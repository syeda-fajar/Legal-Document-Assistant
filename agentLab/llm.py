import requests


def chat(messages, model="llama3.2:3b", temperature=0):
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": False,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]

import requests

EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"

def embed(texts):
    r = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embeddings"]