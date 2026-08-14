"""
Ingestion / embedding / retrieval stage of the RAG pipeline.

- Loads the 8 corpus documents from docs/ (one chunk per document, since each
  document is already short and topically self-contained).
- Embeds each chunk locally with sentence-transformers (all-MiniLM-L6-v2) --
  no API key, no network call to any LLM provider.
- Stores the embeddings in a persistent ChromaDB collection ("zepto_policies").
- Exposes retrieve_top_chunks(query, k) which embeds the query and returns the
  top-k most similar chunks via cosine similarity. This retrieval step always
  runs for real, in both MOCK_LLM=1 and MOCK_LLM=0 modes.
"""

import os
from typing import Any, Dict, List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

# Sets the folder containing the policy documents.
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")

# Sets the location where ChromaDB stores the vector database.
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# Defines the name of the ChromaDB collection.
COLLECTION_NAME = "zepto_policies"

# Stores the embedding model after it is loaded for reuse.
_model: SentenceTransformer | None = None

# Stores the ChromaDB client after it is created.
_client = None

# Stores the ChromaDB collection after it is created.
_collection = None


# Loads the local sentence-transformers embedding model only when needed.
def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# Creates and returns a persistent ChromaDB client.
def get_chroma_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


# Loads each policy document as one text chunk.
def _load_documents() -> List[Tuple[str, str]]:
    """Load each doc_XX.txt as a single chunk: (chunk_id, content)."""
    docs = []
    for fname in sorted(os.listdir(DOCS_DIR)):
        if fname.endswith(".txt"):
            doc_id = os.path.splitext(fname)[0]  # e.g. "doc_01"
            with open(os.path.join(DOCS_DIR, fname), "r", encoding="utf-8") as f:
                content = f.read().strip()
            docs.append((doc_id, content))
    return docs


# Embeds the documents and stores them in the ChromaDB collection.
def _ingest_documents(collection) -> None:
    model = get_embedding_model()
    docs = _load_documents()
    ids = [d[0] for d in docs]
    texts = [d[1] for d in docs]
    embeddings = model.encode(texts).tolist()

    # Clear any stale rows with the same ids before (re)adding, so re-running
    # ingestion (e.g. on container restart) never raises a duplicate-id error.
    try:
        collection.delete(ids=ids)
    except Exception:
        pass

    collection.add(ids=ids, documents=texts, embeddings=embeddings)


# Returns the existing ChromaDB collection or creates and populates it when needed.
def build_or_get_collection():
    """Return the ChromaDB collection, ingesting the corpus the first time."""
    global _collection

    # Reuse the collection if it was already loaded during this run.
    if _collection is not None:
        return _collection

    # Connect to the persistent ChromaDB database.
    client = get_chroma_client()
    existing_names = [c.name for c in client.list_collections()]

    # Reuse the collection when all 8 policy documents are already stored.
    if COLLECTION_NAME in existing_names:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() >= 8:
            _collection = collection
            return _collection

    # Create the collection and add the policy documents if it is not ready.
    collection = client.get_or_create_collection(COLLECTION_NAME)
    _ingest_documents(collection)
    _collection = collection
    return _collection

# Finds the most relevant policy chunks for a user's question.
def retrieve_top_chunks(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Embed the query and retrieve the top-k most similar chunks (cosine similarity)."""
    
    # Get the ChromaDB collection containing the policy embeddings.
    collection = build_or_get_collection()

    # Convert the user's question into an embedding vector.
    model = get_embedding_model()
    query_embedding = model.encode([query]).tolist()

    # Search ChromaDB for the k most similar policy chunks.
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(ids)

    # Return the retrieved document IDs, content, and similarity distances.
    return [
        {"id": cid, "content": text, "distance": dist}
        for cid, text, dist in zip(ids, documents, distances)
    ]

if __name__ == "__main__":
    # Manual smoke test: `python ingest.py`
    coll = build_or_get_collection()
    print(f"Collection '{COLLECTION_NAME}' has {coll.count()} chunks.")
    for chunk in retrieve_top_chunks("What is the refund policy?", k=3):
        print(chunk["id"], "->", chunk["content"][:80])