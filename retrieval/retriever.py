"""
retriever.py
============

This file contains the RETRIEVAL half of RAG (the "R" in RAG).

Retrieval happens "online" - every time the user asks a question, while
they wait for an answer. Its ONLY job is to find the most relevant chunks
of text that were stored during ingestion (see ingestion/ingestion.py).

This file intentionally does NOT talk to the LLM or build any prompts -
that responsibility belongs to generation/generator.py. Keeping retrieval
and generation separate mirrors how RAG actually works conceptually:

    "Find relevant information"   -->   "Use that information to answer"
         (this file)                        (generator.py)

The retrieval flow implemented here is:

    User Question
         |
         v
    Sentence Transformer     (must be the SAME model used in ingestion.py,
         |                     otherwise the vectors won't be comparable)
         v
    Question Embedding
         |
         v
    ChromaDB similarity search
         |
         v
    Top-K most relevant chunks
"""

from typing import List, Dict

from sentence_transformers import SentenceTransformer
import chromadb

from ingestion.ingestion import EMBEDDING_MODEL_NAME, CHROMA_DB_PATH, COLLECTION_NAME


# How many chunks to retrieve per question.
#
# WHY not just retrieve everything? Because the LLM has a limited context
# window (it can only "read" so much text at once), and stuffing it with
# too many chunks - most of them irrelevant - makes it harder for the model
# to focus on the parts that actually answer the question. top_k lets us
# hand the LLM only the handful of chunks most likely to be useful.
TOP_K = 3

# Module-level cache for the embedding model. Loading a SentenceTransformer
# model from disk takes a noticeable moment, so we only want to do it ONCE
# per app run, not on every single question.
_embedding_model = None


def _get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it across calls."""
    global _embedding_model
    if _embedding_model is None:
        # CRITICAL RAG DETAIL: this must be the exact same model used to
        # embed the document chunks during ingestion (EMBEDDING_MODEL_NAME
        # is imported from ingestion.py so the two files can never
        # accidentally drift apart). Two different embedding models
        # produce vectors in two different, unrelated numerical spaces -
        # comparing them would be like measuring one distance in miles
        # and another in kilograms.
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def retrieve_relevant_chunks(question: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Find the chunks most semantically similar to the user's question.

    Args:
        question: The user's natural-language question.
        top_k: How many chunks to return, ranked by relevance.

    Returns:
        A list of dictionaries, each shaped like:
            {"text": <chunk text>, "filename": <source pdf>, "distance": <float>}
        ordered from most relevant to least relevant.

    Raises:
        ValueError: if the question is empty, or the document collection
                    hasn't been created yet (i.e. nothing was ingested).
    """
    if not question or not question.strip():
        raise ValueError("The question is empty. Please type a question before searching.")

    # STEP 1: Turn the question into the same kind of numerical vector
    # that we created for every document chunk during ingestion. Without
    # this step, we would have no way to numerically compare "meaning"
    # between the question and the stored chunks.
    embedding_model = _get_embedding_model()
    question_embedding = embedding_model.encode([question]).tolist()

    # STEP 2: Connect to the persistent ChromaDB store that ingestion.py
    # already wrote to on disk.
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        raise ValueError(
            "No indexed documents were found. Please upload PDFs and click "
            "'Process Documents' before asking a question."
        )

    if collection.count() == 0:
        raise ValueError(
            "The document collection is empty. Please upload PDFs and click "
            "'Process Documents' before asking a question."
        )

    # STEP 3: Ask ChromaDB for the `top_k` stored vectors that are closest
    # (most similar in meaning) to the question's vector. Internally,
    # ChromaDB compares the question embedding against every stored chunk
    # embedding and ranks them by distance (smaller distance = more similar).
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(top_k, collection.count()),
    )

    # ChromaDB returns results as parallel lists nested inside a single
    # query batch (index [0] because we only sent one question). We
    # reshape that into a simple, readable list of dictionaries for the
    # rest of our app to use.
    retrieved_chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc_text, metadata, distance in zip(documents, metadatas, distances):
        retrieved_chunks.append(
            {
                "text": doc_text,
                "filename": metadata.get("filename", "unknown"),
                "distance": distance,
            }
        )

    if not retrieved_chunks:
        raise ValueError("No relevant information was found in the indexed documents for this question.")

    return retrieved_chunks
