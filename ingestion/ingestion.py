"""
ingestion.py
============

This file contains the complete INDEXING pipeline of our RAG system.

Indexing is the "offline" part of RAG - it happens once (or whenever new
documents are added), BEFORE the user asks any questions. Its job is to
convert raw PDF files into a searchable format.

The pipeline implemented here follows these steps:

    PDF file
       |
       v
    Extract raw text        (a computer cannot search inside a PDF binary,
       |                      it needs plain text)
       v
    Split text into chunks  (an LLM cannot read a 200-page document at once,
       |                      and we don't want to retrieve irrelevant text)
       v
    Convert chunks to       (semantic search needs numbers, not words -
    embeddings                see the comment inside create_embeddings())
       |
       v
    Store everything in     (a vector database that can quickly find the
    ChromaDB                  most similar chunks to any new question)

Every function below does ONE step of this pipeline, so that if you forget
how RAG works, you can read this file top to bottom and reconstruct the
whole idea.
"""

from typing import List, Dict
import os

import fitz  # PyMuPDF - used to open PDFs and pull out their text
from sentence_transformers import SentenceTransformer
import chromadb


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# Keeping these values in one place (instead of hard-coding them deep inside
# functions) makes the project easy to tweak later without hunting through
# the code.

# This is the local embedding model. "Local" means it downloads once from
# HuggingFace and then runs entirely on your own CPU/GPU - no API key,
# no internet needed after the first download, no cost per request.
#
# all-MiniLM-L6-v2 is a good beginner choice because:
#   - It is small (~80MB), so it downloads and loads quickly.
#   - It produces 384-dimensional embeddings, which is enough to capture
#     meaning for most beginner projects without being slow.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# chunk_size = the maximum number of CHARACTERS in one chunk of text.
# chunk_overlap = how many characters the END of one chunk repeats at the
#                 START of the next chunk.
#
# WHY do we need overlap?
# Imagine a sentence explaining something important gets cut exactly in
# half by a chunk boundary. Without overlap, neither chunk would contain
# the full sentence, and the retriever might miss the answer entirely.
# Overlap acts like a safety net that keeps context from being lost.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Where ChromaDB will save its data on disk. Because it is "persistent",
# we don't need to re-process our PDFs and regenerate embeddings every time
# we restart the app - ChromaDB remembers everything between runs.
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "document_qa_collection"


# ---------------------------------------------------------------------------
# STEP 1: LOAD DOCUMENTS
# ---------------------------------------------------------------------------
def load_documents(pdf_paths: List[str]) -> List[Dict]:
    """
    Extract raw text from a list of PDF files.

    Args:
        pdf_paths: List of file paths pointing to PDF documents.

    Returns:
        A list of dictionaries, one per PDF, each containing:
            {"filename": <name of the pdf>, "text": <all extracted text>}
    """
    # A PDF is a binary file format designed for visual layout (fonts,
    # positions, images) - not for reading as plain text. Neither our
    # embedding model nor our LLM can understand that binary layout
    # directly. So the very first thing RAG needs to do is convert the
    # PDF into plain text that downstream steps can actually process.
    documents = []

    for path in pdf_paths:
        filename = os.path.basename(path)

        try:
            pdf_file = fitz.open(path)
        except Exception as error:
            # We don't want one broken/corrupted PDF to crash the whole
            # ingestion run, so we skip it and keep processing the rest.
            print(f"[load_documents] Could not open '{filename}': {error}")
            continue

        # A PDF is made of pages. We loop through every page and glue
        # its text onto the end of one big string that represents the
        # entire document.
        full_text = ""
        for page in pdf_file:
            full_text += page.get_text()
        pdf_file.close()

        # An empty PDF (e.g. a scanned image with no selectable text)
        # would otherwise silently create an empty, useless chunk later.
        # We catch that problem here, as close to the source as possible.
        if not full_text.strip():
            print(f"[load_documents] Warning: '{filename}' has no extractable text (it may be a scanned image).")
            continue

        documents.append({"filename": filename, "text": full_text})

    return documents


# ---------------------------------------------------------------------------
# STEP 2: CHUNK DOCUMENTS
# ---------------------------------------------------------------------------
def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Split extracted document text into smaller overlapping chunks.

    Smaller chunks make semantic retrieval more precise because
    ChromaDB can return only the sections relevant to the question,
    instead of an entire document.

    Args:
        documents: Output of load_documents() -
                   [{"filename": ..., "text": ...}, ...]

    Returns:
        A list of chunk dictionaries:
            [{"filename": ..., "chunk_text": ..., "chunk_id": ...}, ...]
    """
    # WHY do we chunk at all, instead of embedding the whole document?
    #
    # 1) Embedding models compress text into a FIXED-size vector. If you
    #    feed in an entire 50-page document, most of its unique detail
    #    gets "averaged away" and the resulting vector becomes a vague
    #    summary that won't match specific questions well.
    #
    # 2) Even if it embedded perfectly, retrieving "the whole document"
    #    for every question would flood the LLM's prompt with mostly
    #    irrelevant text, making it harder (and slower/more expensive)
    #    for the LLM to find the actual answer.
    #
    # Chunking solves both problems: each small piece of text gets its
    # own precise embedding, and retrieval can return just the 2-3
    # chunks that are actually relevant to the question.
    all_chunks = []

    for doc in documents:
        text = doc["text"]
        filename = doc["filename"]

        start = 0
        chunk_index = 0
        text_length = len(text)

        while start < text_length:
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            if chunk_text:  # avoid storing empty/whitespace-only chunks
                all_chunks.append(
                    {
                        "filename": filename,
                        "chunk_text": chunk_text,
                        # A unique ID is required by ChromaDB to identify
                        # each stored item (similar to a primary key in a
                        # normal database).
                        "chunk_id": f"{filename}_chunk_{chunk_index}",
                    }
                )
                chunk_index += 1

            # Move the window forward by (chunk_size - chunk_overlap)
            # instead of chunk_size. This is exactly what creates the
            # overlap: the next chunk starts a little BEFORE the previous
            # one ended, so a sentence sitting on the boundary appears
            # fully in at least one chunk.
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return all_chunks


# ---------------------------------------------------------------------------
# STEP 3: CREATE EMBEDDINGS
# ---------------------------------------------------------------------------
def create_embeddings(chunks: List[Dict], embedding_model: SentenceTransformer) -> List[List[float]]:
    """
    Convert each text chunk into a numerical embedding vector.

    Args:
        chunks: Output of chunk_documents().
        embedding_model: A loaded SentenceTransformer model.

    Returns:
        A list of embedding vectors (one per chunk), in the same order
        as the input chunks.
    """
    # An embedding is a list of numbers (a vector) that represents the
    # MEANING of a piece of text, not its exact wording.
    #
    #   "The cat sat on the mat"   ->  [0.12, -0.43, 0.78, ...]
    #   "A cat was sitting on a mat" -> [0.13, -0.41, 0.77, ...]  (very close!)
    #   "The stock market crashed"  -> [-0.90, 0.31, 0.05, ...]  (far away)
    #
    # Sentences with SIMILAR MEANING end up with vectors that are close
    # together in this numerical space, even if they don't share the same
    # exact words. This is exactly what lets us later search for "similar
    # meaning" instead of "exact keyword match".
    #
    # IMPORTANT: the SAME embedding model must be used for both the
    # document chunks (here, during ingestion) and the user's question
    # (later, during retrieval). If we used two different models, their
    # numerical spaces wouldn't line up, and "closeness" would become
    # meaningless.
    texts = [chunk["chunk_text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    # SentenceTransformer returns a NumPy array; ChromaDB expects plain
    # Python lists, so we convert here.
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# STEP 4: STORE IN CHROMADB
# ---------------------------------------------------------------------------
def store_in_chromadb(chunks: List[Dict], embeddings: List[List[float]]) -> None:
    """
    Save chunks, their embeddings, and metadata into a persistent
    ChromaDB collection.

    Args:
        chunks: Output of chunk_documents().
        embeddings: Output of create_embeddings(), aligned by index
                    with `chunks`.
    """
    # A regular database is great for looking things up by an exact key
    # (e.g. "find the row where id = 5"). But our question is "find the
    # text whose MEANING is closest to this question" - a regular
    # database has no concept of "meaning" or "closeness".
    #
    # A VECTOR DATABASE like ChromaDB is built specifically to store
    # vectors (embeddings) and quickly answer "which stored vectors are
    # most similar to this new vector?" - which is exactly the operation
    # semantic search needs.
    #
    # For every chunk, we store THREE aligned pieces of information:
    #   1. document (chunk_text) - the actual text we'll show to the LLM
    #   2. embedding              - the numerical vector used for search
    #   3. metadata (filename)    - extra info we want back at query time,
    #                                e.g. so we can tell the user WHICH
    #                                PDF an answer came from
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # get_or_create_collection means: reuse the collection if it already
    # exists, or create it fresh if this is the first run. We deliberately
    # do NOT delete/recreate it on every ingestion call, because that
    # would throw away previously indexed documents every time the app
    # restarts - persistence is the whole point of using ChromaDB instead
    # of an in-memory list.
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents_text = [chunk["chunk_text"] for chunk in chunks]
    metadatas = [{"filename": chunk["filename"]} for chunk in chunks]

    # `upsert` = update if the id already exists, insert if it's new.
    # This makes it safe to re-run ingestion on the same PDFs without
    # creating duplicate entries.
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents_text,
        metadatas=metadatas,
    )


# ---------------------------------------------------------------------------
# MAIN PIPELINE FUNCTION
# ---------------------------------------------------------------------------
def ingest_documents(pdf_paths: List[str]) -> int:
    """
    Run the full ingestion pipeline: load -> chunk -> embed -> store.

    Args:
        pdf_paths: List of paths to PDF files to index.

    Returns:
        The number of chunks that were created and stored.

    Raises:
        ValueError: if no valid text could be extracted from any PDF.
    """
    # This function exists purely to WIRE TOGETHER the four steps above
    # in the correct order. Keeping it separate from the individual steps
    # means the Streamlit app only ever needs to call this one function,
    # while we (humans reading the code later) can still study each step
    # in isolation above.
    print("Step 1/4: Extracting text from PDFs...")
    documents = load_documents(pdf_paths)

    if not documents:
        raise ValueError(
            "No text could be extracted from the provided PDF(s). "
            "They may be empty, corrupted, or scanned images without a text layer."
        )

    print("Step 2/4: Splitting text into chunks...")
    chunks = chunk_documents(documents)

    print("Step 3/4: Generating embeddings locally (this may take a moment)...")
    # Loading the model here (instead of at import time) means the
    # (fairly large) model file is only downloaded/loaded when we
    # actually need to ingest something.
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = create_embeddings(chunks, embedding_model)

    print("Step 4/4: Storing chunks and embeddings in ChromaDB...")
    store_in_chromadb(chunks, embeddings)

    print(f"Done! Indexed {len(chunks)} chunks from {len(documents)} document(s).")
    return len(chunks)
