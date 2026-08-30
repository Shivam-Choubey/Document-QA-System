"""
app/main.py
===========

This file is the Streamlit UI - the only part of the project the user
directly interacts with. It does NOT contain any RAG logic itself; it
simply calls the three pipeline modules in order:

    ingestion.ingest_documents()          -> builds the searchable index
    retrieval.retrieve_relevant_chunks()  -> finds relevant context
    generation.generate_answer()          -> writes the final answer

Keeping the UI free of RAG logic means you could swap Streamlit for a
different interface (a CLI, a Flask API, etc.) later without touching
ingestion.py, retriever.py, or generator.py at all.
"""

import os
import sys
import tempfile

import streamlit as st

# Allow running this file directly with `streamlit run app/main.py` from the
# project root, by making sure the project root is on Python's import path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.ingestion import ingest_documents
from retrieval.retriever import retrieve_relevant_chunks
from generation.generator import generate_answer


st.set_page_config(page_title="Document Q&A System", layout="centered")

st.title("Document Q&A System")
st.caption("Ask questions about your own PDF documents, answered by a fully local RAG pipeline.")


# ---------------------------------------------------------------------------
# SECTION 1: DOCUMENT UPLOAD + PROCESSING
# ---------------------------------------------------------------------------
st.header("1. Upload Documents")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button("Process Documents", type="primary"):
    if not uploaded_files:
        # Basic error handling: nothing was uploaded.
        st.error("Please upload at least one PDF before processing.")
    else:
        # Streamlit gives us uploaded files as in-memory objects, but our
        # ingestion pipeline (via PyMuPDF) expects file PATHS on disk. So we
        # write each uploaded file to a temporary location first.
        temp_paths = []
        temp_dir = tempfile.mkdtemp()
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_paths.append(temp_path)

        with st.spinner("Processing documents (extracting text, chunking, embedding, storing)..."):
            try:
                num_chunks = ingest_documents(temp_paths)
                st.success(f"Successfully processed {len(uploaded_files)} document(s) into {num_chunks} chunks.")
            except ValueError as error:
                # Errors we deliberately raised ourselves in ingestion.py
                # (e.g. empty/unreadable PDFs) get a friendly message.
                st.error(str(error))
            except Exception as error:
                # Anything unexpected (model download failure, disk issues,
                # etc.) still gets shown to the user instead of crashing
                # the whole app silently.
                st.error(f"Something went wrong while processing documents: {error}")


st.divider()


# ---------------------------------------------------------------------------
# SECTION 2: QUESTION + ANSWER
# ---------------------------------------------------------------------------
st.header("2. Ask a Question")

question = st.text_input("Ask a question about your documents...")

if st.button("Get Answer"):
    if not question or not question.strip():
        # Basic error handling: empty question box.
        st.error("Please type a question before asking.")
    else:
        try:
            with st.spinner("Searching documents for relevant context..."):
                # RETRIEVAL step: find the chunks most relevant to the
                # question. This is where the "R" in RAG happens.
                relevant_chunks = retrieve_relevant_chunks(question)

            with st.spinner("Generating answer using the local LLM..."):
                # GENERATION step: turn (question + retrieved context)
                # into a final natural-language answer. This is the "G"
                # in RAG.
                result = generate_answer(question, relevant_chunks)

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Sources")
            if result["sources"]:
                for source in result["sources"]:
                    st.write(f"- {source}")
            else:
                st.write("No sources were used for this answer.")

            # Showing the raw retrieved chunks is optional, but it is
            # extremely useful for LEARNING and for double-checking that
            # retrieval actually found the right information - which is
            # exactly why this project exists.
            with st.expander("See retrieved context (for learning/debugging)"):
                for i, chunk in enumerate(relevant_chunks, start=1):
                    st.markdown(f"**Chunk {i} — from `{chunk['filename']}` (distance: {chunk['distance']:.4f})**")
                    st.write(chunk["text"])

        except ValueError as error:
            # Errors we deliberately raised ourselves in retriever.py
            # (e.g. "no documents indexed yet") get a friendly message
            # instead of a scary traceback.
            st.error(str(error))
        except Exception as error:
            st.error(f"Something went wrong while answering the question: {error}")
