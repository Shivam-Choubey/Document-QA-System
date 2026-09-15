"""
app/main.py
===========

Thin HTTP layer over the existing RAG pipeline. No RAG logic here -
just JSON in, JSON out. The three pipeline modules are called exactly
as before:

    ingestion.ingest_documents()
    retrieval.retrieve_relevant_chunks()
    generation.generate_answer()
"""

import os
import sys
import shutil
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.ingestion import ingest_documents
from retrieval.retriever import retrieve_relevant_chunks
from generation.generator import generate_answer


app = FastAPI(title="Document Q&A")

HERE = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=HERE), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# /ingest  - multipart upload of one or more PDFs
# ---------------------------------------------------------------------------
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one PDF.")

    temp_dir = tempfile.mkdtemp()
    temp_paths = []
    try:
        for f in files:
            # Basic guard: only PDFs
            if not f.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"Not a PDF: {f.filename}")
            path = os.path.join(temp_dir, f.filename)
            with open(path, "wb") as out:
                shutil.copyfileobj(f.file, out)
            temp_paths.append(path)

        try:
            num_chunks = ingest_documents(temp_paths)
        except ValueError as e:
            # Deliberate, friendly errors from ingestion.py
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Something went wrong while processing documents: {e}",
            )

        return {
            "num_chunks": num_chunks,
            "files": [f.filename for f in files],
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# /ask  - question -> answer + sources + retrieved chunks
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please type a question before asking.")

    try:
        relevant_chunks = retrieve_relevant_chunks(question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    try:
        result = generate_answer(question, relevant_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return JSONResponse(
        {
            "answer": result["answer"],
            "sources": result["sources"],
            "chunks": [
                {
                    "filename": c["filename"],
                    "distance": c["distance"],
                    "text": c["text"],
                }
                for c in relevant_chunks
            ],
        }
    )