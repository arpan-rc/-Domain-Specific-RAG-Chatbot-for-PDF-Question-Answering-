
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from document_loader import load_document
from rag_pipeline import (
    generate_rag_response,
    retrieve_documents,
    validate_document_size,
)
from vector_store import create_and_save_vector_store, load_local_vector_store

app = FastAPI(title="Domain RAG Chatbot API")

# Maximum upload size: 100 MiB
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024
CHUNK_SIZE_BYTES = 1024 * 1024  # Read uploads in 1 MiB chunks


class QueryRequest(BaseModel):
    question: str
    top_chunks: int = 4


@app.get("/")
def read_root():
    return {"message": "Domain RAG Chatbot API is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name

    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    documents_dir = Path("documents")
    documents_dir.mkdir(parents=True, exist_ok=True)
    file_path = documents_dir / filename

    total_bytes = 0

    try:
        # Stream to disk and stop as soon as the limit is exceeded.
        with file_path.open("wb") as output:
            while True:
                chunk = await file.read(CHUNK_SIZE_BYTES)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds the maximum allowed size of 100 MiB.",
                    )

                output.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is empty.",
            )

        # Second check before document extraction and indexing.
        validate_document_size(file_path)

        docs = load_document(str(file_path))
        vector_store, chunk_count = create_and_save_vector_store(docs)

        return {
            "filename": filename,
            "size_bytes": total_bytes,
            "status": "Processed successfully",
            "chunks_indexed": chunk_count,
        }

    except Exception:
        # Do not leave a partial or failed upload on disk.
        if file_path.exists():
            file_path.unlink()
        raise

    finally:
        await file.close()


@app.post("/query")
def query_rag(request: QueryRequest):
    vector_store = load_local_vector_store()

    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No document index found. Upload a document first.",
        )

    retrieved_docs = retrieve_documents(
        vector_store,
        request.question,
        number_of_chunks=request.top_chunks,
    )

    # Pass an empty history because this endpoint currently has no
    # conversation-history parameter or session management.
    answer = generate_rag_response(
        request.question,
        retrieved_docs,
        conversation_history=[],
    )

    sources = [
        {
            "source": document.metadata.get("source"),
            "page": document.metadata.get("page"),
        }
        for document in retrieved_docs
    ]

    return {"answer": answer, "sources": sources}