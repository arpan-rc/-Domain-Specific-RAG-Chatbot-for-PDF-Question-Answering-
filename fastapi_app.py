import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from document_loader import load_document
from rag_pipeline import generate_rag_response, retrieve_documents, validate_document_size
from vector_store import create_and_save_vector_store, load_local_vector_store

app = FastAPI(title="Domain RAG Chatbot API")

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024
CHUNK_SIZE_BYTES = 1024 * 1024
DOCUMENTS_DIR = Path("documents")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    top_chunks: int = Field(default=4, ge=1, le=10)


@app.get("/")
def read_root():
    return {"message": "Domain RAG Chatbot API is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        await file.close()
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="upload_", suffix=".pdf", dir=DOCUMENTS_DIR, delete=False
        ) as output:
            temp_path = Path(output.name)
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
            raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
        with temp_path.open("rb") as uploaded:
            if uploaded.read(5) != b"%PDF-":
                raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

        validate_document_size(temp_path)
        docs = load_document(str(temp_path), filename=filename)
        if not docs:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found in the PDF. Scanned PDFs may require OCR.",
            )

        _, chunk_count = create_and_save_vector_store(docs)
        return {
            "filename": filename,
            "size_bytes": total_bytes,
            "status": "Processed successfully",
            "chunks_indexed": chunk_count,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {exc}") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        await file.close()


@app.post("/query")
def query_rag(request: QueryRequest):
    vector_store = load_local_vector_store()
    if vector_store is None:
        raise HTTPException(
            status_code=400, detail="No document index found. Upload a document first."
        )
    try:
        retrieved_docs = retrieve_documents(
            vector_store, request.question, number_of_chunks=request.top_chunks
        )
        answer = generate_rag_response(
            request.question, retrieved_docs, conversation_history=[]
        )
        sources = []
        seen = set()
        for document in retrieved_docs:
            metadata = getattr(document, "metadata", {}) or {}
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")
            key = (str(source), str(page))
            if key not in seen:
                seen.add(key)
                sources.append({"source": source, "page": page})
        return {"answer": answer, "sources": sources}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
