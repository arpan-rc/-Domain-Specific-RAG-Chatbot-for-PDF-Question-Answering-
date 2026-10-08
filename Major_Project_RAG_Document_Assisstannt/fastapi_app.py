import os
import shutil
from document_loader import load_document
from fastapi import FastAPI,File,HTTPException,UploadFile
from pydantic import BaseModel
from rag_pipeline import generate_rag_response,retrieve_documents
from vector_store import create_and_save_vector_store,load_local_vector_store

app=FastAPI(title = "Domain RAG Chatbot API")

class QueryRequest(BaseModel):
    question: str
    top_chunks: int = 4

@app.get("/")
def read_root():
    return {"message": "Domain RAG Chatbot API is running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are supported."
        )

    os.makedirs("documents",exist_ok=True)
    file_path = f"documents/{file.filename}"

    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    docs = load_document(file_path)
    vector_store,chunk_count = create_and_save_vector_store(docs)

    return {
        "filename": file.filename,
        "status": "Processed successfully",
        "chunks_indexed": chunk_count
    }

@app.post("/query")
def query_rag(request: QueryRequest):
    vector_store = load_local_vector_store()
    if not vector_store:
        raise HTTPException(
            status_code=400,
            detail="No document index found. Upload a document first.",
        )

    retrieved_docs = retrieve_documents(vector_store, request.question,number_of_chunks=request.top_chunks)
    answer = generate_rag_response(request.question,retrieved_docs)

    sources = [
        {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
        for d in retrieved_docs
    ]

    return {"answer": answer, "sources": sources}