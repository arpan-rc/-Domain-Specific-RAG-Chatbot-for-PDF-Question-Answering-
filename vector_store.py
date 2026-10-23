import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def create_and_save_vector_store(
    documents,
    save_path="vector_store/saved_index"
):
    """Split documents, create embeddings, and store them in FAISS."""
    # If saved_index exists as a file instead of a directory, remove it
    if os.path.isfile(save_path):
        os.remove(save_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save vector store natively
    vector_store.save_local(save_path)
    
    return vector_store, len(chunks)

def load_local_vector_store(save_path="vector_store/saved_index"):
    """Loads a previously saved local FAISS vector index."""
    faiss_file = os.path.join(save_path, "index.faiss")
    
    if os.path.exists(faiss_file):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return FAISS.load_local(
            save_path, embeddings, allow_dangerous_deserialization=True
        )
    return None