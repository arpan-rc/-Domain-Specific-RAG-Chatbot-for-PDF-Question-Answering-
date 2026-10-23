from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_PATH = Path("vector_store/saved_index")


def create_and_save_vector_store(documents, save_path=DEFAULT_INDEX_PATH):
    """Split extracted documents, preserve metadata, and save a local FAISS index."""
    documents = list(documents or [])
    if not documents:
        raise ValueError("Cannot build a vector index because no document text was extracted.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks were created from the uploaded documents.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(destination))
    return vector_store, len(chunks)


def load_local_vector_store(save_path=DEFAULT_INDEX_PATH):
    """Load an application-controlled local FAISS index, or return None if absent."""
    destination = Path(save_path)
    if not (destination / "index.faiss").is_file() or not (destination / "index.pkl").is_file():
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # LangChain's FAISS docstore is pickled. Only load indexes created and controlled
    # by this application; never load index files supplied by an end user.
    return FAISS.load_local(
        str(destination), embeddings, allow_dangerous_deserialization=True
    )
