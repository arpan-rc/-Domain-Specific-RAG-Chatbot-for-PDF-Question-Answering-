import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from prompt import RAG_SYSTEM_PROMPT

load_dotenv()

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024


def validate_document_size(file_path):
    """Validate that a file exists, is non-empty, and is at most 100 MiB."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document does not exist: {path}")
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise ValueError("The uploaded document is empty.")
    if size_bytes > MAX_FILE_SIZE_BYTES:
        raise ValueError("Document exceeds the maximum allowed size of 100 MiB.")
    return size_bytes


def retrieve_documents(vector_store, question, number_of_chunks=4):
    """Retrieve relevant chunks, limiting the number of results to 1–10."""
    if not question or not question.strip():
        raise ValueError("Please enter a question.")
    if not isinstance(number_of_chunks, int) or isinstance(number_of_chunks, bool):
        raise ValueError("number_of_chunks must be an integer.")
    number_of_chunks = max(1, min(number_of_chunks, 10))
    return vector_store.similarity_search(query=question.strip(), k=number_of_chunks)


def generate_rag_response(question, retrieved_documents, conversation_history=None):
    """Generate an answer grounded in retrieved document passages."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Please enter a question.")
    documents = list(retrieved_documents or [])
    if not documents:
        return "I couldn't find relevant passages in the processed documents to answer that question."

    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    if not api_key:
        raise ValueError("GROQ_API_KEY was not found in the environment or .env file.")

    context_parts = []
    for index, document in enumerate(documents, start=1):
        metadata = getattr(document, "metadata", {}) or {}
        context_parts.append(
            f"Context {index}\n"
            f"Source: {metadata.get('source', 'Unknown')}\n"
            f"Page: {metadata.get('page', 'Unknown')}\n"
            f"Text:\n{getattr(document, 'page_content', '')}"
        )

    history_lines = []
    for message in (conversation_history or [])[-6:]:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        content = str(message.get("content", "")).strip()
        if content:
            history_lines.append(f"{role}: {content}")
    history = "\nPrevious Conversation:\n" + "\n".join(history_lines) if history_lines else ""

    prompt = RAG_SYSTEM_PROMPT.format(
        context="\n\n".join(context_parts) + history,
        question=question,
    )
    model = ChatGroq(api_key=api_key, model_name=model_name, temperature=0)
    response = model.invoke(prompt)
    return str(response.content).strip()
