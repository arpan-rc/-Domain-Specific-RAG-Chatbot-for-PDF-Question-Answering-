import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from prompt import RAG_SYSTEM_PROMPT

load_dotenv()

def retrieve_documents(vector_store, question, number_of_chunks=4):
    """Retrieve chunks most relevant to the student's question."""

    return vector_store.similarity_search(
        query=question,
        k=number_of_chunks,
    )

def generate_rag_response(question, retrieved_documents, conversation_history):
    """Generate an answer using only the retrieved context."""
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL","openai/gpt-oss-20b")

    if not api_key:
        raise ValueError("GROQ Api Key was not found in the .env file")

    context_parts = []

    for index,document in enumerate(retrieved_documents, start= 1):
        source = document.metadata.get("source","Unknown")
        page = document.metadata.get("page","Unknown")
        context_parts.append(
            f"""Context {index}
        Source: {source}
        Page: {page}
        Text:
        {document.page_content}
        """
        )

    context = "\n\n".join(context_parts)

    #Conversation Memory incorporation
    history_str = ""
    if conversation_history:
        history_str = "\nPrevious Conversation:\n" + "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in conversation_history[-4:]]
        )

    prompt = RAG_SYSTEM_PROMPT.format(
        context = context + history_str, question=question
    )

    model = ChatGroq(
            api_key=api_key,
            model_name=model_name,
            temperature=0,
        )
    
    response = model.invoke(prompt)
    return response.content