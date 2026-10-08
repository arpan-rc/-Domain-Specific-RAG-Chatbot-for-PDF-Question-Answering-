import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from document_loader import load_document
from vector_store import create_and_save_vector_store, load_local_vector_store
from rag_pipeline import generate_rag_response

# Configuration & Security Constants (Point 12)
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_TYPES = ["pdf"]

# Streamlit Page Config
st.set_page_config(
    page_title="Domain-Specific RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# Helper function to normalize vector store objects from tuple outputs
def extract_vector_store(store_obj):
    if isinstance(store_obj, tuple):
        return store_obj[0]
    return store_obj

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    raw_store = load_local_vector_store()
    st.session_state.vector_store = extract_vector_store(raw_store)

# UI Title & Responsible AI Disclaimer (Point 12)
st.title("📄 Domain-Specific RAG Chatbot")
st.warning(
    "⚠️ **Responsible AI Disclaimer:** Generated responses are based on retrieved document passages. "
    "Please verify high-stakes or critical information against the original source documents."
)

# Sidebar - Document Operations
with st.sidebar:
    st.header("Document Operations")
    
    uploaded_files = st.file_uploader(
        "Upload Domain PDFs",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        help=f"Maximum file size per document is {MAX_FILE_SIZE_MB}MB."
    )
    
    # File Validation & Confidentiality Warning (Point 12)
    valid_files = []
    if uploaded_files:
        for file in uploaded_files:
            if file.size > MAX_FILE_SIZE_BYTES:
                st.error(f"❌ '{file.name}' exceeds the {MAX_FILE_SIZE_MB}MB size limit.")
            else:
                valid_files.append(file)
        
        st.caption("🔒 *Please ensure you have authorization before uploading internal or confidential documents.*")

    enable_ocr = st.checkbox("Enable OCR for scanned PDFs", value=False)

    if st.button("Process Documents", type="primary"):
        if not valid_files:
            st.warning("Please upload at least one valid PDF document.")
        else:
            with st.spinner("Extracting text and building vector store..."):
                try:
                    # Save files temporarily and extract text
                    temp_dir = "temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    all_documents = []
                    file_paths = []
                    
                    for file in valid_files:
                        temp_path = os.path.join(temp_dir, file.name)
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                        file_paths.append(temp_path)

                        docs = load_document(temp_path)
                        if isinstance(docs, list):
                            all_documents.extend(docs)
                        elif docs:
                            all_documents.append(docs)

                    if not all_documents:
                        st.error("No extractable text found in the uploaded document(s).")
                    else:
                        # Save and extract vector store cleanly
                        raw_store = create_and_save_vector_store(all_documents)
                        st.session_state.vector_store = extract_vector_store(raw_store)
                        st.success("Documents processed and vector database built successfully!")
                    
                    # Clean up temporary upload files
                    for temp_path in file_paths:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
                except Exception as e:
                    st.error(f"An error occurred while processing documents: {str(e)}")

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main Chat Interface
st.subheader("Chat")

# Render Chat History
for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display Sources if available
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.write(f"- **Document:** {source.get('source', 'Unknown')} | **Page:** {source.get('page', 'N/A')}")
        
        # Display Feedback Buttons for Assistant Responses
        if message["role"] == "assistant":
            current_feedback = message.get("feedback")
            
            if current_feedback is None:
                col1, col2, _ = st.columns([1, 1, 10])
                if col1.button("👍", key=f"thumbs_up_{idx}"):
                    st.session_state.chat_history[idx]["feedback"] = "positive"
                    st.toast("Thank you for your feedback!", icon="👍")
                    st.rerun()
                if col2.button("👎", key=f"thumbs_down_{idx}"):
                    st.session_state.chat_history[idx]["feedback"] = "negative"
                    st.toast("Thank you for your feedback!", icon="👎")
                    st.rerun()
            else:
                st.caption(f"Feedback recorded: {'👍 Helpful' if current_feedback == 'positive' else '👎 Needs Improvement'}")

# User Query Input
user_query = st.chat_input("Ask a question about the uploaded documents...")

if user_query:
    if not st.session_state.vector_store:
        st.error("Please upload and process documents first before asking questions.")
    else:
        # Append User Message to UI
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate RAG Response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents & generating answer..."):
                try:
                    vector_db = extract_vector_store(st.session_state.vector_store)

                    # Check if vector_db is a retriever or a vectorstore instance
                    if hasattr(vector_db, "as_retriever"):
                        retriever = vector_db.as_retriever(search_kwargs={"k": 4})
                        retrieved_docs = retriever.invoke(user_query)
                    elif hasattr(vector_db, "get_relevant_documents"):
                        retrieved_docs = vector_db.get_relevant_documents(user_query)
                    elif hasattr(vector_db, "similarity_search"):
                        retrieved_docs = vector_db.similarity_search(user_query, k=4)
                    else:
                        raise AttributeError("Vector store object does not support document retrieval.")

                    # Call RAG Pipeline
                    answer = generate_rag_response(
                        question=user_query,
                        retrieved_documents=retrieved_docs,
                        conversation_history=st.session_state.chat_history
                    )

                    # Extract Source Metadata
                    sources = [
                        {
                            "source": doc.metadata.get("source", "Unknown"),
                            "page": doc.metadata.get("page", "N/A")
                        }
                        for doc in retrieved_docs
                    ]

                    # Display Answer & Sources
                    st.markdown(answer)
                    if sources:
                        with st.expander("View Sources"):
                            for source in sources:
                                st.write(f"- **Document:** {source['source']} | **Page:** {source['page']}")

                    # Store in History
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Failed to generate response: {str(e)}")