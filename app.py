import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from document_loader import load_document
from vector_store import create_and_save_vector_store, load_local_vector_store
from rag_pipeline import generate_rag_response

load_dotenv()

MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_TYPES = ["pdf"]
RETRIEVAL_K = 4
TEMP_UPLOAD_DIR = Path("temp_uploads")

st.set_page_config(
    page_title="Domain-Specific RAG Chatbot",
    page_icon="📄",
    layout="wide",
)


def extract_vector_store(store_obj):
    """Support helpers that return a store directly or (store, extra_info)."""
    return store_obj[0] if isinstance(store_obj, tuple) else store_obj


def get_source_metadata(document):
    metadata = getattr(document, "metadata", {}) or {}
    return {
        "source": metadata.get("source") or "Unknown",
        "page": metadata.get("page") if metadata.get("page") is not None else "Unknown",
    }


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    try:
        st.session_state.vector_store = extract_vector_store(load_local_vector_store())
    except Exception as exc:
        st.session_state.vector_store = None
        st.warning(f"Could not load the saved document index: {exc}")

st.title("📄 Domain-Specific RAG Chatbot")
st.warning(
    "⚠️ **Responsible AI Disclaimer:** Responses are generated from retrieved document passages. "
    "Verify important or high-stakes information against the original documents."
)

with st.sidebar:
    st.header("Document Operations")
    uploaded_files = st.file_uploader(
        "Upload Domain PDFs",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        help=f"Maximum size is {MAX_FILE_SIZE_MB} MiB per PDF.",
    )

    valid_files = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.size == 0:
                st.error(f"❌ '{uploaded_file.name}' is empty.")
            elif uploaded_file.size > MAX_FILE_SIZE_BYTES:
                st.error(
                    f"❌ '{uploaded_file.name}' exceeds the "
                    f"{MAX_FILE_SIZE_MB} MiB per-file limit."
                )
            else:
                valid_files.append(uploaded_file)
        st.caption(
            "🔒 Upload only documents you are authorized to use. "
            "Avoid uploading confidential information to an unapproved deployment."
        )

    enable_ocr = st.checkbox("Enable OCR for scanned PDFs", value=False)

    if st.button("Process Documents", type="primary"):
        if not valid_files:
            st.warning("Please upload at least one valid PDF within the size limit.")
        else:
            TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            temporary_paths = []
            try:
                all_documents = []
                with st.spinner("Extracting text and building vector store..."):
                    for uploaded_file in valid_files:
                        # Never use an untrusted client filename as a filesystem path.
                        fd, temp_name = tempfile.mkstemp(
                            prefix="upload_", suffix=".pdf", dir=str(TEMP_UPLOAD_DIR)
                        )
                        temp_path = Path(temp_name)
                        temporary_paths.append(temp_path)
                        with os.fdopen(fd, "wb") as output:
                            output.write(uploaded_file.getbuffer())

                        if temp_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                            st.warning(f"Skipped '{uploaded_file.name}': exceeds the size limit.")
                            continue
                        with temp_path.open("rb") as source_file:
                            if source_file.read(5) != b"%PDF-":
                                st.warning(f"Skipped '{uploaded_file.name}': file is not a valid PDF.")
                                continue

                        docs = load_document(
                            str(temp_path),
                            filename=Path(uploaded_file.name).name,
                            use_ocr_fallback=enable_ocr,
                        )
                        if docs:
                            if not isinstance(docs, list):
                                docs = [docs]
                            for document in docs:
                                metadata = getattr(document, "metadata", {}) or {}
                                metadata["source"] = Path(uploaded_file.name).name
                                document.metadata = metadata
                            all_documents.extend(docs)

                if not all_documents:
                    st.error(
                        "No extractable text was found. Check the PDFs or enable OCR "
                        "for scanned documents."
                    )
                else:
                    raw_store = create_and_save_vector_store(all_documents)
                    new_store = extract_vector_store(raw_store)
                    if new_store is None:
                        raise RuntimeError("Vector-store creation returned no usable index.")
                    st.session_state.vector_store = new_store
                    st.success(f"Processed documents and indexed {len(all_documents)} extracted page(s).")
                    st.info(
                        "The saved index has been rebuilt. Page citations now use the page metadata "
                        "from the newly processed PDFs."
                    )
            except Exception as exc:
                st.error(f"An error occurred while processing documents: {exc}")
            finally:
                for temp_path in temporary_paths:
                    try:
                        temp_path.unlink(missing_ok=True)
                    except OSError:
                        st.warning(f"Could not remove temporary file: {temp_path.name}")

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

st.subheader("Chat")

for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.write(
                        f"- **Document:** {source.get('source', 'Unknown')} "
                        f"| **Page:** {source.get('page', 'Unknown')}"
                    )
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
                label = "👍 Helpful" if current_feedback == "positive" else "👎 Needs Improvement"
                st.caption(f"Feedback recorded: {label}")

user_query = st.chat_input("Ask a question about the uploaded documents...")

if user_query:
    vector_db = extract_vector_store(st.session_state.vector_store)
    if vector_db is None:
        st.error("Please upload and process documents first before asking questions.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents & generating answer..."):
                try:
                    if hasattr(vector_db, "as_retriever"):
                        retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVAL_K})
                        retrieved_docs = retriever.invoke(user_query)
                    elif hasattr(vector_db, "get_relevant_documents"):
                        retrieved_docs = vector_db.get_relevant_documents(user_query)
                    elif hasattr(vector_db, "similarity_search"):
                        retrieved_docs = vector_db.similarity_search(user_query, k=RETRIEVAL_K)
                    else:
                        raise AttributeError("Vector store object does not support document retrieval.")

                    answer = generate_rag_response(
                        question=user_query,
                        retrieved_documents=retrieved_docs,
                        conversation_history=st.session_state.chat_history[:-1],
                    )

                    sources = []
                    seen = set()
                    for document in retrieved_docs:
                        source = get_source_metadata(document)
                        key = (str(source["source"]), str(source["page"]))
                        if key not in seen:
                            seen.add(key)
                            sources.append(source)

                    st.markdown(answer)
                    if sources:
                        with st.expander("View Sources"):
                            for source in sources:
                                st.write(
                                    f"- **Document:** {source['source']} "
                                    f"| **Page:** {source['page']}"
                                )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as exc:
                    st.error(f"Failed to generate response: {exc}")
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": "Sorry, I couldn't generate an answer because an error occurred.",
                            "sources": [],
                        }
                    )
