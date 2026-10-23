# Domain-Specific RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using passages retrieved from uploaded PDF documents. It provides a Streamlit chat interface and a FastAPI REST API.

## Features

- Upload PDF files up to **100 MiB per file** through the Streamlit UI and FastAPI upload endpoint.
- Extract text page by page and preserve the original filename and **one-based PDF page number** in source metadata.
- Optionally run OCR on scanned/image-only PDF pages in Streamlit (requires Tesseract OCR and Poppler).
- Split extracted text into overlapping chunks and create embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Store and retrieve document chunks with a local FAISS index.
- Generate answers using a Groq-hosted language model.
- Return source filenames and page numbers with answers where available.
- Keep recent conversation messages in the Streamlit session and provide answer feedback controls.

> **Important:** Generated answers can be incorrect. Verify important or high-stakes information against the original documents.

## Project structure

Keep the application files together in the project root:

```text
project/
├── app.py                 # Streamlit user interface
├── fastapi_app.py         # FastAPI REST API
├── document_loader.py     # PDF/TXT extraction and page metadata
├── rag_pipeline.py        # Retrieval, validation, and Groq response generation
├── prompt.py              # RAG system prompt and answer rules
├── vector_store.py        # Chunking, embeddings, and FAISS save/load
├── requirements.txt       # Python dependencies
├── .env                   # Local API key and optional model setting (do not commit)
├── documents/             # Temporary API upload location
├── temp_uploads/          # Temporary Streamlit uploads
└── vector_store/
    └── saved_index/       # Generated FAISS index files
```

The `documents/`, `temp_uploads/`, and `vector_store/saved_index/` directories are created or populated while the application runs. Do not commit uploaded documents, secrets, or generated indexes unless you have a specific, safe reason to do so.

## Requirements

- Python 3.10 or newer, using a version supported by the installed dependencies.
- A Groq API key.
- Internet access for the initial embedding-model download and Groq API requests.
- For OCR: Tesseract OCR and Poppler installed and available on your system PATH.
- `python-multipart` is included in `requirements.txt` for FastAPI file uploads.

## 1. Set up the environment

Open a terminal in the directory containing `requirements.txt`.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Configure the Groq API key

Create a `.env` file in the project root (the same directory as `rag_pipeline.py`):

```dotenv
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

Replace the placeholder with your actual Groq API key. `GROQ_MODEL` is optional; if omitted, the code uses `openai/gpt-oss-20b`.

**Keep `.env` private.** Never commit API keys or share them in screenshots or logs. Add `.env`, `.venv/`, uploaded documents, and generated vector indexes to `.gitignore`.

## 3. Run the Streamlit chatbot

From the project root, with the virtual environment activated:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (commonly `http://localhost:8501`).

### Use the Streamlit interface

1. In the sidebar, under **Document Operations**, upload one or more PDF files.
2. Each PDF must be no larger than 100 MiB.
3. Optionally select **Enable OCR for scanned PDFs** if the PDF contains image-only pages and OCR dependencies are installed.
4. Click **Process Documents** and wait for extraction, embedding generation, and index saving to finish.
5. Ask a question in the chat box.
6. Open **View Sources** to inspect the filenames and page numbers associated with retrieved passages.
7. Use the thumbs-up/thumbs-down controls to record feedback, or **Clear Chat History** to clear the current session.

The Streamlit retrieval flow requests up to four relevant chunks per question. Processing a new batch **replaces** the saved index; it does not automatically merge the new PDFs with previously indexed PDFs.

### Page numbers and existing indexes

The document loader now records the actual PDF page number, starting at page 1, rather than assigning page 1 to every page. Chunking preserves this metadata.

**Reprocess your PDFs after installing these changes.** An existing FAISS index may still contain the old, incorrect page metadata. Re-upload and process the documents to rebuild it.

## 4. Run the FastAPI REST API

In a second terminal, activate the same virtual environment and run:

```bash
uvicorn fastapi_app:app --reload
```

The API is commonly available at `http://127.0.0.1:8000`. Interactive documentation is available at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Check that the API is running |
| `POST` | `/upload` | Upload and index a PDF |
| `POST` | `/query` | Ask a question against the saved index |

### Upload a PDF

Using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@path/to/your-document.pdf"
```

Replace the example path with a real PDF file. The API streams the upload to a temporary file, rejects uploads larger than 100 MiB, checks that the file begins with a PDF signature, extracts text, and builds the saved index. The temporary upload file is removed after processing.

Example response shape:

```json
{
  "filename": "your-document.pdf",
  "size_bytes": 123456,
  "status": "Processed successfully",
  "chunks_indexed": 12
}
```

The values above are illustrative; the actual response depends on the uploaded file.

### Ask a question

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main findings?",
    "top_chunks": 4
  }'
```

The response includes an `answer` and a `sources` list containing source filenames and page numbers where available. `top_chunks` defaults to 4 and is limited to values from 1 to 10.

You can also send requests through Swagger UI at `/docs`.

## How the RAG pipeline works

1. **Document loading:** `document_loader.py` extracts text from PDF pages and supports TXT input when called directly.
2. **Metadata:** Each extracted page records its source filename and one-based page number.
3. **Chunking:** `vector_store.py` splits text into chunks of 800 characters with 120 characters of overlap and preserves document metadata.
4. **Embedding:** Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2`.
5. **Indexing:** FAISS saves vectors and document metadata locally under `vector_store/saved_index/`.
6. **Retrieval:** The app searches for chunks similar to the user's question.
7. **Generation:** `rag_pipeline.py` builds a prompt using retrieved context and calls the configured Groq model.
8. **Sources:** Retrieved metadata is shown in Streamlit or returned by the API.

## Troubleshooting

- **Missing Groq API key:** Check that `.env` is in the project root and contains `GROQ_API_KEY=...`. Restart the application after changing environment variables.
- **No document index found:** Process a PDF first and confirm that `vector_store/saved_index/index.faiss` and `index.pkl` exist.
- **No extractable text:** The PDF may be image-only, encrypted, or malformed. Enable OCR in Streamlit for scanned PDFs and confirm that Tesseract and Poppler are installed.
- **Incorrect page numbers after updating:** Reprocess the PDFs so the saved FAISS index is rebuilt with corrected metadata.
- **Upload rejected:** Confirm the file is a valid PDF and no larger than 100 MiB.
- **Model/API errors:** Check your internet connection, Groq API key, model name/availability, and account limits.
- **Dependency installation errors:** Upgrade `pip`, check your Python version, and install OCR system dependencies separately if needed.

## Security and data handling

- Upload only documents you are authorized to process.
- Never commit your `.env` file or expose your Groq API key.
- Keep uploaded PDFs and FAISS index files private if they contain sensitive information.
- The FastAPI app does **not** provide authentication or per-user index isolation. Do not expose it to untrusted users or the public internet without adding authentication, authorization, rate limits, and appropriate storage isolation.
- The local FAISS loader uses LangChain's `allow_dangerous_deserialization=True` because its document store is serialized. Only load index files created and controlled by this application; never load an untrusted index.
- The current implementation uses a shared local index path. In a multi-user deployment, users may overwrite or query the same index. Add per-user/session isolation before deploying it for multiple users.

## Current limitations

- The index is local to the running project and is replaced when a new batch is processed. 8
- The API currently has no authentication, conversation-session memory, or per-user index isolation.
- OCR depends on external system packages and may fail if they are unavailable.

## License

This project includes a `LICENSE` file containing the **MIT License**. See [LICENSE](LICENSE) for the full terms. Retain the license and copyright notice when redistributing this software.
