# Domain-Specific RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using content retrieved from uploaded PDF documents. It provides a Streamlit chat interface and a FastAPI REST API.

<img width="4064" height="1464" alt="image" src="https://github.com/user-attachments/assets/4b283e22-3017-4ec8-89ca-4301e9eb6fcc" />


## Features

- Upload and process PDF documents.
- Extract text from PDFs and split it into overlapping chunks.
- Create semantic embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Store and retrieve document chunks with a local FAISS vector index.
- Generate answers with a Groq-hosted language model.
- Show retrieved source filenames and page metadata in the chat interface/API response.
- Keep recent conversation messages in the Streamlit chat session.
- Provide thumbs-up/thumbs-down feedback controls in the Streamlit UI.

> **Important:** Answers are generated from retrieved document context. Verify important or high-stakes information against the original documents.

## Project structure

Place the following files in the same project directory:

```text
your-project/
├── app.py                 # Streamlit user interface
├── fastapi_app.py         # FastAPI REST API
├── document_loader.py     # PDF/TXT text extraction
├── rag_pipeline.py        # Retrieval and Groq response generation
├── prompt.py              # RAG system prompt and answer rules
├── vector_store.py        # Chunking, embeddings, FAISS save/load
├── requirements.txt       # Python dependencies
├── .env                   # Your local API key and optional model setting
├── documents/             # Created by the API when uploading PDFs
├── temp_uploads/          # Temporary Streamlit uploads
└── vector_store/
    └── saved_index/       # Saved FAISS index (created when documents are processed)
```

The `documents/`, `temp_uploads/`, and `vector_store/` directories are created or populated while the application runs; you generally do not need to create them manually.

## Requirements

- Python 3.10 or newer is a reasonable starting point; use a Python version supported by the installed dependencies.
- A Groq API key.
- Internet access on first run to download the embedding model and to call the Groq API.
- For OCR of scanned PDFs, additional system dependencies may be needed: Tesseract OCR and Poppler (used by `pytesseract` and `pdf2image`).

## 1. Set up the environment

Open a terminal in the directory containing `requirements.txt`.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can use Command Prompt instead:

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

## 2. Configure your Groq API key

Create a file named `.env` in the project root (the same directory as `rag_pipeline.py`):

```dotenv
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

Replace `your_groq_api_key_here` with your actual Groq API key. `GROQ_MODEL` is optional; if omitted, the code defaults to `openai/gpt-oss-20b`.

**Keep `.env` private.** Do not commit API keys to source control or share them in screenshots or logs. Add `.env`, `.venv/`, uploaded documents, and generated vector indexes to `.gitignore` if you use Git.

## 3. Run the Streamlit chatbot

From the project root, with the virtual environment activated:

```bash
streamlit run app.py
```

Streamlit will print a local URL (commonly `http://localhost:8501`). Open that URL in your browser.

### Use the Streamlit interface

1. In the sidebar, under **Document Operations**, upload one or more PDF files.
2. Each file must be no larger than 100 MB in the Streamlit interface.
3. Click **Process Documents** and wait for text extraction, embedding creation, and FAISS index saving to finish.
4. Enter a question in the chat box, such as `Summarize the main points in this document`.
5. Review the answer and open **View Sources** to see the source metadata returned for retrieved passages.
6. Use the thumbs-up/thumbs-down controls to record feedback for an answer.
7. Click **Clear Chat History** to clear the current chat session.

The application retrieves up to four relevant chunks for each question in the Streamlit flow. Process documents before asking questions. Processing a new batch creates and saves a new index at `vector_store/saved_index`; treat it as replacing the previously saved index rather than automatically merging with it.

### Scanned PDFs / OCR

The UI includes an **Enable OCR for scanned PDFs** checkbox, but in the current code its value is not passed to `load_document()`. As a result, checking it does not currently activate OCR. Also, OCR requires Tesseract and Poppler to be installed and available on your system. For image-only PDFs, OCR wiring may need to be added before relying on this option.

## 4. Run the FastAPI REST API (alternative interface)

You can run the API instead of, or alongside, Streamlit. In a second terminal, activate the same virtual environment and run:

```bash
uvicorn fastapi_app:app --reload
```

The API is commonly available at `http://127.0.0.1:8000`. Interactive API documentation is available at:

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

Replace `path/to/your-document.pdf` with the path to a real PDF file. The API returns the filename, processing status, and number of chunks indexed.

The API's upload route accepts PDF filenames ending in `.pdf`. Unlike the Streamlit interface, the current API route does not implement the same 100 MB file-size validation.

### Ask a question

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main findings?",
    "top_chunks": 4
  }'
```

Example response shape:

```json
{
  "answer": "Generated answer based on the retrieved document context.",
  "sources": [
    {
      "source": "your-document.pdf",
      "page": 1
    }
  ]
}
```

The example answer is illustrative; actual answers and source entries depend on the uploaded document and retrieval results. `top_chunks` is optional and defaults to `4`.

You can also send these requests from Swagger UI at `/docs`.

## How the RAG pipeline works

1. **Document loading:** `document_loader.py` extracts text from PDF pages and supports TXT input when called directly.
2. **Chunking:** `vector_store.py` splits text into chunks of 800 characters with 120 characters of overlap.
3. **Embedding:** Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Indexing:** FAISS stores the vectors locally in `vector_store/saved_index`.
5. **Retrieval:** The application searches for chunks similar to the user's question (four by default in the UI/API examples).
6. **Prompt construction:** `prompt.py` instructs the model to answer from the supplied context and to say when the answer cannot be found.
7. **Generation:** `rag_pipeline.py` calls the configured Groq model and returns its response.
8. **Sources:** Retrieved document metadata is included with the response where available.

## Troubleshooting

- **`GROQ Api Key was not found in the .env file`:** Confirm `.env` is in the project root and contains `GROQ_API_KEY=...`. Restart the app after changing environment variables.
- **No document index found:** Upload and process a PDF first. Confirm `vector_store/saved_index/index.faiss` exists.
- **No extractable text found:** The PDF may be image-only or protected. Try a text-searchable PDF; OCR support needs to be wired into the current UI flow and its system dependencies installed.
- **Model or API errors:** Check your internet connection, Groq API key, model availability, and account limits.
- **Dependency installation errors:** Upgrade `pip`, confirm your Python version is compatible, and install any OS-level OCR dependencies separately if you need OCR.
- **Unexpected source page values:** The current `document_loader.py` assigns page metadata as `1` for each extracted PDF page. Source page numbers may therefore be inaccurate until the loader is updated to use the actual page number.

## Security and data handling notes

- Only upload documents you are authorized to use.
- Do not expose your Groq API key.
- Treat uploaded PDFs and the generated FAISS index as potentially sensitive; keep them out of public repositories.
- The FastAPI app currently saves uploaded PDFs under `documents/` and does not include authentication. Do not expose it publicly without adding authentication, upload limits, validation, and other appropriate protections.
- FAISS loading currently uses `allow_dangerous_deserialization=True`. Only load index files that you trust.

## Current limitations

- The Streamlit OCR checkbox is not connected to the document loader yet.
- PDF page metadata currently reports page `1` for each extracted page.
- The API has no authentication and no explicit upload size limit in the current code.
- The saved FAISS index is local to the running project; back it up if you need to preserve it.
- The RAG prompt asks the model to use supplied context only, but generated answers can still be incorrect. Check the original source for important claims.


