import os
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document


def load_document(file_path_or_stream, filename=None, use_ocr_fallback=False):
    """Extract PDF/TXT content and retain source plus one-based page metadata."""
    is_path = isinstance(file_path_or_stream, (str, os.PathLike))
    if is_path:
        path = Path(file_path_or_stream)
        source_name = Path(filename).name if filename else path.name
    else:
        source_name = Path(filename or getattr(file_path_or_stream, "name", "uploaded_file")).name

    file_extension = Path(source_name).suffix.lower()
    documents = []

    if file_extension == ".txt":
        if is_path:
            text = Path(file_path_or_stream).read_text(encoding="utf-8", errors="ignore")
        else:
            raw = file_path_or_stream.getvalue()
            text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
        if text.strip():
            documents.append(Document(page_content=text, metadata={"source": source_name, "page": 1}))
        return documents

    if file_extension != ".pdf":
        raise ValueError("Only PDF and TXT documents are supported.")

    reader_source = str(file_path_or_stream) if isinstance(file_path_or_stream, os.PathLike) else file_path_or_stream
    reader = PdfReader(reader_source)

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # OCR is optional and only supported when a local file path is available.
        if not text.strip() and use_ocr_fallback and is_path:
            try:
                import pytesseract
                from pdf2image import convert_from_path

                images = convert_from_path(
                    str(file_path_or_stream), first_page=page_num, last_page=page_num
                )
                if images:
                    text = pytesseract.image_to_string(images[0]) or ""
            except Exception as exc:
                # Continue with normal extraction if optional OCR tools are unavailable.
                print(f"OCR failed on page {page_num} of {source_name}: {exc}")

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": source_name, "page": page_num},
                )
            )

    return documents
