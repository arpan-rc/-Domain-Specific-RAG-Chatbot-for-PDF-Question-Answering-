import os
import fitz #PyMuPDF
from pypdf import PdfReader
from langchain_core.documents import Document

def load_document(file_path_or_stream,filename=None,use_ocr_fallback=False):
    """
    Loads text from PDF or TXT sources (file paths or Streamlit uploaded file objects)
    and returns a list of LangChain Document objects with page/source metadata.
    """
    documents = []

    # 1. Determine filename and extension
    if isinstance(file_path_or_stream,str):
        source_name = filename or os.path.basename(file_path_or_stream)
        file_extension = source_name.lower().split(".")[-1]
    else:
        source_name = filename or getattr(file_path_or_stream, "name", "uploaded_file")
        file_extension = source_name.lower().split(".")[-1]

    #2. Process TXT files
    if file_extension == "txt":
        if isinstance(file_path_or_stream,str):
            with open(file_path_or_stream,"r",encoding="utf-8",errors="ignore") as f:
                text = f.read()
        else:
            text = file_path_or_stream.getvalue().decode("utf-8",errors="ignore")

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source":source_name,
                        "page":1
                    }
                )
            )

    #3. Process PDF files
    elif file_extension == "pdf":
        reader = PdfReader(file_path_or_stream)

        for page_num, page in enumerate(reader.pages,start=1):
            text = page.extract_text() or ""

            #OCR Fallback for scanned/image pages
            if not text.strip() and use_ocr_fallback and isinstance(file_path_or_stream,str):
                try:
                    import pytesseract
                    from pdf2image import convert_from_path

                    images = convert_from_path(
                        file_path_or_stream,first_page=page_num,last_page=page_num
                    )
                    if images:
                        text = pytesseract.image_to_string(images[0])
                except Exception:
                    text = ""

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                        "source":source_name,
                       "page":1
                        }
                    )
                )

    return documents