# document_loader.py
# pip install PyMuPDF
import fitz  # PyMuPDF
import os

def load_pdf(file_path: str) -> str:
    """Extract text from a single PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        page_text = page.get_text("text").strip()
        if page_text:
            text += page_text + "\n"
    doc.close()
    return text

def save_all_pdfs_to_txt(folder_path: str):
    """Combine all PDF text from a folder into a single text file."""
    all_text = ""
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            print(f"📂 Loading: {file_name}")
            pdf_text = load_pdf(file_path)
            all_text += pdf_text + "\n\n"  # separate PDFs with newlines

    return all_text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Split text into overlapping chunks."""
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks


