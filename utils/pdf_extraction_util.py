import os
import logging
from typing import Optional
import fitz # PyMuPDF for robust PDF text extraction

logger = logging.getLogger('pdf_extraction_util')

def extract_text_from_pdf(pdf_path: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    Returns (text, error) tuple.
    """
    if not os.path.exists(pdf_path):
        return None, f"File not found at path: {pdf_path}"

    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
        
        text = text.strip()

        if not text:
             return None, "PDF text extraction failed: Document appears empty or protected."

        logger.info(f"Successfully extracted {len(text)} characters from {os.path.basename(pdf_path)}")
        return text, None

    except Exception as e:
        error_msg = f"Error extracting text from PDF at {pdf_path}: {str(e)}"
        logger.error(error_msg)
        return None, error_msg