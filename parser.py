import fitz
import io
from PIL import Image
import numpy as np
import easyocr
import streamlit as st

@st.cache_resource
def get_ocr_reader():
    # easyocr supports multiple languages, adding en
    return easyocr.Reader(['en'])

def extract_content(file_bytes: bytes, filename: str) -> tuple[str, bool]:
    """
    Extracts text from a document locally.
    Uses PyMuPDF for digital text.
    For images/scans, uses EasyOCR.
    Returns (extracted_text, is_readable).
    """
    ext = filename.lower().split('.')[-1]
    
    try:
        if ext == 'txt':
            text = file_bytes.decode('utf-8', errors='ignore')
            return text, True
            
        elif ext == 'pdf':
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = ""
            for page in doc:
                page_text = page.get_text("text")
                text += page_text
            
            # Digital PDF branch
            if len(text.strip()) > 50:
                return text, True
            
            # Scanned PDF branch (not enough digital text)
            full_transcription = ""
            for i, page in enumerate(doc):
                pix = page.get_pixmap()
                img_data = pix.tobytes("jpeg")
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                transcription, is_readable = _process_image_local(img)
                if not is_readable:
                    return f"Page {i+1} of {filename} is UNREADABLE.", False
                full_transcription += f"\n[Page {i+1}]\n{transcription}"
                
            return full_transcription, True
            
        elif ext in ['png', 'jpg', 'jpeg']:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            return _process_image_local(img)
            
        else:
            return f"Unsupported file type: {ext}", False
            
    except Exception as e:
        return f"Error parsing {filename}: {str(e)}", False

def _process_image_local(img: Image.Image) -> tuple[str, bool]:
    try:
        reader = get_ocr_reader()
        # easyocr expects numpy array
        img_np = np.array(img)
        # detail=0 returns just the text
        result = reader.readtext(img_np, detail=0)
        
        if not result or len(result) == 0:
            return "UNREADABLE", False
            
        text = " ".join(result)
        # Simple local legibility gatekeeper based on character count
        if len(text.strip()) < 5:
            return "UNREADABLE", False
            
        return text, True
        
    except Exception as e:
        print(f"OCR error: {e}")
        return "Error analyzing image locally.", False
