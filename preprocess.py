import os
import fitz  # PyMuPDF for PDFs
import docx2txt
import pytesseract
from PIL import Image
import re
import tempfile
from spellchecker import SpellChecker

spell = SpellChecker()

# -------------------------------
#  CLEANING + NORMALIZATION (IMPROVED)
# -------------------------------

def clean_text(text):
    """
    Cleans text by normalizing spacing and removing special characters
    while keeping alphanumeric, hyphens, dots, commas, parentheses, etc.
    """
    # --- THIS IS THE FIX ---
    # Collapse all horizontal whitespace (spaces, tabs, etc.) to a single space
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    
    # Collapse 2 OR MORE newlines into a single paragraph break (2 newlines)
    # This will fix the large gaps from the PDF.
    text = re.sub(r'\n{2,}', '\n\n', text)
    
    # Keep: a-z, A-Z, 0-9, spaces, newlines, and common/legal punctuation
    # Added: ₹, $, #, /, %
    text = re.sub(r'[^a-zA-Z0-9\s\n\-.,()₹$#/%]+', '', text) 
    
    return text.strip() 
    # --- END FIX ---# We will NOT lowercase here. Let the model see case.
    


# -------------------------------
#  SPELL CHECK (protect glossary)
# -------------------------------

def correct_spelling(text, glossary_words=None):
    """
    Performs light spell correction, skipping glossary terms and short tokens.
    """
    glossary_words = set(w.lower() for w in glossary_words) if glossary_words else set()

    corrected_words = []
    for word in text.split():
        lw = word.lower()

        # Skip corrections for glossary terms, short tokens, numeric, etc.
        if lw in glossary_words or not word.isalpha() or len(word) <= 2:
            corrected_words.append(word)
            continue

        corrected = spell.correction(word)
        corrected_words.append(corrected if corrected else word)

    return ' '.join(corrected_words)


# -------------------------------
#  TEXT EXTRACTION FUNCTIONS (PDF Function Replaced)
# -------------------------------

def extract_text_from_pdf(file_path, glossary_words=None):
    """
    Extracts text from a PDF, intelligently skipping headers and footers.
    """
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            page_rect = page.rect
            page_height = page_rect.height
            
            # Define header/footer margins (e.g., 10% of page height)
            header_margin = page_height * 0.10
            footer_margin = page_height * 0.90
            
            blocks = page.get_text("blocks")
            
            main_content_blocks = []
            for b in blocks:
                x0, y0, x1, y1, block_text, _, _ = b
                if y0 > header_margin and y1 < footer_margin:
                    main_content_blocks.append(block_text)
            
            # --- THIS IS THE FIX ---
            # Join blocks with a NEWLINE to preserve paragraph structure
            text += "\n".join(main_content_blocks)
            # --- END FIX ---
    
    cleaned = clean_text(text)
    # Spell correction is disabled as it's unreliable on legal/financial text
    # corrected = correct_spelling(cleaned, glossary_words)
    return cleaned # Return the cleaned text


def extract_text_from_docx(file_path, glossary_words=None):
    text = docx2txt.process(file_path)
    cleaned = clean_text(text)
    # corrected = correct_spelling(cleaned, glossary_words)
    return cleaned


def extract_text_from_image(file_path_or_obj, glossary_words=None):
    image = Image.open(file_path_or_obj)
    text = pytesseract.image_to_string(image)
    cleaned = clean_text(text)
    # corrected = correct_spelling(cleaned, glossary_words)
    return cleaned


def extract_text_from_txt(file_path_or_obj, glossary_words=None):
    if hasattr(file_path_or_obj, "read"):
        text = file_path_or_obj.read().decode('utf-8', errors='ignore')
    else:
        with open(file_path_or_obj, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    cleaned = clean_text(text)
    # corrected = correct_spelling(cleaned, glossary_words)
    return cleaned


# -------------------------------
#  UNIVERSAL FILE HANDLER
# -------------------------------

def extract_text_from_upload(uploaded_file, glossary_words=None, use_ocr=False):
    ext = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    if ext == '.pdf':
        return extract_text_from_pdf(tmp_path, glossary_words)
    elif ext == '.docx':
        return extract_text_from_docx(tmp_path, glossary_words)
    elif ext in ['.png', '.jpg', '.jpeg']:
        return extract_text_from_image(tmp_path, glossary_words)
    elif ext == '.txt':
        return extract_text_from_txt(tmp_path, glossary_words)
    else:
        raise ValueError(f"Unsupported file format: {ext}")