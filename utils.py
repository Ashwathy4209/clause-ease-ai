import os
import re
import streamlit as st # Added for secrets check (optional but recommended)

# --- CONFIG CHECKS ---
def is_google_configured():
    """
    Checks if Google Auth is configured via file or secrets.
    Prioritizes local file, falls back to checking existence of secrets.
    """
    if os.path.exists('client_secrets.json'):
        return True
    # If deploying to Streamlit Cloud, you usually use st.secrets instead of a file
    if "google_oauth" in st.secrets: 
        return True
    return False

# --- CONSTANTS ---
# Compiled once at module level for performance
LEGAL_KEYWORDS = [
    'agreement', 'contract', 'party', 'parties', 'whereas', 'heretofore', 
    'hereinafter', 'jurisdiction', 'liability', 'indemnify', 'indemnification', 
    'clause', 'article', 'section', 'subsection', 'governing law', 'termination', 
    'confidentiality', 'intellectual property', 'warranty', 'disclaimer', 
    'force majeure', 'arbitration', 'litigation', 'notwithstanding', 'pursuant', 
    'licensor', 'licensee', 'lessor', 'lessee'
]
# Use word boundaries (\b) to avoid partial matches (e.g., 'party' inside 'repartying')
LEGAL_KEYWORD_PATTERN = re.compile(r'\b(?:' + '|'.join(re.escape(k) for k in LEGAL_KEYWORDS) + r')\b', re.IGNORECASE)

# --- TEXT UTILS ---
def is_likely_legal(text, threshold=0.01):
    """
    Checks text for legal keyword density.
    Returns False if text is too short.
    """
    if not text: 
        return False
    
    # OPTIMIZATION: Calculate word count once to avoid double splitting
    words = text.split()
    word_count = len(words)
    
    if word_count < 50: 
        return False # Return False, not None, to keep return types consistent (boolean)

    matches = LEGAL_KEYWORD_PATTERN.findall(text)
    keyword_density = len(matches) / word_count
    
    return keyword_density >= threshold

def get_word_count(text: str):
    """
    Counts words using textstat with a fallback.
    Safely handles missing library.
    """
    if not text or not isinstance(text, str):
        return 0
        
    try:
        # Lazy load INSIDE try block to prevent crash if library is missing
        import textstat
        return textstat.lexicon_count(text)
    except (ImportError, Exception):
        # Fallback: simple split (less accurate for punctuation but fast)
        return len(text.split())