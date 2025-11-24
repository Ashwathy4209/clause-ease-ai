# AI-ClauseEase — Streamlit (Milestone 1 & 2)
This package is a Streamlit implementation of the "Contract Language Simplifier" up to **Milestone 2**:
- User registration & login (token-based)
- Profile page to manage uploaded documents
- Document upload/paste interface
- Text preprocessing (cleaning, sentence segmentation, tokenization)
- Readability metrics: Flesch–Kincaid Grade, Flesch Reading Ease, Gunning Fog

## How to run
1. Create a Python environment (recommended).
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
4. The app uses a local SQLite database at `database/app.db`.

## Notes
- This milestone uses a simple token-based authentication (tokens stored in DB).
- File text extraction supports `.txt` and `.pdf` if `PyPDF2` is installed.
- `python-docx` support for `.docx` is attempted if available; `.odt` support is limited.
- Readability calculations use simple heuristics for syllable counting (no external corpora).
