# ClauseEase AI

ClauseEase AI is an advanced legal document assistant designed to streamline contract analysis and management. It utilizes a hybrid AI architecture, combining the Google Gemini API for deep semantic analysis and risk assessment with local Large Language Models (LLMs) for privacy-focused text simplification.

The application employs a Retrieval-Augmented Generation (RAG) pipeline to ground AI responses in the specific context of uploaded documents, reducing hallucinations and ensuring accurate legal interpretation.

## Key Features

### Google Gemini API Integration (Analysis & RAG)
* **Deep Document Analysis:** Leverages Google's Gemini Pro model to interpret complex clauses, identify obligations, and detect ambiguities.
* **Risk Assessment:** Automatically scans documents to highlight potential legal risks, liabilities, and unfavorable terms.
* **Context-Aware Chat:** Powered by RAG, the chat interface retrieves specific vector embeddings from the document to answer user queries with high precision.

### Local AI Models (Simplification)
* **Text Simplification:** Utilizes fine-tuned local models (BART-Large-CNN and Flan-T5) to rewrite dense legal jargon into plain, understandable English.
* **Offline Processing:** The simplification pipeline runs entirely locally, ensuring that raw text processing for readability does not require external API calls.

### Core System Features
* **Clause Extraction:** Automatically identifies and categorizes critical contract sections (e.g., Termination, Indemnity, Confidentiality).
* **Secure Authentication:** Implements role-based access control (Admin/User) using encrypted session management and JWT tokens.
* **Dashboard Analytics:** Provides visual insights into processed contracts, user activity, and document status.

## Technical Architecture

* **Frontend:** Streamlit
* **Analysis Engine:** Google Gemini API (Cloud)
* **Simplification Engine:** Hugging Face Transformers (Local)
* **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
* **Vector Store:** In-memory vector search for RAG retrieval
* **Database:** SQLite
* **Backend:** Python

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Ashwathy4209/clause-ease-ai.git
cd clause-ease-ai
```

### 2. Set Up Virtual Environment
It is recommended to use an isolated Python environment to manage dependencies.

**Windows:**
```bash
python -m venv clause_env
.\clause_env\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv clause_env
source clause_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Local Models
This application requires local models to function. Run the included setup script to download the necessary weights (approx. 2GB) to the `models_cache/` directory.

```bash
python setup_models.py
```

### 5. Configuration
Create a `.env` file in the root directory. You must configure your API keys for the analysis engine to function.

**File: .env**
```ini
# Security
JWT_SECRET_KEY=your_generated_secure_key
ADMIN_PASSWORD=your_admin_password

# External APIs
GEMINI_API_KEY=your_google_gemini_api_key
```

## Running the Application

Once installation and configuration are complete, launch the application using Streamlit:

```bash
streamlit run app.py
```

## Project Structure

* **app.py**: Main entry point for the application.
* **views/**: Contains modular UI components (Dashboard, Chat, Upload, Admin).
* **db/**: Database management scripts, schema definitions, and authentication logic.
* **models.py**: Handles interactions with the Google Gemini API (Analysis) and Local Models (Simplification).
* **processing.py**: Core logic for text extraction, chunking, and RAG retrieval.
* **setup_models.py**: Utility script to automate the download of required Hugging Face models.
* **models_cache/**: Local storage for downloaded model weights (excluded from version control).
