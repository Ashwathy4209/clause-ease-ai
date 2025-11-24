# ClauseEase AI

ClauseEase AI is an advanced legal document assistant designed to streamline contract analysis. It utilizes a hybrid AI architecture, combining the Google Gemini API for deep legal analysis with local LLMs for text simplification.

The application allows users to upload contracts, simplifies complex legal jargon using local models (BART/T5), and offers a conversational interface (RAG) to analyze specific contract details using the Gemini API.

## Key Features

### Google Gemini API Integration (Analysis Only)
* **Deep Document Analysis:** Uses the Gemini Pro model to interpret the meaning of clauses and identify legal obligations.
* **Risk Assessment:** Analyzes the document to highlight potential risks, liabilities, and unfavorable terms.
* **Contextual Querying:** Powers the Chat interface to answer complex legal questions based on the document context.

### Local AI Models (Simplification Only)
* **Text Simplification:** Utilizes fine-tuned local models (BART-Large-CNN and Flan-T5) to rewrite complex legal jargon into plain, understandable English.
* **Privacy-Focused Processing:** Text simplification is performed entirely offline, ensuring no data leaves the local environment for this specific task.

### Core Functionality
* **RAG Architecture:** Implements Retrieval-Augmented Generation to ground AI responses in the specific text of the uploaded contract.
* **Clause Extraction:** Automatically identifies, categorizes, and extracts critical clauses (e.g., Termination, Liability, Indemnity).
* **Secure Authentication:** Role-based access control (Admin/User) with encrypted session management.
* **Dashboard Analytics:** Visual overview of processed contracts and user activity.

## Technical Architecture

* **Frontend:** Streamlit
* **Analysis Engine:** Google Gemini API (Cloud)
* **Simplification Engine:** Hugging Face Transformers (Local)
* **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
* **Database:** SQLite
* **Backend:** Python

## Installation and Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Ashwathy4209/clause-ease-ai.git](https://github.com/Ashwathy4209/clause-ease-ai.git)
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
This application requires local models for the simplification features. You must run the included setup script to download them to the `models_cache/` directory.

```bash
python setup_models.py
```

### 5. Configuration
Create a `.env` file in the root directory. You must configure your API keys for the analysis engine to function.

**File: .env**
```ini
# Security
JWT_SECRET_KEY=your_generated_secret_key
ADMIN_PASSWORD=your_admin_password

# Analysis Engine
GEMINI_API_KEY=your_google_gemini_api_key
```

## Running the Application

Once installation and configuration are complete, launch the application:

```bash
streamlit run app.py
```

## Project Structure

* **app.py**: Main application entry point.
* **views/**: UI modules for Dashboard, Chat, and Document Upload.
* **db/**: Database management and authentication logic.
* **models.py**: Logic for interacting with Google Gemini (Analysis) and Local Models (Simplification).
* **processing.py**: Text extraction and RAG retrieval logic.
* **setup_models.py**: Utility script to download local simplification models.
* **models_cache/**: Storage for local model weights (excluded from version control).
