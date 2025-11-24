# ClauseEase AI

ClauseEase AI is an advanced legal document assistant designed to streamline contract analysis. It utilizes a Retrieval-Augmented Generation (RAG) architecture and the Google Gemini API to provide deep semantic understanding, clause extraction, and natural language querying of legal documents.

The application allows users to upload PDF or text contracts, simplifies complex legal jargon using local LLMs, and offers a conversational interface to query specific contract details with high accuracy.

## Key Features

### RAG-Based Chatbot
* **Context-Aware Querying:** Utilizes a Retrieval-Augmented Generation (RAG) pipeline to retrieve relevant document chunks before generating responses.
* **Vector Search:** Implements semantic search using local embeddings to ground answers strictly in the document text, minimizing hallucinations.
* **Source Attribution:** Capable of referencing specific sections of the contract used to derive answers.

### Google Gemini API Integration
* **Advanced Legal Analysis:** Leverages Google's Gemini Pro model for high-level reasoning, risk assessment, and summary generation.
* **Ambiguity Resolution:** Analyzes complex and ambiguous legal phrasing to provide clear interpretations.
* **Strategic Insights:** Identifies potential risks and favorable clauses based on broader legal contexts.

### Document Processing & Management
* **Automated Simplification:** Converts complex legal text into plain English using fine-tuned local models (BART/T5).
* **Clause Extraction:** Automatically identifies, categorizes, and extracts critical clauses (e.g., Termination, Liability, Indemnity).
* **Secure Authentication:** Implements role-based access control (Admin/User) with encrypted session management and JWT protection.
* **Dashboard Analytics:** Visual overview of processed contracts, user activity, and document status.

## Technical Architecture

* **Frontend:** Streamlit
* **LLM Engine:** Hybrid approach using Google Gemini API (Cloud) and Hugging Face Transformers (Local).
* **Embedding Model:** Sentence-Transformers (all-MiniLM-L6-v2).
* **Vector Store:** In-memory vector search for RAG implementation.
* **Database:** SQLite for user management and metadata storage.
* **Backend:** Python.

## Installation and Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Ashwathy4209/clause-ease-ai.git](https://github.com/Ashwathy4209/clause-ease-ai.git)
cd clause-ease-ai
