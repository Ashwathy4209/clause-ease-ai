# models.py
# (FINAL ROBUST VERSION v13)
# ─────────────────────────────────────────────────────────────
# 1. ARCHITECTURE: Full "Heavy" error handling & path verification preserved.
# 2. CHATBOT: Strictly Local (FAISS + FLAN-T5). No Gemini for chat.
#    - FIX: Added 'repetition_penalty' to stop "Jones a.k.a" loops.
# 3. SIMPLIFICATION: Local (Recursive Chunking for Speed).
# 4. REPORTS: Gemini 2.0 Flash for Risks, Issues, Glossary, and Graph Data.
# ─────────────────────────────────────────────────────────────

import os
import glob
import torch
import numpy as np
import re
import json
import nltk
import streamlit as st  # Required for caching
from dotenv import load_dotenv
import google.generativeai as genai

# --- TRANSFORMERS & PIPELINE IMPORTS ---
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from sentence_transformers import SentenceTransformer

# --- LANGCHAIN IMPORTS ---
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

# ════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & SETUP
# ════════════════════════════════════════════════════════════════

# Load Environment Variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini ONLY if key exists (Strictly for Reports/Graph)
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("CRITICAL WARNING: GOOGLE_API_KEY not found. Reports, Glossary, and Graph features will FAIL.")

# Download NLTK Data (Safety Check)
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt')
        nltk.download('punkt_tab')
download_nltk_data()

# ════════════════════════════════════════════════════════════════
# 2. ROBUST LOCAL CACHE MODEL PATHS
# ════════════════════════════════════════════════════════════════

BASE_PATH = os.path.join(os.getcwd(), "models_cache")

def find_snapshot_dir(model_folder):
    """Finds the actual snapshot directory containing model files."""
    snapshots_dir = os.path.join(BASE_PATH, model_folder, "snapshots")
    if os.path.isdir(snapshots_dir):
        matches = glob.glob(os.path.join(snapshots_dir, "*"))
        if matches and os.path.isdir(matches[0]):
            return matches[0]
    return os.path.join(BASE_PATH, model_folder)

MODEL_PATHS = {
    "distilbart": find_snapshot_dir("models--sshleifer--distilbart-cnn-12-6"),
    "bart_large": find_snapshot_dir("models--facebook--bart-large-cnn"),
    "flan_t5": find_snapshot_dir("models--google--flan-t5-base"),
    "embed": find_snapshot_dir("models--sentence-transformers--all-MiniLM-L6-v2"),
}

# Global dictionary to map friendly names to HuggingFace IDs for fallback
HF_MODEL_IDS = {
    "DistilBART": "sshleifer/distilbart-cnn-12-6",
    "BART-Large": "facebook/bart-large-cnn",
    "FLAN-T5": "google/flan-t5-base",
    "FLAN-T5-RAG": "google/flan-t5-base"
}

# ════════════════════════════════════════════════════════════════
# 3. CACHED MODEL LOADING (ROBUST + LOOP FIX)
# ════════════════════════════════════════════════════════════════

# We use @st.cache_resource to load models ONCE into RAM.
@st.cache_resource(show_spinner=False)
def load_pipeline_cached(model_choice: str):
    """
    Loads and caches the model pipeline. 
    Includes robust path checking and fallback logic.
    FIX: Adds repetition penalties to prevent Chatbot loops.
    """
    print(f"⚡ Initializing Model Pipeline for: {model_choice} ...")
    try:
        task = "summarization"
        model_dir = None

        # 1. Determine Local Path
        if model_choice == "DistilBART":
            model_dir = MODEL_PATHS["distilbart"]
        elif model_choice == "BART-Large":
            model_dir = MODEL_PATHS["bart_large"]
        elif model_choice == "FLAN-T5" or model_choice == "FLAN-T5-RAG":
            model_dir = MODEL_PATHS["flan_t5"]
            task = "text2text-generation"
        else:
            return f"Error: Unknown model choice '{model_choice}'"
        
        # 2. Validate Local Path / Fallback to HF Hub
        use_local = True
        if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
             print(f"⚠️ Local cache missing for {model_choice} at {model_dir}. Attempting download from HuggingFace...")
             use_local = False
             model_dir = HF_MODEL_IDS.get(model_choice) # Use ID string instead of path

        # 3. Load Tokenizer & Model
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        
        # 4. GPU / Device Handling
        device = 0 if torch.cuda.is_available() else -1
        if device == 0:
            print(f"✅ Loaded {model_choice} on GPU (CUDA).")
        else:
            print(f"⚠️ Loaded {model_choice} on CPU (slower).")

        # 5. Create Pipeline (WITH REPETITION PENALTY FIX)
        pipe = pipeline(
            task, 
            model=model, 
            tokenizer=tokenizer, 
            device=device, 
            model_kwargs={
                "low_cpu_mem_usage": True,
                # --- CRITICAL FIX FOR CHATBOT LOOPING ---
                "repetition_penalty": 1.15, 
                "no_repeat_ngram_size": 3,
                "early_stopping": True
            } 
        )
        return pipe

    except Exception as e:
        error_msg = f"Error: Failed to load model '{model_choice}'. Details: {e}"
        print(error_msg)
        return error_msg

# ════════════════════════════════════════════════════════════════
# 4. TEXT SIMPLIFICATION (LOGIC OPTIMIZED FOR SPEED)
# ════════════════════════════════════════════════════════════════

def simplify_text(text: str, model_choice: str = "DistilBART", level: str = "Intermediate") -> str:
    """
    Simplify or summarize text using LOCAL MODELS ONLY.
    OPTIMIZATION: Uses Recursive Chunking (500 chars) & Greedy Decoding.
    """
    try:
        # Load from Cache
        pipe = load_pipeline_cached(model_choice)
        
        if isinstance(pipe, str) and pipe.startswith("Error:"): 
            return pipe
        if not text or not text.strip(): 
            return "Error: Input text is empty."

        # --- SPEED OPTIMIZATION: Chunking ---
        # Using RecursiveCharacterTextSplitter is 10x faster than NLTK sentence tokenization
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Optimal size for speed vs context
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)
        
        outputs = []

        for i, chunk in enumerate(chunks):
            # Skip tiny chunks (noise)
            if len(chunk) < 30: 
                continue

            # --- FLAN-T5 LOGIC (Simplification) ---
            if pipe.task == "text2text-generation":
                input_len = len(chunk.split())
                
                # Dynamic Length Ratios
                min_len = max(10, int(input_len * 0.6)) 
                max_len = max(min_len + 20, int(input_len * 1.3))
                
                # Level-Specific Prompts
                if level == "Basic":
                    prompt = f"Explain this legal text simply for a 10-year old:\n{chunk}"
                elif level == "Advanced":
                    prompt = f"Rewrite professionally, keeping legal terms:\n{chunk}"
                else: # Intermediate
                    prompt = f"Simplify this text into plain English:\n{chunk}"

                try:
                    # GREEDY DECODING (num_beams=1) -> 4x Faster than Beam Search
                    output = pipe(
                        prompt, 
                        max_new_tokens=max_len, 
                        min_new_tokens=min_len, 
                        num_beams=1, 
                        do_sample=False 
                    )
                    if output and isinstance(output, list) and 'generated_text' in output[0]:
                        outputs.append(output[0]['generated_text'])
                    else:
                        outputs.append(chunk)
                except Exception as e:
                    print(f"Chunk {i} Error: {e}")
                    outputs.append(chunk)

            # --- DISTILBART LOGIC (Summarization) ---
            else:
                try:
                    # DistilBART is purely for summarization
                    output = pipe(
                        chunk, 
                        max_new_tokens=250, 
                        min_new_tokens=30, 
                        num_beams=1, # Greedy decoding for speed
                        do_sample=False
                    )
                    if output and isinstance(output, list) and 'summary_text' in output[0]:
                        outputs.append(output[0]['summary_text'])
                    else:
                        outputs.append(chunk)
                except Exception as e:
                    print(f"Chunk {i} Error: {e}")
                    outputs.append(chunk)
        
        # Clean up results
        simplified = " ".join(outputs)
        cleaned_simplified = re.sub(r'\n\s*\n', '\n\n', simplified)
        return cleaned_simplified.strip()

    except Exception as e:
        error_msg = f"Error: Simplification failed. Details: {e}"
        print(error_msg)
        return error_msg

# ════════════════════════════════════════════════════════════════
# 5. GEMINI REPORTS (ANALYSIS, GLOSSARY & GRAPHS)
# ════════════════════════════════════════════════════════════════

def get_ai_analysis(text: str, analysis_type: str = "legal_issues") -> list:
    """
    Uses Gemini 2.0 Flash to generate analysis lists (Risks, Issues, etc).
    """
    if not GOOGLE_API_KEY:
        return ["Error: Google API Key missing."]
        
    try:
        input_text = text[:25000] # Safe context limit
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Specific Prompts for Report UI
        if analysis_type == "legal_risks":
            prompt = f"Analyze this LEGAL document. Identify exactly 3 major liabilities or risks. Return a list of 3 items. No numbering. TEXT: {input_text}"
        elif analysis_type == "legal_issues":
            prompt = f"Analyze this LEGAL document. Identify exactly 3 missing clauses or ambiguities. Return a list of 3 items. No numbering. TEXT: {input_text}"
        elif analysis_type == "technical_summary":
            prompt = f"Analyze this TECHNICAL document. Provide exactly 3 bullet points summarizing the methodology. Return a list of 3 items. No numbering. TEXT: {input_text}"
        else: 
            prompt = f"Analyze this document. Identify exactly 3 key concepts explained. Return a list of 3 items. No numbering. TEXT: {input_text}"

        response = model.generate_content(prompt)
        
        # Clean output list
        items = response.text.split('\n')
        clean_items = [re.sub(r'^[\d\.\-\*\s]+', '', i).strip() for i in items if len(i.strip()) > 5]
        
        return clean_items[:5]

    except Exception as e:
        return [f"Gemini Analysis Error: {str(e)}"]

def extract_glossary_data(text: str) -> list:
    """
    Extracts legal terms and definitions for Database storage using Gemini.
    Returns JSON: [{'term': '...', 'definition': '...'}]
    """
    if not GOOGLE_API_KEY: return []
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        Analyze this document. Extract the top 10 most important LEGAL TERMS and their definitions.
        STRICTLY Return ONLY a valid JSON array.
        Format: [{{"term": "Example", "definition": "Meaning..."}}]
        Document: {text[:30000]}
        """
        resp = model.generate_content(prompt)
        clean_json = resp.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        print(f"Glossary Error: {e}")
        return []

def get_graph_data(text: str) -> dict:
    """
    Generates 0-100 scores for the Dashboard Graphs using Gemini.
    """
    if not GOOGLE_API_KEY: 
        return {"Risk Score": 0, "Ambiguity": 0, "Complexity": 0, "Completeness": 0}
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        Analyze this document and provide a rating from 0 to 100 for:
        1. Risk Score (Higher = More Risky)
        2. Ambiguity (Higher = More Unclear)
        3. Complexity (Higher = Harder to Read)
        4. Completeness (Higher = Better Coverage)
        
        Return ONLY valid JSON: {{ "Risk Score": 0, "Ambiguity": 0, "Complexity": 0, "Completeness": 0 }}
        Document: {text[:15000]}
        """
        resp = model.generate_content(prompt)
        clean_json = resp.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        print(f"Graph Error: {e}")
        return {"Risk Score": 50, "Ambiguity": 50, "Complexity": 50, "Completeness": 50}

# ════════════════════════════════════════════════════════════════
# 6. ROBUST RAG IMPLEMENTATION (LOCAL FAISS + FLAN-T5)
# ════════════════════════════════════════════════════════════════

PROMPT_TEMPLATE = """
You are a helpful and conversational assistant. Use the provided context to answer the user's question.
The user's recent chat history (if any) is provided before the main question.
ONLY use the information from the 'Document Context'. Do not make up answers.
If the answer is not in the context, say "I'm sorry, that information is not in the document."

Document Context:
{context}

Chat History & Question:
{question}

Helpful Answer:"""

class ClauseEaseRAG:
    """
    LangChain-powered RAG for legal Q&A.
    Strictly uses LOCAL models (FAISS for retrieval, FLAN-T5 for generation).
    """
    
    def __init__(self, document_text: str):
        if not document_text or len(document_text) < 10:
            raise ValueError("Empty or invalid document text for RAG initialization.")
        
        self.full_text = document_text 
        self.chat_history = []
        
        # Splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50, 
            separators=["\n\n", "\n", ". ", " "]
        )
        docs = self.splitter.create_documents([document_text])
        if not docs:
            raise ValueError("Text splitting resulted in zero documents.")

        # Embeddings (Robust Loading)
        embed_dir = MODEL_PATHS["embed"]
        if not os.path.exists(embed_dir):
            print("⚠️ Local embedding model missing. Downloading sentence-transformers/all-MiniLM-L6-v2")
            embed_dir = "sentence-transformers/all-MiniLM-L6-v2"
            
        self.embedding_model = HuggingFaceEmbeddings(model_name=embed_dir)

        # Vector Store
        self.vectorstore = FAISS.from_documents(docs, self.embedding_model)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Generator (Use Cached Pipeline - LOCAL)
        # Note: This pipeline now includes the 'repetition_penalty' fix.
        pipe = load_pipeline_cached("FLAN-T5-RAG")
        if isinstance(pipe, str) and pipe.startswith("Error"):
            raise ValueError(f"Failed to load RAG generator: {pipe}")

        llm = HuggingFacePipeline(pipeline=pipe)

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, 
            input_variables=["context", "question"] 
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.retriever,
            chain_type="stuff",
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt} 
        )

    def query(self, question: str) -> str:
        """Ask a natural language question to the RAG system (Local)."""
        if not question or not question.strip():
            return "Error: Empty query."
        
        # Format history
        history_string = "\n".join(f"User: {q}\nAI: {a}" for q, a in self.chat_history)
        combined_input = f"{history_string}\n\nUser: {question}"

        try:
            # Run the Chain
            response = self.qa_chain.invoke({"query": combined_input.strip()})
            answer = response.get("result", response.get("answer", "No relevant answer found."))
            
            # Clean up artifacts if any (sometimes local models output tokens)
            if "<pad>" in answer: answer = answer.replace("<pad>", "")
            if "</s>" in answer: answer = answer.replace("</s>", "")
            
            # Update history
            self.chat_history.append((question, answer))
            self.chat_history = self.chat_history[-3:] 
            
            return answer.strip()
        
        except Exception as e:
            return f"Error: RAG query failed → {e}"


def create_rag_chain(text):
    """Initialize ClauseEase RAG instance with safety checks."""
    try:
        return ClauseEaseRAG(text)
    except Exception as e:
        error_msg = f"Error: Failed to initialize RAG system. Details: {e}"
        print(error_msg)
        return error_msg

# ------------------------------------------------------------------
# 7. MASTER QUERY ROUTER
# ------------------------------------------------------------------

CHITCHAT_RESPONSES = {
    "greeting": "Hello! I'm ready to help you with your document.",
    "thanks": "You're welcome!",
    "goodbye": "Goodbye!",
    "help": "I am a legal assistant. Ask me about your document.",
}

def query_rag_chain(chain, prompt):
    """
    Routes queries to the correct handler (Chitchat, Summary, RAG).
    """
    try:
        lower_prompt = prompt.lower().strip()
        
        # Chitchat
        if lower_prompt in {'hi', 'hello', 'hey'}: return CHITCHAT_RESPONSES["greeting"]
        if lower_prompt in {'thanks', 'thank you'}: return CHITCHAT_RESPONSES["thanks"]
        
        # Summarization (Meta) - Uses Local Simplification Logic
        if "summarize" in lower_prompt or "what is this document" in lower_prompt:
            if hasattr(chain, 'full_text'):
                return simplify_text(chain.full_text, "FLAN-T5", "Basic")
            return "Error: Full text unavailable for summary."

        # RAG Query
        if hasattr(chain, 'query') and callable(chain.query):
            return chain.query(prompt)
            
        return "Error: RAG chain not initialized."
    
    except Exception as e:
        return f"Error processing query: {e}"