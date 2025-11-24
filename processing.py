import streamlit as st
import time
from db import save_document, update_glossary_from_ai_output
from utils import get_word_count, is_likely_legal
from readability import analyze_readability

def process_document_logic(tenant_db, tenant_user_id, source_type):
    """
    Handles processing: Simplified Text, Readability, AND Gemini Graph Data.
    """
    
    # --- LAZY LOAD HEAVY MODULES ---
    import models 
    
    status = None
    current_step = "Initialization"
    
    try:
        with st.status("Processing document...", expanded=True) as status_context:
            status = status_context

            # Step 1: Validate Text
            current_step = "Validating Text"
            st.write(f"{current_step}...")
            if st.session_state.current_text is None:
                raise ValueError("No text found to process.")
            time.sleep(0.1)

            # --- Step 2: RAG Building ---
            current_step = "Building RAG Model"
            st.write(f"{current_step}...")
            st.session_state.rag_chain = models.create_rag_chain(st.session_state.current_text)
            
            # Safety check on RAG object
            if isinstance(st.session_state.rag_chain, str) and st.session_state.rag_chain.startswith("Error:"):
                raise ValueError(st.session_state.rag_chain)
            
            st.session_state.model_ready = hasattr(st.session_state.rag_chain, 'query')
            if not st.session_state.model_ready:
                raise ValueError("RAG chain failed to initialize.")
            time.sleep(0.2)

            # --- Step 3: Simplification ---
            chosen_level = st.session_state.simplification_level
            current_step = f"Simplifying Text ({chosen_level})"
            st.write(f"{current_step}...")
            
            s_text = models.simplify_text(
                st.session_state.current_text, 
                model_choice=st.session_state.simplification_model, 
                level=chosen_level
            )
            st.session_state.simplified_text = s_text
            
            if "Error:" in str(s_text):
                raise ValueError(f"Simplification failed: {s_text}")
            time.sleep(0.2)

            # --- Step 4: Readability Analysis ---
            current_step = "Analyzing Readability"
            st.write(f"{current_step}...")
            
            st.session_state.doc_analytics = analyze_readability(st.session_state.current_text)
            st.session_state.simplified_doc_analytics = analyze_readability(st.session_state.simplified_text)
            st.session_state.is_likely_legal = is_likely_legal(st.session_state.current_text)
            time.sleep(0.1)

            # --- Step 5: Gemini AI Analysis (GRAPH DATA & RISKS) ---
            # THIS WAS MISSING IN YOUR CODE
            current_step = "Consulting Gemini AI (Graph & Risks)"
            st.write(f"{current_step}...")
            
            try:
                # 1. GET GRAPH DATA (0-100 Scores)
                graph_data = models.get_graph_data(st.session_state.current_text)
                st.session_state.graph_data = graph_data 
                
                # 2. GET TEXT LISTS (Removed 'model' arg to fix TypeError)
                st.session_state.ai_issues = models.get_ai_analysis(
                    st.session_state.current_text, analysis_type="legal_issues"
                )
                st.session_state.ai_risks = models.get_ai_analysis(
                    st.session_state.current_text, analysis_type="legal_risks"
                )
                
                # 3. EXTRACT GLOSSARY (Optional: Use Gemini extraction if you want)
                # extracted_terms = models.extract_glossary_data(st.session_state.current_text)
                
            except Exception as ai_e:
                st.warning(f"AI analysis failed: {ai_e}")
                st.session_state.graph_data = {}
                st.session_state.ai_issues = []
                st.session_state.ai_risks = []
            
            time.sleep(0.2)
            
            # --- Step 6: Saving Document ---
            current_step = "Saving Document"
            st.write(f"{current_step}...")
            
            # Calculate flags/counts
            is_legal_flag = 1 if st.session_state.is_likely_legal is True else (0 if st.session_state.is_likely_legal is False else -1)
            wc_orig = get_word_count(st.session_state.current_text)
            wc_simple = get_word_count(st.session_state.simplified_text)

            doc_id = save_document(
                tenant_db, tenant_user_id, st.session_state.uploaded_file_name,
                st.session_state.current_title, st.session_state.current_text,
                st.session_state.simplified_text,    
                chosen_level,                    
                is_legal_flag,                   
                wc_orig,                         
                wc_simple                        
            )
            st.session_state.current_document_id = int(doc_id)
            time.sleep(0.2)

            # --- Step 7: Glossary Update ---
            current_step = "Updating Glossary"
            st.write(f"{current_step}...")
            if st.session_state.simplified_text:
                update_glossary_from_ai_output(
                    tenant_db,
                    st.session_state.current_text,
                    st.session_state.simplified_text
                )
            
            status.update(label="Processing Complete!", state="complete", expanded=False)
        
        time.sleep(0.5)
        return True

    except Exception as e:
        error_message = f"Processing Failed at step '{current_step}': {e}"
        if status:
             status.update(label="Error Occurred", state="error", expanded=True)
             st.error(error_message)
        else:
             st.error(error_message)

        # Reset state on failure
        st.session_state.model_ready = False
        st.session_state.simplified_text = None
        st.session_state.doc_analytics = None
        st.session_state.graph_data = None
        st.session_state.rag_chain = None
        return False