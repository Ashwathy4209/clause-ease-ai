import streamlit as st
import pandas as pd
import altair as alt
from db import get_glossary_terms
from readability import highlight_legal_terms, analyze_readability, color_code_complexity

def local_css():
    """
    Injects local CSS for the document viewer, badges, graphs, and ISSUE CARDS.
    """
    st.markdown("""
    <style>
    /* Document Viewer Styling */
    .doc-viewer {
        height: 500px;
        overflow-y: auto;
        padding: 15px;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 14px;
        line-height: 1.6;
        background-color: white;
        color: #333;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.03);
    }
    
    /* Document Type Badges */
    .doc-type-legal {
        background-color: #FEF2F2; 
        color: #991B1B; 
        padding: 8px 12px; 
        border-radius: 6px; 
        font-weight: 600; 
        text-align: center;
        border: 1px solid #FECACA;
        margin-bottom: 10px;
    }
    .doc-type-non-legal {
        background-color: #ECFDF5; 
        color: #065F46; 
        padding: 8px 12px; 
        border-radius: 6px; 
        font-weight: 600; 
        text-align: center;
        border: 1px solid #A7F3D0;
        margin-bottom: 10px;
    }

    /* --- NEW: IMPROVED ISSUES UI --- */
    .issues-container {
        background-color: #fff;
        border-radius: 8px;
    }
    .issue-card {
        background-color: #fff;
        border: 1px solid #FED7D2; /* Light Red Border */
        border-left: 4px solid #F87171; /* Strong Red Accent */
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        transition: transform 0.1s ease-in-out;
    }
    .issue-card:hover {
        background-color: #FEF2F2;
    }
    .issue-header {
        font-weight: 700;
        font-size: 13px;
        color: #B91C1C;
        display: flex;
        align-items: center;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .issue-body {
        font-size: 13px;
        color: #374151;
        line-height: 1.5;
    }
    
    /* Success Card (No Issues) */
    .success-card {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #4ADE80;
        padding: 15px;
        border-radius: 6px;
        color: #166534;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def show_page(tenant_db, tenant_user_id):
    """
    Renders the "Legal Assistant" comparison and analysis page.
    """
    # 1. Inject Styles
    local_css()

    col_preview, col_simplify, col_analysis = st.columns([2, 2, 1.2], gap="medium") # Adjusted width slightly for better UI
    
    # 2. Load Glossary
    glossary_terms = {}
    try:
        glossary_terms = get_glossary_terms(tenant_db)
    except Exception as e:
        st.error(f"Glossary load error: {e}")

    # =========================================================
    # COL 1: ORIGINAL TEXT (With Glossary Highlights)
    # =========================================================
    with col_preview:
        st.subheader("Extracted Document Text")
        current_text = st.session_state.get("current_text")
        
        if current_text:
            try:
                highlighted_html = highlight_legal_terms(current_text, glossary_terms) 
                st.markdown(f'<div class="doc-viewer">{highlighted_html}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating highlights: {e}")
                st.markdown(f'<div class="doc-viewer" style="white-space: pre-wrap;">{current_text}</div>', unsafe_allow_html=True) 
        else:
            st.markdown('<div class="doc-viewer">No document processed or loaded.</div>', unsafe_allow_html=True)

    # =========================================================
    # COL 2: SIMPLIFIED TEXT (With Highlights)
    # =========================================================
    with col_simplify:
        model_name = st.session_state.get("simplification_model", "N/A")
        st.subheader(f"Simplified Version ({model_name})")
        
        level = st.session_state.get("simplification_level", "N/A")
        st.caption(f"Simplification Level Chosen: *{level}*")
        
        simplified_text = st.session_state.get("simplified_text")

        if simplified_text:
            if "Error:" in str(simplified_text):
                st.error(f"Simplification failed: {simplified_text}")
            else:
                try:
                    highlighted_simplified = highlight_legal_terms(simplified_text, glossary_terms)
                    st.markdown(f'<div class="doc-viewer" style="background-color: #F8FBFB;">{highlighted_simplified}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Highlight error: {e}")
                    st.markdown(f'<div class="doc-viewer" style="background-color: #F8FBFB; white-space: pre-wrap;">{simplified_text}</div>', unsafe_allow_html=True) 
        else:
            st.markdown('<div class="doc-viewer" style="background-color: #F8FBFB; display: flex; align-items: center; justify-content: center; color: #888;">No simplified text available.</div>', unsafe_allow_html=True)

    # =========================================================
    # COL 3: ANALYSIS & ISSUES (UPDATED UI)
    # =========================================================
    with col_analysis:
        st.subheader("Analysis Report")
        is_legal = st.session_state.get("is_likely_legal")
        
        # --- Document Type Badge ---
        if is_legal is True: 
            st.markdown('<div class="doc-type-legal"> Likely Legal Document</div>', unsafe_allow_html=True)
        elif is_legal is False: 
            st.markdown('<div class="doc-type-non-legal"> Likely Non-Legal Document</div>', unsafe_allow_html=True)
        elif st.session_state.current_text:
             st.info("Analysis Pending...")
        else: 
            st.info("No Data")

        st.markdown("---")

        # --- Readability Score ---
        st.markdown("**Readability Score**")
        analytics = st.session_state.get("doc_analytics")
        if analytics and isinstance(analytics, dict):
            ease = analytics.get("flesch_ease", 0)
            ease_pct = int(max(0, min(100, ease)))
            st.progress(ease_pct / 100)
            
            lvl = "Very Difficult"
            if ease >= 80: lvl = "Very Easy"
            elif ease >= 60: lvl = "Easy"
            elif ease >= 30: lvl = "Difficult"
            
            c_a, c_b = st.columns([1,2])
            with c_a: st.metric("Score", f"{ease:.0f}")
            with c_b: st.caption(f"**Level:** {lvl}")
        else:
            st.markdown('<span style="color: #90A4AE;">Not calculated</span>', unsafe_allow_html=True)

        st.markdown("---")

        # --- ISSUES DETECTED SECTION (NEW UI) ---
        st.subheader("Risk Detection")
        current_text = st.session_state.get("current_text")
        
        if current_text:
            # CASE 1: Non-Legal Document (Show Success Card)
            if is_legal is False:
                st.markdown("""
                <div class="success-card">
                    <div></div>
                    <div><strong>Safe:</strong> Non-legal document detected. No clause analysis required.</div>
                </div>
                """, unsafe_allow_html=True)
            
            # CASE 2: Legal Document (Show Styled Issue Cards)
            elif is_legal is True:
                issues = st.session_state.get("ai_issues", [])
                
                # Determine list of issues
                final_issues = []
                if issues and isinstance(issues, list) and len(issues) > 0:
                    final_issues = issues
                else:
                    # Fallback Logic
                    text_lower = current_text.lower()
                    if "governing law" not in text_lower: final_issues.append("Missing 'Governing Law' clause.")
                    if "liability" not in text_lower: final_issues.append("Missing 'Liability' or 'Indemnity' clause.")
                    if "termination" not in text_lower: final_issues.append("Missing 'Termination' conditions.")

                # Render Issues
                if final_issues:
                    st.markdown(f"**Found {len(final_issues)} potential issues:**")
                    st.markdown('<div class="issues-container">', unsafe_allow_html=True)
                    for issue in final_issues:
                        st.markdown(f"""
                        <div class="issue-card">
                            <div class="issue-header"> Attention Needed</div>
                            <div class="issue-body">{issue}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-card">
                        <div></div>
                        <div><strong>All Clear:</strong> No obvious major issues detected in this scan.</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # CASE 3: Analysis Pending
            else:
                st.info("Waiting for analysis...")
        else:
            st.caption("Upload a document to see issues.")

    # =========================================================
    # BOTTOM SECTION: GRAPH & METRICS
    # =========================================================
    st.markdown("---")
    col_analytics_detail, col_graph = st.columns([1, 2], gap="medium")
    
    # --- Detailed Metrics ---
    with col_analytics_detail:
        st.subheader("Detailed Metrics")
        analytics = st.session_state.get("doc_analytics")
        if analytics and isinstance(analytics, dict):
            fk = analytics.get("flesch_kincaid", 0.0)
            fog = analytics.get("gunning_fog", 0.0)
            rt = analytics.get("read_time_minutes", 0)
            
            c1, c2 = st.columns(2)
            with c1: st.metric("Grade Level", f"{fk:.1f}")
            with c2: st.metric("Gunning Fog", f"{fog:.1f}")
            st.metric("Est. Read Time", f"{rt} min")
        else:
            st.info("No detailed metrics available.")
    
    # --- NEW GRAPH: Matches models.py "get_graph_data" ---
    with col_graph:
        st.subheader("Document Health & Readability")
        
        graph_data = st.session_state.get("graph_data", {})
        
        if graph_data and isinstance(graph_data, dict) and "Risk Score" in graph_data:
            df_graph = pd.DataFrame([
                {"Metric": "Risk Score",   "Score": graph_data.get("Risk Score", 0),   "Color": "#EF4444"}, # Red
                {"Metric": "Ambiguity",    "Score": graph_data.get("Ambiguity", 0),    "Color": "#F59E0B"}, # Orange
                {"Metric": "Complexity",   "Score": graph_data.get("Complexity", 0),   "Color": "#3B82F6"}, # Blue
                {"Metric": "Completeness", "Score": graph_data.get("Completeness", 0), "Color": "#10B981"}  # Green
            ])
            
            try:
                chart = alt.Chart(df_graph).mark_bar(
                    cornerRadiusTopLeft=5, 
                    cornerRadiusTopRight=5
                ).encode(
                    x=alt.X('Metric', sort=["Risk Score", "Ambiguity", "Complexity", "Completeness"], title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Score', scale=alt.Scale(domain=[0, 100]), title="Score (0-100)"),
                    color=alt.Color('Color', scale=None, legend=None), 
                    tooltip=['Metric', 'Score']
                ).properties(
                    height=220
                )
                
                st.altair_chart(chart, use_container_width=True)
                st.caption("Metrics estimated by AI (0-100 Scale).")
            except Exception as e:
                st.error(f"Graph rendering failed: {e}")
                st.dataframe(df_graph)
        else:
            if st.session_state.current_text:
                st.info("Graph data not available. Please re-process the document.")
            else:
                st.info("Process a document to see health metrics.")
                