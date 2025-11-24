import streamlit as st
import pandas as pd
import altair as alt
from db import get_all_documents

def show_page(tenant_db, tenant_user_id):
    """
    Renders the User-Specific Dashboard.
    Focus: Personal productivity, simplification stats, and recent history.
    """
    user_name = st.session_state.get('user_name', 'User').split(' ')[0]
    st.title(f"Welcome back, {user_name}!")
    st.markdown("Here is your personal document activity overview.")
    
    st.markdown("---")

    # --- 1. FETCH DATA ---
    try:
        # Get ALL documents, then filter for THIS user
        all_docs_df = get_all_documents(tenant_db)
        
        if all_docs_df.empty:
            user_docs_df = pd.DataFrame()
        else:
            user_docs_df = all_docs_df[all_docs_df['user_id'] == tenant_user_id].copy()

    except Exception as e:
        st.error(f"Error loading data: {e}")
        user_docs_df = pd.DataFrame()

    # --- 2. CALCULATE PERSONAL METRICS ---
    if not user_docs_df.empty:
        total_docs = len(user_docs_df)
        
        # Calculate Words Saved (Original - Simplified)
        # Ensure numeric safety
        user_docs_df['original_word_count'] = pd.to_numeric(user_docs_df['original_word_count'], errors='coerce').fillna(0)
        user_docs_df['simplified_word_count'] = pd.to_numeric(user_docs_df['simplified_word_count'], errors='coerce').fillna(0)
        
        total_orig = user_docs_df['original_word_count'].sum()
        total_simp = user_docs_df['simplified_word_count'].sum()
        words_saved = int(total_orig - total_simp)
        efficiency = int((words_saved / total_orig * 100)) if total_orig > 0 else 0
        
        last_active = pd.to_datetime(user_docs_df['uploaded_at']).max().strftime('%b %d, %Y')
    else:
        total_docs = 0
        words_saved = 0
        efficiency = 0
        last_active = "N/A"

    # --- 3. DISPLAY METRICS (CSS Styled) ---
    st.markdown("""
    <style>
    .metric-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #00796B; }
    .metric-label { font-size: 0.9rem; color: #666; margin-bottom: 5px; }
    .metric-delta { font-size: 0.8rem; color: #2E7D32; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">My Documents</div><div class="metric-value">{total_docs}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Words Removed</div><div class="metric-value">{words_saved:,}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Efficiency Gain</div><div class="metric-value">{efficiency}%</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Last Active</div><div class="metric-value" style="font-size:1.2rem; line-height:2.4rem;">{last_active}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- 4. CHARTS ---
    if not user_docs_df.empty:
        st.subheader("Simplification Impact")
        
        # Prepare data for Bar Chart
        chart_df = user_docs_df.head(10).copy() # Show last 10 docs
        chart_df['Title'] = chart_df['document_title'].apply(lambda x: x[:15] + '...' if len(str(x)) > 15 else str(x))
        
        # Melt for Stacked/Grouped Bar
        melted_df = chart_df.melt(id_vars=['Title'], value_vars=['original_word_count', 'simplified_word_count'], 
                                  var_name='Version', value_name='Count')
        
        melted_df['Version'] = melted_df['Version'].replace({
            'original_word_count': 'Original Length', 
            'simplified_word_count': 'Simplified Length'
        })

        # Altair Chart
        chart = alt.Chart(melted_df).mark_bar().encode(
            x=alt.X('Title', sort=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('Count', title='Word Count'),
            color=alt.Color('Version', scale=alt.Scale(domain=['Original Length', 'Simplified Length'], range=['#ef5350', '#66bb6a'])),
            tooltip=['Title', 'Version', 'Count']
        ).properties(
            height=350,
            title="Word Reduction Analysis (Recent Docs)"
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Upload your first document to see analytics here!")