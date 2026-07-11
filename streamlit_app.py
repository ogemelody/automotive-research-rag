import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# Page config
st.set_page_config(
    page_title="AutoRAG",
    page_icon="🚗⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .stButton>button {
        background: linear-gradient(90deg, #06b6d4 0%, #0284c7 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
    h1, h2, h3 {
        color: #06b6d4;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.markdown("# 🚗⚡ AutoRAG")
    st.markdown("**Semantic Intelligence for Automotive Research**")

with col2:
    st.markdown("""
    <div style="text-align: right;">
        <p style="color: #64748b; font-size: 12px;">Built by MelodyEgwuchukwu🩷</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("###  System Status")

    # API Connection
    try:
        response = requests.get("http://localhost:8000/stats", timeout=2)
        if response.status_code == 200:
            stats = response.json()
            st.success("✅ Backend Connected")
            st.metric("Papers Indexed", stats['total_papers'])
        else:
            st.error("❌ Backend Error")
    except:
        st.error("❌ Backend Offline - Start python backend/main.py")

    st.divider()

    st.markdown("###  Settings")
    top_k = st.slider("Number of sources to retrieve", 3, 100, 5)
    temperature = st.slider("Response creativity", 0.0, 1.0, 0.7)

    st.divider()

    st.markdown("###  About")
    st.info(
        """
        AutoRAG combines semantic search (BAAI/bge embeddings + Chroma vector DB) with local Ollama LLM to retrieve relevant research papers and generate grounded answers with source citations for automotive and electric vehicle research questions.

        **No API costs • Runs locally**
        """
    )

# Main content
st.markdown("### 🔍 Ask About Automotive Research")

# Search input
question = st.text_input(
    "Enter your question:",
    placeholder="What are thermal management challenges in EV batteries?",
    label_visibility="collapsed"
)

# Suggestions
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("💡 Thermal challenges"):
        question = "What are thermal management challenges in EV batteries?"

with col2:
    if st.button("🔋 Solid-state batteries"):
        question = "How do solid-state batteries work?"

with col3:
    if st.button("⚡ Fast charging"):
        question = "What is the fast charging protocol?"

st.divider()

# Search button and results
if st.button("🔍 Search", use_container_width=True, type="primary"):
    if question:
        with st.spinner("🔄 Searching papers and generating answer..."):
            try:
                response = requests.post(
                    "http://localhost:8000/query",
                    json={"question": question, "top_k": top_k},
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()

                    # Display results in tabs
                    tab1, tab2 = st.tabs(["📝 Answer", "📚 Sources"])

                    with tab1:
                        st.markdown("### Answer")
                        st.markdown(f"""
                        <div style="background: #1e293b; padding: 20px; border-radius: 10px; color: #e2e8f0;">
                        {result['answer']}
                        </div>
                        """, unsafe_allow_html=True)

                        col1, col2 = st.columns([0.8, 0.2])
                        with col2:
                            st.metric("Response Time", f"{result['latency_ms']:.0f}ms")

                    with tab2:
                        st.markdown("### 📄 Sources")

                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(
                                    f"[{i}] {source['id']} - {source['relevance_score'] * 100:.0f}% relevance"
                            ):
                                st.write(source['content'])
                                st.markdown(f"""
                                **Metrics:**
                                - Relevance: {source['relevance_score'] * 100:.1f}%
                                - Similarity: {source['similarity_score'] * 100:.1f}%
                                """)
                        else:
                            st.warning("No papers found for this query")
                else:
                    st.error(f"Error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Make sure it's running on http://localhost:8000")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question")

st.divider()

# Footer
st.markdown("""
---
**Built with Streamlit • Powered by AutoRAG • Running locally with Ollama**

 [Connect with Melody](https://www.linkedin.com/in/melodyegwuchukwu/) • [Ollama](https://ollama.ai)
""")

