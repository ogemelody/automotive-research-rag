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
    st.markdown("**AI-Powered Automotive Research Assistant**")

with col2:
    st.markdown("""
    <div style="text-align: right;">
        <p style="color: #10b981; font-weight: bold;">🟢 Offline Mode (Ollama)</p>
        <p style="color: #64748b; font-size: 12px;">19 papers indexed</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### 📚 System Status")

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

    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Number of sources to retrieve", 3, 10, 5)
    temperature = st.slider("Response creativity", 0.0, 1.0, 0.7)

    st.divider()

    st.markdown("### 📖 About")
    st.info(
        """
        AutoRAG retrieves relevant papers and uses Ollama to synthesize answers.

        **No API costs • Runs locally • Fully offline**
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
                    tab1, tab2, tab3 = st.tabs(["📝 Answer", "📚 Sources", "🔗 Knowledge Graph"])

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

                    with tab3:
                        st.markdown("### 🔗 Knowledge Graph")

                        if result['graph']['nodes']:
                            # Create interactive Plotly graph
                            fig = create_knowledge_graph(result['graph'])
                            st.plotly_chart(fig, use_container_width=True)

                            st.info(
                                f"""
                                **Graph Details:**
                                - Nodes: {len(result['graph']['nodes'])} papers retrieved
                                - Connections: {len(result['graph']['edges'])} relationships
                                - Node size = Relevance score
                                - Hover for details
                                """
                            )
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

[API Documentation](http://localhost:8000/docs) • [Ollama](https://ollama.ai)
""")


def create_knowledge_graph(graph_data):
    """Create interactive Plotly graph"""

    # Node positions (simple layout)
    import math
    n = len(graph_data['nodes'])
    radius = 2

    x_pos = [radius * math.cos(2 * math.pi * i / n) for i in range(n)]
    y_pos = [radius * math.sin(2 * math.pi * i / n) for i in range(n)]

    # Create figure
    fig = go.Figure()

    # Add edges
    edge_x = []
    edge_y = []
    for edge in graph_data['edges']:
        source_idx = next(i for i, n in enumerate(graph_data['nodes']) if n['id'] == edge['source'])
        target_idx = next(i for i, n in enumerate(graph_data['nodes']) if n['id'] == edge['target'])

        edge_x.append(x_pos[source_idx])
        edge_x.append(x_pos[target_idx])
        edge_x.append(None)

        edge_y.append(y_pos[source_idx])
        edge_y.append(y_pos[target_idx])
        edge_y.append(None)

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='#475569'),
        hoverinfo='none',
        showlegend=False
    ))

    # Add nodes
    node_colors = [node['relevance'] for node in graph_data['nodes']]
    node_sizes = [node['size'] for node in graph_data['nodes']]

    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale='Viridis',
            showscale=True,
            line=dict(width=2, color='#0f172a')
        ),
        text=[node['label'] for node in graph_data['nodes']],
        textposition="middle center",
        textfont=dict(color='white', size=10),
        hovertext=[node['id'] for node in graph_data['nodes']],
        hoverinfo='text',
        showlegend=False
    ))

    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        plot_bgcolor='#0f172a',
        paper_bgcolor='#1e293b',
        font=dict(color='#e2e8f0')
    )

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    return fig