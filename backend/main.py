import sys
import os
from pathlib import Path
from dotenv import load_dotenv  # ADD THIS LINE
import time
from datetime import datetime

# Load .env file
load_dotenv()  # ADD THIS LINE

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from anthropic import Anthropic

# Import YOUR actual modules
from src.embeddings.generate_embeddings import EmbeddingGenerator
from src.embeddings.chroma_setup import ChromaVectorDB

# Initialize FastAPI
app = FastAPI(title="AutoRAG API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load components
print("Loading embeddings model...")
embedding_gen = EmbeddingGenerator()

print("Loading vector database...")
vector_db = ChromaVectorDB()

print("Initializing Claude API...")
api_key = os.getenv("ANTHROPIC_API_KEY")  # CHANGED THIS
if not api_key:
    print("⚠️  WARNING: ANTHROPIC_API_KEY not found in .env file")
    print("Please create .env with: ANTHROPIC_API_KEY=sk-ant-...")
claude = Anthropic(api_key=api_key)  # CHANGED THIS

print("✅ All components ready!")


# ============ DATA MODELS ============

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    id: str
    content: str
    similarity_score: float
    relevance_score: float


class GraphNode(BaseModel):
    id: str
    label: str
    relevance: float
    size: float


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float


class KnowledgeGraph(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    graph: KnowledgeGraph
    latency_ms: float


class StatsResponse(BaseModel):
    total_papers: int
    indexed_at: str
    status: str


# ============ ROUTES ============

@app.get("/")
async def root():
    """Health check"""
    return {
        "name": "AutoRAG API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return StatsResponse(
        total_papers=vector_db.count(),
        indexed_at=datetime.now().isoformat(),
        status="ready"
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Execute RAG query using YOUR actual code"""
    start_time = time.time()

    try:
        # Step 1: Embed the question
        print(f"Embedding question: {req.question}")
        question_embedding = embedding_gen.embed_text(req.question)

        # Step 2: Search Chroma
        print("Searching Chroma vector database...")
        search_results = vector_db.query(question_embedding, n_results=req.top_k)

        # Step 3: Format retrieved chunks
        if not search_results or not search_results.get('ids'):
            answer = "No relevant papers found."
            sources = []
            results = []
        else:
            results = []
            for chunk_id, doc, distance in zip(
                    search_results['ids'][0],
                    search_results['documents'][0],
                    search_results['distances'][0]
            ):
                relevance = 1 / (1 + distance)  # Convert distance to similarity
                results.append({
                    'chunk_id': chunk_id,
                    'document': doc,
                    'relevance': relevance
                })

            # Step 4: Build context for Claude
            context_text = "\n\n".join([
                f"[Source {i + 1}]\n{r['document']}"
                for i, r in enumerate(results)
            ])

            # Step 5: Generate answer with Claude
            print("Generating answer with Claude...")
            response = claude.messages.create(
                model="claude-opus-4-6",
                max_tokens=1500,
                system="""You are an expert automotive engineer analyzing research papers.
Answer based ONLY on the provided research excerpts.
Cite sources using [Source N] format.
Be precise and technical.""",
                messages=[{
                    "role": "user",
                    "content": f"Based on these research excerpts, answer: {req.question}\n\nRESEARCH EXCERPTS:\n{context_text}"
                }]
            )

            answer = response.content[0].text

            # Format sources for response
            sources = [
                Source(
                    id=r['chunk_id'],
                    content=r['document'][:300],
                    similarity_score=r['relevance'],
                    relevance_score=r['relevance']
                )
                for r in results
            ]

        # Step 6: Build knowledge graph
        graph = build_knowledge_graph(results)

        latency = (time.time() - start_time) * 1000

        return QueryResponse(
            question=req.question,
            answer=answer,
            sources=sources,
            graph=graph,
            latency_ms=latency
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        raise


def build_knowledge_graph(results):
    """Create relationship graph for visualization"""
    nodes = []
    edges = []

    # Create nodes
    for i, result in enumerate(results):
        nodes.append(GraphNode(
            id=result['chunk_id'],
            label=result['chunk_id'].split('_')[0][:8],
            relevance=result['relevance'],
            size=max(20, result['relevance'] * 40)
        ))

    # Create edges
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            edges.append(GraphEdge(
                source=nodes[i].id,
                target=nodes[j].id,
                weight=0.5
            ))

    return KnowledgeGraph(nodes=nodes, edges=edges)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)