import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import time
from datetime import datetime
import requests

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from src.embeddings.generate_embeddings import EmbeddingGenerator
from src.embeddings.chroma_setup import ChromaVectorDB

app = FastAPI(title="AutoRAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading embeddings model...")
embedding_gen = EmbeddingGenerator()

print("Loading vector database...")
vector_db = ChromaVectorDB()

print("✅ All components ready! Using LOCAL Ollama for answers")


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
    llm: str


def generate_answer_with_ollama(question: str, context: str) -> str:
    """Generate answer using local Ollama"""

    prompt = f"""You are an expert automotive engineer analyzing research papers.

Based on these research excerpts, answer the following question.

QUESTION: {question}

RESEARCH EXCERPTS:
{context}

Answer based ONLY on the excerpts provided. Be precise and technical. Cite sources using [Source N] format."""

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7
            },
            timeout=120
        )

        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: Ollama returned {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ ERROR: Ollama not running. Please run 'ollama run mistral' in another terminal"
    except Exception as e:
        return f"❌ Error: {str(e)}"


@app.get("/")
async def root():
    return {
        "name": "AutoRAG API",
        "status": "running",
        "version": "1.0.0",
        "llm": "Local Ollama (FREE)"
    }


@app.get("/stats")
async def get_stats():
    return StatsResponse(
        total_papers=vector_db.count(),
        indexed_at=datetime.now().isoformat(),
        status="ready",
        llm="Ollama Mistral (Local)"
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Execute RAG query with local Ollama"""
    start_time = time.time()

    try:
        print(f"Embedding question: {req.question}")
        question_embedding = embedding_gen.embed_text(req.question)

        print("Searching Chroma vector database...")
        search_results = vector_db.query(question_embedding, n_results=req.top_k)

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
                relevance = 1 / (1 + distance)
                results.append({
                    'chunk_id': chunk_id,
                    'document': doc,
                    'relevance': relevance
                })

            context_text = "\n\n".join([
                f"[Source {i + 1}]\n{r['document']}"
                for i, r in enumerate(results)
            ])

            print("Generating answer with LOCAL Ollama...")
            answer = generate_answer_with_ollama(req.question, context_text)

            sources = [
                Source(
                    id=r['chunk_id'],
                    content=r['document'][:300],
                    similarity_score=r['relevance'],
                    relevance_score=r['relevance']
                )
                for r in results
            ]

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

    for i, result in enumerate(results):
        nodes.append(GraphNode(
            id=result['chunk_id'],
            label=result['chunk_id'].split('_')[0][:8],
            relevance=result['relevance'],
            size=max(20, result['relevance'] * 40)
        ))

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