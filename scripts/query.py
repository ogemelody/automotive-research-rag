import sys
import os
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import requests
from src.embeddings.generate_embeddings import EmbeddingGenerator
from src.embeddings.chroma_setup import ChromaVectorDB


def query_papers(question: str, n_results: int = 5) -> str:
    db = ChromaVectorDB()
    total = db.count()

    if total == 0:
        print("No papers ingested yet. Run: python scripts/ingest_all.py")
        sys.exit(1)

    print(f"Searching {total} chunks across ingested papers...")

    embedding_gen = EmbeddingGenerator()
    query_embedding = embedding_gen.embed_text(question)

    results = db.query(query_embedding.tolist(), n_results=n_results)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        return "No relevant content found for that question."

    context_parts = []
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        paper_id = meta.get("paper_id", "unknown")
        similarity = 1 - dist
        context_parts.append(f"[Source {i + 1} | {paper_id} | similarity: {similarity:.2f}]\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    # Prepare the prompt for Ollama
    prompt = f"""You are an automotive research assistant. Answer the question below using ONLY the provided research paper excerpts. Cite which source each point comes from.

Question: {question}

Research excerpts:
{context}

Provide a clear, structured answer based on the sources above."""

    # Query Ollama
    ollama_url = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")

    print("Generating analysis...\n")

    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 40,
            }
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response generated.")
        else:
            return f"Error querying Ollama: {response.status_code} - {response.text}"

    except requests.exceptions.ConnectionError:
        return (
            "Error: Could not connect to Ollama.\n"
            "Make sure Ollama is running:\n"
            "1. Install Ollama from https://ollama.ai\n"
            "2. Run: ollama serve\n"
            "3. In another terminal, pull the model: ollama pull mistral"
        )
    except Exception as e:
        return f"Error generating response: {str(e)}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/query.py \"your question here\"")
        print('Example: python scripts/query.py "What are the key findings on battery thermal management?"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"\nQuestion: {question}\n")
    print("=" * 60)

    answer = query_papers(question)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()