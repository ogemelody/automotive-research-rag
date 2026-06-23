import sys
import os
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import anthropic
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
        context_parts.append(f"[Source {i+1} | {paper_id} | similarity: {similarity:.2f}]\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print("Generating analysis...\n")
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""You are an automotive research assistant. Answer the question below using ONLY the provided research paper excerpts. Cite which source each point comes from.

Question: {question}

Research excerpts:
{context}

Provide a clear, structured answer based on the sources above."""
            }
        ]
    )

    return message.content[0].text


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