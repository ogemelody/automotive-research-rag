import chromadb
from pathlib import Path


class ChromaVectorDB:
    def __init__(self, persist_dir="data/chroma_db"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name="automotive-research",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, texts, ids, embeddings, metadatas):
        """Add chunks with embeddings"""
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"✓ Added {len(ids)} chunks")

    def query(self, query_embedding, n_results=5):
        """Query for similar chunks"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

    def count(self):
        return self.collection.count()
