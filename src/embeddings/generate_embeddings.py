from sentence_transformers import SentenceTransformer
import torch
import numpy as np


class EmbeddingGenerator:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed single text"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_batch(self, texts: list, batch_size=32) -> list:
        """Embed multiple texts efficiently"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
