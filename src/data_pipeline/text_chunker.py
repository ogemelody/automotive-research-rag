import spacy
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    content: str
    chunk_id: str
    paper_id: str
    token_count: int


class TextChunker:
    def __init__(self, chunk_size=750, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.max_length = 2000000

    def count_tokens(self, text: str) -> int:
        """Approximate: 1 token ≈ 4 chars"""
        return len(text) // 4

    def chunk_paper(self, text: str, paper_id: str) -> List[Chunk]:
        """Split into sentence-aware chunks"""
        doc = self.nlp(text[:1000000])
        sentences = [sent.text for sent in doc.sents]

        chunks = []
        current_chunk = []
        chunk_id = 0

        for sentence in sentences:
            current_chunk.append(sentence)
            token_count = self.count_tokens(" ".join(current_chunk))

            if token_count >= self.chunk_size:
                # Save chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(Chunk(
                    content=chunk_text,
                    chunk_id=f"{paper_id}_chunk_{chunk_id}",
                    paper_id=paper_id,
                    token_count=token_count
                ))
                chunk_id += 1

                # Keep last 20% for overlap
                overlap_sents = max(1, len(current_chunk) // 5)
                current_chunk = current_chunk[-overlap_sents:]

        # Final chunk
        if current_chunk:
            chunks.append(Chunk(
                content=" ".join(current_chunk),
                chunk_id=f"{paper_id}_chunk_{chunk_id}",
                paper_id=paper_id,
                token_count=self.count_tokens(" ".join(current_chunk))
            ))

        return chunks
