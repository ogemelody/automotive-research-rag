
# AutoRAG: Semantic Intelligence for Automotive Research

 Retrieval-Augmented Generation system for analyzing 
automotive and electric vehicle research papers.Then uses local Ollama to synthesize intelligent, cited answers to your technical questions.

**Ask intelligent questions. Get grounded answers with citations.**

## What It Does

- Ingests automotive research PDFs
- Generates semantic embeddings (BAAI/bge-large)
- Performs hybrid retrieval (semantic + keyword search)
- Get synthesize grounded answers
- Cites sources with paper references

![Demo](./assets/demo.gif)
## To test
### Ingest papers
python scripts/ingest_all.py

#### Query system
python scripts/query.py "What are thermal management challenges?"
