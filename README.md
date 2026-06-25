
# AutoRAG: Semantic Intelligence for Automotive Research

 Retrieval-Augmented Generation system for analyzing 
automotive and electric vehicle research papers.

**Ask intelligent questions. Get grounded answers with citations.**

## What It Does

- Ingests automotive research PDFs
- Generates semantic embeddings (BAAI/bge-large)
- Performs hybrid retrieval (semantic + keyword search)
- Get synthesize grounded answers
- Cites sources with paper references

## To test
### Ingest papers
python scripts/ingest_all.py

#### Query system
python scripts/query.py "What are thermal management challenges?"
