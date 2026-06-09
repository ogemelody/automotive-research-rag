

from src.data_pipeline.pdf_extractor import PDFExtractor
from src.data_pipeline.text_cleaner import TextCleaner
from src.data_pipeline.text_chunker import TextChunker
from src.embeddings.chroma_setup import ChromaVectorDB
from src.embeddings.generate_embeddings import EmbeddingGenerator
from src.data_pipeline.incremental_updates import IncementalPipelineManager
from src.data_pipeline.quality_gates import QualityGates
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_full_pipeline():
    """End-to-end: Papers → Vector DB"""

    logger.info("=" * 60)
    logger.info("STARTING FULL INGESTION PIPELINE")
    logger.info("=" * 60)

    # Load paper inventory
    papers_df = pd.read_csv("data/papers_inventory.csv")

    # Initialize components
    extractor = PDFExtractor()
    cleaner = TextCleaner()
    chunker = TextChunker()
    embedding_gen = EmbeddingGenerator()
    vector_db = ChromaVectorDB()
    update_manager = IncementalPipelineManager()

    # Track results
    all_chunks = []
    pipeline_results = []

    # Process papers
    for idx, row in papers_df.iterrows():
        paper_id = row['filename'].replace('.pdf', '')
        pdf_path = Path("data/raw/papers") / row['filename']

        logger.info(f"\n[{idx + 1}/{len(papers_df)}] {row['title']}")

        # Check if already processed
        if not update_manager.should_reprocess(pdf_path):
            continue

        # STAGE 1: Extraction
        logger.info("  → Extracting...")
        extracted = extractor.extract_pdf(pdf_path, row.to_dict())
        if not extracted or not QualityGates.gate_extraction(extracted):
            pipeline_results.append({'paper_id': paper_id, 'stage': 'extraction', 'status': 'failed'})
            continue

        raw_text = extracted['content']['full_text']

        # STAGE 2: Cleaning
        logger.info("  → Cleaning...")
        cleaned_text = cleaner.clean_paper(raw_text)
        cleaning_quality = cleaner.estimate_quality(cleaned_text)
        if not QualityGates.gate_cleaning(cleaning_quality):
            pipeline_results.append({'paper_id': paper_id, 'stage': 'cleaning', 'status': 'failed',
                                     'quality': cleaning_quality['quality_score']})
            continue

        # STAGE 3: Chunking
        logger.info("  → Chunking...")
        chunks = chunker.chunk_paper(cleaned_text, paper_id)
        if not QualityGates.gate_chunking(chunks):
            pipeline_results.append({'paper_id': paper_id, 'stage': 'chunking', 'status': 'failed'})
            continue

        logger.info(f"  ✓ Created {len(chunks)} chunks")

        # STAGE 4: Embeddings
        logger.info("  → Generating embeddings...")
        chunk_dicts = [c.to_dict() for c in chunks]
        embeddings = embedding_gen.embed_batch([c['content'] for c in chunk_dicts])

        # STAGE 5: Store in Vector DB
        logger.info("  → Storing in database...")
        vector_db.add_chunks(chunk_dicts, embeddings)

        # Mark as processed
        update_manager.mark_processed(pdf_path, len(chunks))
        all_chunks.extend(chunk_dicts)

        pipeline_results.append({
            'paper_id': paper_id,
            'stage': 'complete',
            'status': 'success',
            'chunks': len(chunks),
            'quality': cleaning_quality['quality_score']
        })

        logger.info(f"  ✓ Paper complete")

    # Final summary
    logger.info(f"\n{'=' * 60}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total papers processed: {len([r for r in pipeline_results if r['status'] == 'success'])}")
    logger.info(f"Total chunks created: {len(all_chunks)}")
    logger.info(f"Vector DB status: {vector_db.get_collection_info()}")

    # Save results
    results_df = pd.DataFrame(pipeline_results)
    results_df.to_csv("data/pipeline_results.csv", index=False)

    # Create and save manifest
    manifest = create_data_manifest(papers_df, all_chunks, embedding_gen)
    with open("data/data_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Results saved: data/pipeline_results.csv")
    logger.info(f"Manifest saved: data/data_manifest.json")


if __name__ == "__main__":
    run_full_pipeline()