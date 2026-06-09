# src/data_pipeline/quality_gates.py

class QualityGates:
    """Enforce quality standards at each stage"""

    EXTRACTION_THRESHOLD = 0.70  # Must be 70% confident
    CLEANING_THRESHOLD = 0.65  # Quality score must be good
    CHUNKING_THRESHOLD = 0.80  # At least 80% chunks above min tokens
    EMBEDDING_THRESHOLD = 0.95  # 95% embeddings generated successfully

    @staticmethod
    def gate_extraction(extraction_result) -> bool:
        confidence = extraction_result['metadata']['extraction_confidence']
        if confidence < QualityGates.EXTRACTION_THRESHOLD:
            logger.warning(f"Low extraction confidence: {confidence:.1%}")
            return False
        return True

    @staticmethod
    def gate_cleaning(quality_metrics) -> bool:
        score = quality_metrics['quality_score']
        if score < QualityGates.CLEANING_THRESHOLD:
            logger.warning(f"Low cleaning quality: {score:.1%}")
            return False
        return True

    @staticmethod
    def gate_chunking(chunks) -> bool:
        min_chunk_size = 300  # tokens
        valid_chunks = sum(1 for c in chunks if c['token_count'] >= min_chunk_size)
        valid_ratio = valid_chunks / len(chunks)

        if valid_ratio < QualityGates.CHUNKING_THRESHOLD:
            logger.warning(f"Too many small chunks: {valid_ratio:.1%} valid")
            return False
        return True


# Usage in pipeline
if not QualityGates.gate_extraction(extraction_result):
    logger.error("Paper failed extraction quality gate")
    # Option 1: Use OCR
    # Option 2: Flag for manual review
    # Option 3: Skip paper
    continue

if not QualityGates.gate_cleaning(cleaning_quality):
    logger.error("Paper failed cleaning quality gate")
    # Investigate: Why is extracted text so bad?
    continue

if not QualityGates.gate_chunking(chunks):
    logger.error("Paper failed chunking quality gate")
    # Adjust chunk size and retry
    continue