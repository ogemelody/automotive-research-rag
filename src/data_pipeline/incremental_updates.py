import json
import logging
from pathlib import Path
from typing import Dict, List
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


class IncementalPipelineManager:
    """Track which papers are processed, update incrementally"""

    def __init__(self, processing_log_path="data/processing_log.json"):
        self.log_path = Path(processing_log_path)
        self.processing_log = self._load_log()

    def _load_log(self) -> Dict:
        """Load processing history"""
        if self.log_path.exists():
            with open(self.log_path) as f:
                return json.load(f)
        return {"papers": {}, "last_updated": None}

    def _save_log(self):
        """Save processing history"""
        with open(self.log_path, 'w') as f:
            json.dump(self.processing_log, f, indent=2)

    def get_paper_hash(self, pdf_path: Path) -> str:
        """Hash of PDF file (detect if it changed)"""
        with open(pdf_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def should_reprocess(self, pdf_path: Path) -> bool:
        """
        Check if paper needs reprocessing
        Reprocess if: new paper OR file changed
        """
        filename = pdf_path.name
        current_hash = self.get_paper_hash(pdf_path)

        if filename not in self.processing_log['papers']:
            logger.info(f"New paper: {filename}")
            return True

        logged_hash = self.processing_log['papers'][filename]['file_hash']
        if current_hash != logged_hash:
            logger.info(f"Paper changed: {filename}")
            return True

        logger.info(f"Paper already processed: {filename}")
        return False

    def mark_processed(self, pdf_path: Path, chunks_created: int):
        """Mark paper as processed"""
        self.processing_log['papers'][pdf_path.name] = {
            'file_hash': self.get_paper_hash(pdf_path),
            'processed_at': datetime.now().isoformat(),
            'chunks': chunks_created
        }
        self.processing_log['last_updated'] = datetime.now().isoformat()
        self._save_log()

    def get_unprocessed_papers(self, papers_dir: Path) -> List[Path]:
        """Get all papers that need processing"""
        pdfs = list(papers_dir.glob('*.pdf'))
        return [p for p in pdfs if self.should_reprocess(p)]