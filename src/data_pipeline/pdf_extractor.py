

import pdfplumber
import pymupdf
import pytesseract
from PIL import Image
import io
import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    filename: str
    status: str  # "success", "success_with_ocr", "error"
    pages: int
    text_chars: int
    has_tables: bool
    confidence: float  # 0-1, how confident are we in extraction
    error_msg: str = None


class PDFExtractor:
    """Multi-strategy PDF extraction for research papers"""

    def __init__(self, output_dir="data/raw/extracted_text"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extraction_log = []

    def extract_pdf(self, pdf_path: str, metadata: Dict = None) -> Dict:
        """
        Main extraction orchestrator
        Tries multiple strategies until one succeeds
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Extracting: {pdf_path.name}")

        # Strategy 1: Try pdfplumber (best for formatted text)
        result = self._try_pdfplumber(pdf_path)
        if result and result['confidence'] > 0.7:
            logger.info(f"✓ pdfplumber successful ({result['confidence']:.1%} confidence)")
            return self._save_extraction(result, pdf_path, metadata)

        # Strategy 2: Try PyMuPDF (good fallback)
        result = self._try_pymupdf(pdf_path)
        if result and result['confidence'] > 0.7:
            logger.info(f"✓ PyMuPDF successful ({result['confidence']:.1%} confidence)")
            return self._save_extraction(result, pdf_path, metadata)

        # Strategy 3: OCR for scanned PDFs
        result = self._try_ocr(pdf_path)
        if result and result['confidence'] > 0.6:
            logger.info(f"⚠ OCR used ({result['confidence']:.1%} confidence)")
            return self._save_extraction(result, pdf_path, metadata, ocr_used=True)

        # All strategies failed
        logger.error(f"✗ All extraction strategies failed for {pdf_path.name}")
        return None

    def _try_pdfplumber(self, pdf_path: Path) -> Dict:
        """
        pdfplumber: best for technical PDFs with formatting
        Good for: extracting tables, preserving layout
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_content = []
                tables = []

                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text() or ""
                    text_content.append(page_text)

                    # Extract tables (useful for battery specs, performance tables)
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table in page_tables:
                            # Convert table to readable format
                            table_text = self._table_to_text(table)
                            text_content.append(f"[TABLE]\n{table_text}")
                            tables.append({
                                "page": page_num + 1,
                                "rows": len(table),
                                "cols": len(table[0]) if table else 0
                            })

                full_text = "\n---PAGE_BREAK---\n".join(text_content)

                # Estimate confidence (more text = more likely good extraction)
                confidence = min(len(full_text) / (len(pdf.pages) * 500), 1.0)
                # Heuristic: 500 chars/page is healthy, more is confidence boost

                return {
                    'text': full_text,
                    'num_pages': len(pdf.pages),
                    'tables': tables,
                    'confidence': confidence,
                    'strategy': 'pdfplumber'
                }
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
            return None

    def _try_pymupdf(self, pdf_path: Path) -> Dict:
        """
        PyMuPDF (fitz): fast, good for most PDFs
        Good for: speed, reliability
        """
        try:
            pdf_document = pymupdf.open(pdf_path)
            text_content = []

            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                page_text = page.get_text()
                text_content.append(page_text)

            full_text = "\n---PAGE_BREAK---\n".join(text_content)
            pdf_document.close()

            confidence = min(len(full_text) / (pdf_document.page_count * 500), 1.0)

            return {
                'text': full_text,
                'num_pages': pdf_document.page_count,
                'tables': [],
                'confidence': confidence,
                'strategy': 'pymupdf'
            }
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}")
            return None

    def _try_ocr(self, pdf_path: Path) -> Dict:
        """
        OCR for scanned PDFs (worst case)
        Good for: scanned research papers that don't extract
        WARNING: Slow, requires Tesseract installed
        """
        try:
            # Check if Tesseract is available
            pytesseract.pytesseract.get_tesseract_version()
        except Exception:
            logger.warning("Tesseract not installed, skipping OCR")
            return None

        try:
            pdf_document = pymupdf.open(pdf_path)
            text_content = []

            for page_num in range(pdf_document.page_count):
                # Convert PDF page to image
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # 2x zoom for better OCR
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # OCR the image
                page_text = pytesseract.image_to_string(img)
                text_content.append(page_text)

            full_text = "\n---PAGE_BREAK---\n".join(text_content)
            pdf_document.close()

            # Lower confidence for OCR (always has errors)
            confidence = 0.6

            return {
                'text': full_text,
                'num_pages': pdf_document.page_count,
                'tables': [],
                'confidence': confidence,
                'strategy': 'ocr'
            }
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return None

    def _table_to_text(self, table: List[List]) -> str:
        """Convert extracted table to readable text"""
        if not table:
            return ""

        # Simple text representation
        lines = []
        for row in table:
            cells = [str(cell or "").strip() for cell in row]
            lines.append(" | ".join(cells))

        return "\n".join(lines)

    def _save_extraction(self, result: Dict, pdf_path: Path,
                         metadata: Dict = None, ocr_used: bool = False) -> Dict:
        """Save extraction to JSON"""

        output_data = {
            "metadata": {
                "filename": pdf_path.name,
                "extraction_strategy": result['strategy'],
                "ocr_used": ocr_used,
                "num_pages": result['num_pages'],
                "extracted_text_chars": len(result['text']),
                "num_tables": len(result['tables']),
                "extraction_confidence": result['confidence'],
                **metadata  # Include paper metadata
            },
            "content": {
                "full_text": result['text'],
                "tables": result['tables']
            }
        }

        # Save JSON
        output_path = self.output_dir / f"{pdf_path.stem}_extracted.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        # Log result
        self.extraction_log.append(ExtractionResult(
            filename=pdf_path.name,
            status="success_with_ocr" if ocr_used else "success",
            pages=result['num_pages'],
            text_chars=len(result['text']),
            has_tables=len(result['tables']) > 0,
            confidence=result['confidence']
        ))

        return output_data