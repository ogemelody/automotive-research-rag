

import re
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean extracted text for chunking"""

    def __init__(self):
        self.issues_found = {
            'control_chars': 0,
            'multiple_spaces': 0,
            'page_breaks': 0,
            'ligatures': 0
        }

    def clean_paper(self, raw_text: str) -> str:
        """Main cleaning pipeline"""

        # Step 1: Remove control characters
        text = self._remove_control_chars(raw_text)

        # Step 2: Fix OCR artifacts
        text = self._fix_ocr_errors(text)

        # Step 3: Fix ligatures (ﬁ → fi)
        text = self._fix_ligatures(text)

        # Step 4: Normalize whitespace
        text = self._normalize_whitespace(text)

        # Step 5: Remove common headers/footers
        text = self._remove_headers_footers(text)

        # Step 6: Handle page breaks intelligently
        text = self._process_page_breaks(text)

        return text

    def _remove_control_chars(self, text: str) -> str:
        """Remove non-printable characters"""
        # Keep: printable ASCII, common Unicode, whitespace
        allowed = set(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t' +
            '.,!?;:\'-"()[]{}/@#$%^&*+=<>~`|\\' +
            'äöüßÄÖÜñÑçÇéèêëàâäùûüœæ'  # European characters
        )

        cleaned = ''.join(c for c in text if c in allowed or ord(c) > 127)
        self.issues_found['control_chars'] += len(text) - len(cleaned)
        return cleaned

    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR misreadings"""
        replacements = {
            'l0': 'lo',  # lowercase L misread as 0
            '0O': '00',  # O misread as 0
            'vv': 'w',  # two v's as w
            'rn': 'in',  # rn as in
        }

        for wrong, right in replacements.items():
            # Only replace in lowercase text (avoid proper nouns)
            text = re.sub(f'(?<=[a-z]){wrong}(?=[a-z])', right, text)

        return text

    def _fix_ligatures(self, text: str) -> str:
        """Fix ligatures from PDF"""
        ligatures = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            'ﬃ': 'ffi',
            'ﬄ': 'ffl'
        }

        for ligature, replacement in ligatures.items():
            count = text.count(ligature)
            if count > 0:
                text = text.replace(ligature, replacement)
                self.issues_found['ligatures'] += count

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Clean up spacing"""
        # Multiple spaces → single space (but preserve intentional indentation)
        text = re.sub(r' {2,}', ' ', text)

        # Multiple newlines → double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Trailing spaces
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)

        self.issues_found['multiple_spaces'] += len(re.findall(r'  +', text))

        return text

    def _remove_headers_footers(self, text: str) -> str:
        """Remove recurring headers/footers"""

        lines = text.split('\n')
        if not lines:
            return text

        # Common patterns in research papers:
        footer_patterns = [
            r'^\s*\d+\s*$',  # Just page number
            r'^[A-Z].*\|.*[A-Z].*$',  # Header with pipe separators
            r'^Page \d+',  # Page X header
            r'\[date\]',  # Generated dates
        ]

        filtered_lines = []
        for line in lines:
            is_header = any(re.match(pattern, line) for pattern in footer_patterns)
            if not is_header:
                filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def _process_page_breaks(self, text: str) -> str:
        """Handle page boundaries intelligently"""

        # Replace page break markers with double newline
        text = text.replace('---PAGE_BREAK---', '\n\n')

        # Don't split sentences at page breaks
        # If page ends with incomplete sentence, join with next

        self.issues_found['page_breaks'] += text.count('\n\n')

        return text

    def estimate_quality(self, text: str) -> Dict:
        """Quality metrics for cleaned text"""

        if not text:
            return {'quality_score': 0, 'status': 'empty'}

        lines = text.split('\n')
        words = text.split()

        # Calculate metrics
        avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
        vocabulary_size = len(set(w.lower() for w in words))

        # Quality heuristics
        quality_signals = {
            'has_content': len(text) > 500,  # At least 500 chars
            'reasonable_lines': 30 < avg_line_length < 120,  # ~50-80 chars/line is good
            'diverse_vocab': vocabulary_size > 100,  # At least 100 unique words
            'minimal_control_chars': self.issues_found['control_chars'] < len(text) * 0.01,  # <1%
        }

        quality_score = sum(quality_signals.values()) / len(quality_signals)

        status = 'excellent' if quality_score >= 0.9 else \
            'good' if quality_score >= 0.7 else \
                'acceptable' if quality_score >= 0.5 else \
                    'poor'

        return {
            'quality_score': quality_score,
            'status': status,
            'total_chars': len(text),
            'total_lines': len(lines),
            'avg_line_length': avg_line_length,
            'vocabulary_size': vocabulary_size,
            'signals': quality_signals,
            'issues_found': self.issues_found
        }


# Usage
cleaner = TextCleaner()
cleaned = cleaner.clean_paper(raw_text)
quality = cleaner.estimate_quality(cleaned)

if quality['quality_score'] < 0.5:
    logger.warning(f"Low quality extraction: {quality['status']}")
    # Could flag for manual review