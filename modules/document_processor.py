"""
modules/document_processor.py
------------------------------
MODULE 1: DOCUMENT UPLOAD & TEXT EXTRACTION

Connection to the adaptive system:
Every downstream detector (exact, TF-IDF, n-gram, semantic) needs clean,
reliably-extracted plain text. This module is the single entry point that
converts a PDF / DOCX / TXT upload into a `Document` object carrying BOTH
the raw text (needed later for sentence-level highlighting in Module 6)
and metadata used by the database layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber
from docx import Document as DocxDocument

from utils.validators import validate_file_extension, validate_non_empty_text
from utils.helpers import load_config, safe_filename, now_iso


@dataclass
class Document:
    """In-memory representation of an uploaded document."""

    name: str
    raw_text: str                      # original text, untouched — needed for highlighting
    source_path: Optional[str] = None
    document_id: Optional[int] = None  # set once persisted to DB
    upload_date: str = field(default_factory=now_iso)

    def __len__(self) -> int:
        return len(self.raw_text)


class DocumentProcessor:
    """Extracts text from PDF / DOCX / TXT uploads."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.allowed_extensions = self.config["app"]["allowed_extensions"]

    # -- public API -----------------------------------------------------

    def load(self, file_path: str | Path, display_name: str | None = None) -> Document:
        """Load a document from disk, auto-detecting the extraction method
        from its file extension."""
        file_path = Path(file_path)
        validate_file_extension(file_path.name, self.allowed_extensions)

        ext = file_path.suffix.lower()
        if ext == ".pdf":
            text = self._extract_pdf(file_path)
        elif ext == ".docx":
            text = self._extract_docx(file_path)
        elif ext == ".txt":
            text = self._extract_txt(file_path)
        else:  # pragma: no cover - validate_file_extension already guards this
            raise ValueError(f"Unsupported extension: {ext}")

        name = display_name or safe_filename(file_path.name)
        validate_non_empty_text(text, source_name=name)

        return Document(name=name, raw_text=text, source_path=str(file_path))

    def load_from_bytes(self, file_bytes: bytes, filename: str) -> Document:
        """Used by the Streamlit uploader, which hands us bytes rather than
        a path on disk. Writes to a temp file so we can reuse the same
        extraction code paths (pdfplumber/python-docx require file-like
        objects or paths)."""
        import io

        validate_file_extension(filename, self.allowed_extensions)
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            text = self._extract_pdf(io.BytesIO(file_bytes))
        elif ext == ".docx":
            text = self._extract_docx(io.BytesIO(file_bytes))
        elif ext == ".txt":
            text = file_bytes.decode("utf-8", errors="ignore")
        else:  # pragma: no cover
            raise ValueError(f"Unsupported extension: {ext}")

        name = safe_filename(filename)
        validate_non_empty_text(text, source_name=name)
        return Document(name=name, raw_text=text)

    # -- extraction backends ---------------------------------------------

    @staticmethod
    def _extract_pdf(source) -> str:
        pages_text = []
        with pdfplumber.open(source) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                pages_text.append(page_text)
        return "\n".join(pages_text)

    @staticmethod
    def _extract_docx(source) -> str:
        doc = DocxDocument(source)
        paragraphs = [p.text for p in doc.paragraphs]
        # also pull text out of tables, which python-docx's .paragraphs skips
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        paragraphs.append(cell.text)
        return "\n".join(paragraphs)

    @staticmethod
    def _extract_txt(source: Path) -> str:
        return source.read_text(encoding="utf-8", errors="ignore")
