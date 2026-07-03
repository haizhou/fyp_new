"""Cheap PDF text-layer probing.

The probe is intentionally local and optional. It uses PyMuPDF when available
to estimate whether a PDF has usable embedded text before any OCR is attempted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PDFTextProbe:
    path: str
    ok: bool
    page_count: int = 0
    extracted_chars: int = 0
    chars_per_page: float = 0.0
    needs_ocr: bool = False
    error_type: str = ""
    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def probe_pdf_text_layer(path: Path, min_chars_per_page: int = 80) -> PDFTextProbe:
    """Return a cheap estimate of whether a PDF has an embedded text layer."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        return PDFTextProbe(
            path=str(path),
            ok=False,
            needs_ocr=True,
            error_type="ImportError",
            error_message="PyMuPDF is not installed; install pymupdf to probe PDFs locally.",
        )

    try:
        with fitz.open(path) as doc:
            page_count = int(doc.page_count)
            extracted_chars = 0
            for page in doc:
                extracted_chars += len((page.get_text("text") or "").strip())
            chars_per_page = extracted_chars / page_count if page_count else 0.0
            return PDFTextProbe(
                path=str(path),
                ok=True,
                page_count=page_count,
                extracted_chars=extracted_chars,
                chars_per_page=round(chars_per_page, 3),
                needs_ocr=page_count > 0 and chars_per_page < min_chars_per_page,
            )
    except Exception as exc:  # pragma: no cover - depends on arbitrary PDF files
        return PDFTextProbe(
            path=str(path),
            ok=False,
            needs_ocr=True,
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
        )


__all__ = ["PDFTextProbe", "probe_pdf_text_layer"]

