"""main.py

End-to-end invoice pipeline: PDF -> extract -> format -> Excel.

Two extraction engines are available:

* ``"text"`` - pdfplumber text/table extraction (fast, for digital PDFs);
* ``"ocr"``  - PaddleOCR pipeline for scanned/image PDFs (see
  :mod:`extractor_ocr`);
* ``"auto"`` - pick automatically: OCR when the PDF has no usable text layer
  (empty text or ``(cid:...)`` font-encoding noise).
"""

from __future__ import annotations

import sys
from pathlib import Path

import exporter
import extractor_ocr
import extractor_pdf
import formatter


# Extraction engines accepted by :func:`run`.
ENGINES = ("auto", "text", "ocr")


def _resolve_engine(engine, pdf_path):
    """Return ``True`` when the OCR pipeline should be used for ``pdf_path``."""
    engine = (engine or "auto").lower()
    if engine == "ocr":
        return True
    if engine == "text":
        return False
    if engine != "auto":
        raise ValueError(f"Unknown engine {engine!r}; expected one of {ENGINES}.")
    # NOTE: raw text is required here - get_pdf_text() strips the (cid:...)
    # markers that should_use_ocr() relies on to detect font-encoding noise.
    text = extractor_pdf.get_raw_pdf_text(pdf_path)
    return extractor_ocr.should_use_ocr(text, pdf_path)


def run(pdf_path, supplier_key=None, output_path=None, engine="auto",
        ocr_engine=None, ocr_resolution=None, ocr_max_pages=None):
    """Run the full pipeline for a single invoice PDF.

    Args:
        pdf_path (str or Path): Path to the invoice PDF.
        supplier_key (str, optional): Supplier key used to look up the column
            mapping. When omitted, the supplier is auto-detected from the PDF
            text using the identifiers in ``suppliers_rules.yaml``.
        output_path (str or Path, optional): Destination ``.xlsx``; defaults to
            the PDF path with an ``.xlsx`` extension.
        engine (str): Extraction engine - ``"auto"`` (default), ``"text"`` or
            ``"ocr"``.
        ocr_engine (object, optional): OCR engine override (advanced/testing);
            must expose ``recognize(image)``.
        ocr_resolution (int, optional): OCR rendering DPI override.
        ocr_max_pages (int, optional): Limit the number of pages OCR processes.

    Returns:
        tuple[Path, pd.DataFrame, dict, str]: (output path, formatted
        DataFrame, summary dict, supplier key). Unknown suppliers use the
        dynamic/fuzzy mapper; OCR runs always use it. The key is
        ``extractor_pdf.DYNAMIC_SUPPLIER_KEY`` when no profile matches.
    """
    pdf_path = Path(pdf_path)
    rules = formatter.load_rules()
    suppliers = rules.get("suppliers", {})
    standard_columns = list(rules["standard_columns"])

    if _resolve_engine(engine, pdf_path):
        # OCR pipeline (scanned / image PDFs, or forced via engine="ocr").
        ocr_kwargs = {}
        if ocr_resolution is not None:
            ocr_kwargs["resolution"] = ocr_resolution
        if ocr_max_pages is not None:
            ocr_kwargs["max_pages"] = ocr_max_pages

        raw, summary, ocr_text = extractor_ocr.extract_transactions_from_pdf(
            pdf_path, rules=rules, engine=ocr_engine, **ocr_kwargs
        )
        if supplier_key is None:
            supplier_key = formatter.detect_supplier(ocr_text, rules)
        # OCR header text is noisy, so always use the fuzzy dynamic mapper.
        formatted = formatter.format_dynamic(raw, standard_columns=standard_columns)
        if supplier_key not in suppliers:
            supplier_key = extractor_pdf.DYNAMIC_SUPPLIER_KEY
    else:
        # Text pipeline.
        if supplier_key is None:
            text = extractor_pdf.get_pdf_text(pdf_path)
            supplier_key = formatter.detect_supplier(text, rules)

        # A known supplier uses its configured profile; anything else (detection
        # returned None, or an unknown key was passed) engages the Smart
        # Fallback Extractor so new PDFs still produce a standard workbook.
        if supplier_key in suppliers:
            transactions, summary = extractor_pdf.extract_tables_from_pdf(pdf_path)
            formatted = formatter.format_transactions(transactions, supplier_key, rules)
        else:
            formatted, summary = extractor_pdf.smart_fallback_extract(pdf_path, rules)
            supplier_key = extractor_pdf.DYNAMIC_SUPPLIER_KEY

    if output_path is None:
        output_path = pdf_path.with_suffix(".xlsx")
    output_path = exporter.export_to_excel(formatted, output_path, summary=summary)

    return output_path, formatted, summary, supplier_key


if __name__ == "__main__":
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "egypt gate (1).pdf"
    supplier_key = sys.argv[2] if len(sys.argv) > 2 else None
    engine = sys.argv[3] if len(sys.argv) > 3 else "auto"

    pdf_path = Path(pdf_file)
    if not pdf_path.is_absolute():
        pdf_path = Path(__file__).resolve().parent / pdf_path

    if not pdf_path.exists():
        candidates = sorted(Path(__file__).resolve().parent.glob("*.pdf"))
        if candidates:
            print(
                f"Note: '{pdf_file}' not found; "
                f"using '{candidates[0].name}' instead.\n"
            )
            pdf_path = candidates[0]
        else:
            raise SystemExit(f"PDF file not found: {pdf_path}")

    output_path, formatted, summary, supplier_key = run(
        pdf_path, supplier_key=supplier_key, engine=engine
    )

    print(f"Engine: {engine}")
    print(f"Detected supplier: {supplier_key}")
    print(f"Extracted {len(formatted)} transaction rows (standard schema).")
    print("First rows:")
    print(formatted.head().to_string(index=False))
    print(f"\nSummary: {summary}")
    print(f"\nExcel written to: {output_path}")
