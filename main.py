"""main.py

End-to-end invoice pipeline: PDF -> extract -> format -> Excel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import exporter
import extractor_pdf
import formatter


def run(pdf_path, supplier_key=None, output_path=None):
    """Run the full pipeline for a single invoice PDF.

    Args:
        pdf_path (str or Path): Path to the invoice PDF.
        supplier_key (str, optional): Supplier key used to look up the column
            mapping. When omitted, the supplier is auto-detected from the PDF
            text using the identifiers in ``suppliers_rules.yaml``.
        output_path (str or Path, optional): Destination ``.xlsx``; defaults to
            the PDF path with an ``.xlsx`` extension.

    Returns:
        tuple[Path, pd.DataFrame, dict, str]: (output path, formatted
        DataFrame, summary dict, supplier key). When no profile matches, the
        Smart Fallback Extractor is used and the key is
        ``extractor_pdf.DYNAMIC_SUPPLIER_KEY``.
    """
    pdf_path = Path(pdf_path)
    rules = formatter.load_rules()
    suppliers = rules.get("suppliers", {})

    if supplier_key is None:
        text = extractor_pdf.get_pdf_text(pdf_path)
        supplier_key = formatter.detect_supplier(text, rules)

    # A known supplier uses its configured profile; anything else (detection
    # returned None, or an unknown key was passed) engages the Smart Fallback
    # Extractor so new PDFs still produce a standard workbook.
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
        pdf_path, supplier_key=supplier_key
    )

    print(f"Detected supplier: {supplier_key}")
    print(f"Extracted {len(formatted)} transaction rows (standard schema).")
    print("First rows:")
    print(formatted.head().to_string(index=False))
    print(f"\nSummary: {summary}")
    print(f"\nExcel written to: {output_path}")
