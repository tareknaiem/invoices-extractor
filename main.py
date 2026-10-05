"""main.py

End-to-end invoice pipeline: PDF -> extract -> format -> Excel.

Two extraction engines are available:

* ``"text"`` - pdfplumber text/table extraction (fast, for digital PDFs);
* ``"ocr"``  - PaddleOCR pipeline for scanned/image PDFs (see
  :mod:`extractor_ocr`);
* ``"auto"`` - pick automatically: OCR when the PDF has no usable text layer
  (empty text or ``(cid:...)`` font-encoding noise).

Direct image inputs (``.jpg``/``.jpeg``/``.png``) have no text layer at all, so
they always run through the OCR pipeline regardless of ``engine``.

Hybrid mode: profiles first, Smart Fallback for unknown suppliers
-----------------------------------------------------------------
Text PDFs are matched against the supplier profiles in ``suppliers_rules.yaml``
via :func:`formatter.detect_supplier`:

* a **known** supplier uses its curated ``columns_mapping``
  (:func:`formatter.format_transactions`);
* an **unknown** one falls back to the Smart Fallback extractor
  (:func:`extractor_pdf.smart_fallback_extract`, i.e.
  :func:`formatter.format_dynamic`).

Profiles win when they match because the curated mappings recover columns the
fuzzy matcher misses (measured on the sample invoices: 59 vs 50 populated
``Total`` values, and ``V.NO`` 28 vs 0 on ``egypt_gate``).

OCR/image inputs keep using :func:`formatter.format_dynamic`: recognised header
text is noisy, so a curated profile would mis-map it.
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

# UI/CLI labels describing which mapping strategy produced a workbook.
PROFILE_LABEL = "profile"
FALLBACK_LABEL = "Smart Fallback"
UNKNOWN_SUPPLIER_LABEL = "Unknown"


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


def count_detected_columns(formatted, standard_columns=None, rules=None):
    """Return the number of standard columns the mapper actually populated.

    A column counts as "detected" when at least one row holds a non-null,
    non-blank value. This is surfaced in the UI/CLI so users can tell a well
    recognised invoice from a mostly empty one.

    Args:
        formatted (pd.DataFrame): Standard-schema DataFrame from the pipeline.
        standard_columns (list[str], optional): Schema to inspect; defaults to
            the rules' ``standard_columns``.
        rules (dict, optional): Parsed rules, used when the schema is omitted.

    Returns:
        int: Count of populated standard columns (0 for an empty frame).
    """
    if formatted is None or getattr(formatted, "empty", True):
        return 0
    if standard_columns is None:
        if rules is None:
            rules = formatter.load_rules()
        standard_columns = list(rules["standard_columns"])

    detected = 0
    for column in standard_columns:
        if column not in formatted.columns:
            continue
        series = formatted[column]
        if series.notna().any() and series.astype(str).str.strip().ne("").any():
            detected += 1
    return detected


def supplier_label(supplier_key):
    """Return the user-facing label for the supplier that was used.

    Mirrors what the UI/CLI print, e.g. ``"egypt_gate (profile)"`` or
    ``"Unknown (Smart Fallback)"``.

    Args:
        supplier_key (str or None): The key returned by :func:`run`.

    Returns:
        str: ``"<supplier> (profile)"`` for a matched profile, or
        ``"Unknown (Smart Fallback)"`` when the dynamic mapper was used.
    """
    if supplier_key and supplier_key != extractor_pdf.DYNAMIC_SUPPLIER_KEY:
        return f"{supplier_key} ({PROFILE_LABEL})"
    return f"{UNKNOWN_SUPPLIER_LABEL} ({FALLBACK_LABEL})"


def run(pdf_path, supplier_key=None, output_path=None, engine="auto",
        ocr_engine=None, ocr_resolution=None, ocr_max_pages=None):
    """Run the full pipeline for a single invoice PDF (hybrid mode).

    Text PDFs use the curated supplier profile when the supplier is known
    (:func:`formatter.detect_supplier` + :func:`formatter.format_transactions`)
    and the Smart Fallback extractor otherwise. OCR/image inputs always use the
    fuzzy dynamic mapper because recognised headers are noisy.

    Args:
        pdf_path (str or Path): Path to the invoice PDF or image file (jpg/png).
        supplier_key (str, optional): Force a specific profile. ``None`` (the
            default) auto-detects the supplier from the PDF text; pass
            ``extractor_pdf.DYNAMIC_SUPPLIER_KEY`` to force the Smart Fallback.
        output_path (str or Path, optional): Destination ``.xlsx``; defaults to
            the PDF path with an ``.xlsx`` extension.
        engine (str): Extraction engine - ``"auto"`` (default), ``"text"`` or
            ``"ocr"``. Direct image files (jpg/png) always use OCR and ignore
            this flag plus ``ocr_resolution``/``ocr_max_pages``.
        ocr_engine (object, optional): OCR engine override (advanced/testing);
            must expose ``recognize(image)``.
        ocr_resolution (int, optional): OCR rendering DPI override.
        ocr_max_pages (int, optional): Limit the number of pages OCR processes.

    Returns:
        tuple[Path, pd.DataFrame, dict, str]: (output path, formatted
        DataFrame, summary dict, supplier key). The key is the matched supplier
        when a profile was used, else ``extractor_pdf.DYNAMIC_SUPPLIER_KEY``.
    """
    pdf_path = Path(pdf_path)
    rules = formatter.load_rules()
    suppliers = rules.get("suppliers", {})
    standard_columns = list(rules["standard_columns"])

    is_image = extractor_ocr.is_image_path(pdf_path)
    if is_image or _resolve_engine(engine, pdf_path):
        # OCR pipeline (scanned / image PDFs, forced via engine="ocr", or a
        # direct image upload - jpg/jpeg/png always OCR regardless of engine).
        if is_image:
            raw, summary, ocr_text = extractor_ocr.extract_transactions_from_image(
                pdf_path, engine=ocr_engine
            )
        else:
            ocr_kwargs = {}
            if ocr_resolution is not None:
                ocr_kwargs["resolution"] = ocr_resolution
            if ocr_max_pages is not None:
                ocr_kwargs["max_pages"] = ocr_max_pages

            raw, summary, ocr_text = extractor_ocr.extract_transactions_from_pdf(
                pdf_path, rules=rules, engine=ocr_engine, **ocr_kwargs
            )
        # OCR header text is noisy, so always use the fuzzy dynamic mapper.
        formatted = formatter.format_dynamic(raw, standard_columns=standard_columns)
        supplier_key = extractor_pdf.DYNAMIC_SUPPLIER_KEY
    else:
        # Text pipeline (hybrid): resolve the supplier, then pick the mapper.
        detected = supplier_key
        if detected is None:
            detected = formatter.detect_supplier(
                extractor_pdf.get_pdf_text(pdf_path), rules
            )

        if detected in suppliers:
            # Known supplier -> curated profile columns_mapping.
            transactions, summary = extractor_pdf.extract_tables_from_pdf(pdf_path)
            formatted = formatter.format_transactions(transactions, detected, rules)
            supplier_key = detected
        else:
            # Unknown supplier (or a forced fallback) -> Smart Fallback.
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

    detected = count_detected_columns(formatted)
    standard_columns = formatter.load_rules()["standard_columns"]

    print(f"Engine: {engine}")
    print(f"Supplier: {supplier_label(supplier_key)}")
    print(f"Detected columns: {detected}/{len(standard_columns)}")
    print(f"Extracted {len(formatted)} transaction rows (standard schema).")
    print("First rows:")
    print(formatted.head().to_string(index=False))
    print(f"\nSummary: {summary}")
    print(f"\nExcel written to: {output_path}")
