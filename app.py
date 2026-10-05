"""app.py

Streamlit web interface for the local invoice extractor pipeline.

Run locally with:

    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import extractor_ocr
import formatter
import main


# Must be the first Streamlit command.
st.set_page_config(
    page_title="Invoice & Table Extractor Pro",
    page_icon="🧾",
    layout="wide",
)


@st.cache_data
def _rules():
    """Load the supplier rules YAML (cached)."""
    return formatter.load_rules()


@st.cache_data
def _supplier_keys():
    return list(_rules().get("suppliers", {}).keys())


@st.cache_data
def _standard_columns():
    return list(_rules().get("standard_columns", []))


def process_pdf(pdf_bytes: bytes, filename: str, supplier_override: str | None,
                engine: str = "auto"):
    """Run the full pipeline on a single uploaded PDF or image (jpg/png).

    Args:
        pdf_bytes: Raw PDF/image file bytes.
        filename: Original uploaded filename (used for the temp file + xlsx).
        supplier_override: Explicit supplier key, or ``None`` to auto-detect.
        engine: Extraction engine - ``"auto"``, ``"text"`` or ``"ocr"``.

    Returns:
        tuple: (formatted DataFrame, summary dict, supplier key, xlsx bytes).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        pdf_path = tmp / filename
        pdf_path.write_bytes(pdf_bytes)

        xlsx_path = tmp / f"{Path(filename).stem}.xlsx"

        try:
            out_path, formatted, summary, supplier_key = main.run(
                pdf_path,
                supplier_key=supplier_override,
                output_path=xlsx_path,
                engine=engine,
            )
        except extractor_ocr.OcrUnavailableError as exc:
            raise ValueError(str(exc)) from exc

        return formatted, summary, supplier_key, out_path.read_bytes()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown("**Processing mode**")
    mode = st.radio(
        "Input type",
        options=["Auto-detect", "PDF (text-based)", "Image / Scanned (OCR)"],
        index=0,
        help=(
            "Auto-detect uses the text layer when present and falls back to "
            "OCR for scanned PDFs. Force 'Image / Scanned (OCR)' for "
            "photographed or image-only invoices; force 'PDF (text-based)' to "
            "skip OCR."
        ),
    )
    engine = {
        "Auto-detect": "auto",
        "PDF (text-based)": "text",
        "Image / Scanned (OCR)": "ocr",
    }[mode]

    if extractor_ocr.is_ocr_available():
        st.caption("🖼️ OCR ready (PaddleOCR · Arabic).")
    else:
        st.caption(
            "ℹ️ OCR backend (PaddleOCR) not installed — the OCR mode will "
            "report an error. See requirements.txt."
        )

    st.divider()

    supplier_choice = st.selectbox(
        "Supplier",
        options=["Auto-detect"] + _supplier_keys(),
        index=0,
        help=(
            "Auto-detect the supplier from the PDF text, or force a specific "
            "profile. Unknown suppliers are handled automatically by the "
            "dynamic (fuzzy) fallback extractor."
        ),
    )
    supplier_override = None if supplier_choice == "Auto-detect" else supplier_choice

    st.divider()

    with st.expander("📋 Standard schema"):
        st.write(_standard_columns())


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🧾 Invoice & Table Extractor Pro")
st.caption("Upload invoice PDFs or image scans (JPG/JPEG/PNG) and export a standardized Excel workbook.")

uploaded_files = st.file_uploader(
    "Upload invoice file(s) (PDF or image)",
    type=["pdf", "PDF", "jpg", "JPG", "jpeg", "JPEG", "png", "PNG"],
    accept_multiple_files=True,
    help="Select one or more invoice files to process: PDF documents or direct images (jpg/jpeg/png).",
)

process_clicked = st.button(
    "🚀 Process Invoices",
    type="primary",
    disabled=not uploaded_files,
)

if process_clicked:
    results = []
    total = len(uploaded_files)
    progress = st.progress(0.0, text=f"Processing 0/{total}…")

    for idx, uploaded in enumerate(uploaded_files, start=1):
        progress.progress(
            (idx - 1) / total, text=f"Processing {uploaded.name} ({idx}/{total})…"
        )
        try:
            formatted, summary, supplier_key, xlsx_bytes = process_pdf(
                uploaded.getvalue(), uploaded.name, supplier_override, engine
            )
            results.append(
                {
                    "name": uploaded.name,
                    "ok": True,
                    "supplier": supplier_key,
                    "rows": len(formatted),
                    "formatted": formatted,
                    "summary": summary,
                    "xlsx": xlsx_bytes,
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
            results.append(
                {
                    "name": uploaded.name,
                    "ok": False,
                    "supplier": None,
                    "rows": 0,
                    "formatted": None,
                    "summary": None,
                    "xlsx": None,
                    "error": str(exc),
                }
            )
        progress.progress(idx / total, text=f"Processed {uploaded.name}")

    progress.empty()
    st.session_state["results"] = results


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
results = st.session_state.get("results")
if results:
    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    for r in failed:
        st.error(f"❌ **{r['name']}** — {r['error']}")

    if ok:
        st.success(f"✅ Processed {len(ok)} of {len(results)} invoice(s) successfully.")

        combined = pd.concat([r["formatted"] for r in ok], ignore_index=True)
        st.subheader("📄 Extracted table (standard schema)")
        st.dataframe(combined)

        st.subheader("⬇️ Download Excel")
        for i, r in enumerate(ok):
            stem = Path(r["name"]).stem
            st.download_button(
                label=f"📥 Download {stem}.xlsx",
                data=r["xlsx"],
                file_name=f"{stem}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_{i}_{stem}",
            )
            st.caption(f"Supplier: `{r['supplier']}` · {r['rows']} rows")

    if failed and not ok:
        st.info("No invoices could be processed. Review the errors above and try again.")

    if st.button("🧹 Clear results"):
        st.session_state.pop("results", None)
        st.rerun()
