"""app.py

Streamlit web interface for the flexible AI invoice extractor.

The user types:
  - Provider name (any name)
  - Base URL (optional, from the provider's docs)
  - Model name (free text)
  - API key (password)

The key is saved to a .env file under PROVIDER_API_KEY, where PROVIDER is the
typed provider name in uppercase with spaces and dashes replaced by underscores.

Run locally with:

    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import ai_extract
import exporter


# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Flexible AI Invoice Extractor",
    page_icon="🧾",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar: provider name, base URL, model name, API key, Save key
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 Configure API")

    provider_name = st.text_input(
        "Provider name",
        placeholder="e.g. my-invoice-service",
        help="Any name you choose. Used to build the .env key name.",
    )

    base_url = st.text_input(
        "Base URL",
        placeholder="Optional - copy from provider's docs",
        help="Leave empty if the API does not need a custom address.",
    )

    model_name = st.text_input(
        "Model name",
        placeholder="As written in the provider's docs",
        help="Model name as provided by the API.",
    )

    api_key = st.text_input(
        "API Key",
        type="password",
        help="Your API key. It is saved to a .env file in the project folder.",
    )

    col_save, col_clear = st.columns([1, 1])
    with col_save:
        save_clicked = st.button("Save key", type="primary")
    with col_clear:
        clear_clicked = st.button("Clear key")

    if clear_clicked:
        env_path = Path(".env")
        if env_path.exists():
            env_path.unlink()
            st.success("API key cleared from .env")
        else:
            st.info("No .env file found to clear")

    if save_clicked:
        if not provider_name.strip():
            st.error("Please enter a provider name before saving.")
        elif not api_key.strip():
            st.error("Please enter an API key before saving.")
        else:
            # Build the env key name: PROVIDER in uppercase, spaces and dashes
            # replaced by underscores.
            key_name = provider_name.strip().upper().replace(" ", "_").replace("-", "_")
            env_path = Path(".env")
            existing = ""
            if env_path.exists():
                existing = env_path.read_text(encoding="utf-8")
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(f"{key_name}={api_key.strip()}\n")
            st.success(f"API key saved to .env as {key_name}")

    st.caption(
        "Your key is written to `.env` in this project folder. "
        "It is only used to call the API."
    )


# ---------------------------------------------------------------------------
# Main area: file uploader + process button + progress + results table
# ---------------------------------------------------------------------------
st.title("🧾 Flexible AI Invoice Extractor")
st.caption(
    "Upload one or more invoices (PDF / JPG / JPEG / PNG) and extract their data "
    "using any AI API you configure."
)

uploaded_files = st.file_uploader(
    "Upload invoices",
    type=["pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

process_clicked = st.button(
    "🚀 Process Invoices",
    type="primary",
    disabled=not uploaded_files,
)

# Validate that all required inputs are provided before processing.
if uploaded_files and process_clicked:
    if not provider_name.strip():
        st.error("Please enter a provider name in the sidebar.")
    elif not model_name.strip():
        st.error("Please enter a model name in the sidebar.")
    elif not api_key.strip():
        st.error("Please enter an API key in the sidebar.")
    else:
        # Escape hatch: if the user typed a custom provider name, we still use
        # the openai package with the provided base_url.
        if not base_url.strip():
            base_url_val = None
        else:
            base_url_val = base_url.strip()

        results = []
        total = len(uploaded_files)
        progress = st.progress(0.0, text=f"Processing 0/{total}…")

        for idx, uploaded in enumerate(uploaded_files, start=1):
            progress.progress(
                (idx - 1) / total,
                text=f"Processing {uploaded.name} ({idx}/{total})…",
            )
            try:
                # Read the key from .env (if it exists) as a fallback.
                key = api_key.strip()
                df = ai_extract.extract_invoice_to_dataframe(
                    uploaded.getvalue(),
                    uploaded.name,
                    base_url=base_url_val,
                    model=model_name.strip(),
                    api_key=key,
                )
                results.append(
                    {
                        "name": uploaded.name,
                        "ok": True,
                        "df": df,
                        "error": None,
                    }
                )
            except ValueError as exc:
                # Per-file failure: do not stop the others.
                results.append(
                    {
                        "name": uploaded.name,
                        "ok": False,
                        "df": None,
                        "error": str(exc),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
                results.append(
                    {
                        "name": uploaded.name,
                        "ok": False,
                        "df": None,
                        "error": f"Unexpected error: {exc}",
                    }
                )
            progress.progress(idx / total, text=f"Processed {uploaded.name}")

        progress.empty()
        st.session_state["results"] = results

# ---------------------------------------------------------------------------
# Results display + download
# ---------------------------------------------------------------------------
results = st.session_state.get("results")
if results:
    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    for r in failed:
        st.error(f"❌ **{r['name']}** — {r['error']}")

    if ok:
        st.success(f"✅ Processed {len(ok)} of {len(results)} invoice(s) successfully.")

        # Combine all successful extractions into one table.
        combined = pd.concat([r["df"] for r in ok], ignore_index=True)

        st.subheader("📄 Extracted table")
        st.dataframe(combined)

        st.subheader("⬇️ Download Excel")

        # Build one Excel file from the combined results.
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            output_path = tmp / "invoices.xlsx"

            try:
                exporter.export_to_excel(
                    combined,
                    output_path,
                    sheet_name="Invoices",
                    total_label="Total",
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Could not build Excel: {exc}")
            else:
                st.download_button(
                    label="📥 Download invoices.xlsx",
                    data=output_path.read_bytes(),
                    file_name="invoices.xlsx",
                    mime="application/vnd.openxmlformats-officedocument."
                          "spreadsheetml.sheet",
                    type="primary",
                )

    if failed and not ok:
        st.info("No invoices could be processed. Review the errors above and try again.")

    if st.button("🧹 Clear results"):
        st.session_state.pop("results", None)
        st.rerun()
