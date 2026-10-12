"""server.py

FastAPI backend for the Flexible AI Invoice Extractor.

This module replaces the previous Streamlit UI (``app.py``) with a lightweight,
modern web architecture:

    FastAPI backend (this file)  <->  static SPA frontend (static/index.html)

It exposes a small JSON API plus one file-download endpoint, and it also serves
the static frontend so the whole app runs from a single process:

    uvicorn server:app --reload        # then open http://127.0.0.1:8000

Endpoints
---------
``GET  /``                     Serve the single-page frontend (index.html).
``GET  /api/config``           Return the saved API configuration status
                               (provider / base URL / model and whether a key is
                               stored). The API key value is NEVER returned.
``POST /api/config``           Persist the provider, base URL, model and API key
                               to the project ``.env`` (individual keys only; no
                               other env entries are touched).
``DELETE /api/config``         Remove the stored configuration keys.
``POST /api/extract``          Upload one or more invoices (PDF / JPG / JPEG /
                               PNG), run them through ``ai_extract`` and build a
                               formatted Excel workbook. Returns per-file results
                               plus a ``download_id`` for the generated file.
``GET  /api/download/{id}``    Download the generated ``invoices.xlsx``.

Design notes
------------
* Configuration is stored in ``.env`` next to this file via ``python-dotenv``'s
  ``set_key``/``unset_key`` so unrelated environment entries are preserved.
  The API key is never logged, echoed back, or included in error messages
  (see ``constitution.md`` Rule 2 - Security).
* Uploaded files are processed fully in memory and the produced workbook lives
  only in a short-lived in-memory cache; nothing is written into the project
  tree (see ``constitution.md`` Rule 2.3).
* The exported workbook includes the standard schema plus an ``EGP Converter``
  column so ``exporter`` writes the live per-row conversion formulas and the
  ``SUM`` subtotal row (see ``constitution.md`` Rule 3.2). The per-invoice
  ``Review`` QA flag (OK / MISMATCH / NO FOOTER) is surfaced in the UI instead of
  being embedded in the client-facing workbook.
* The blocking OpenAI call runs in a worker thread (``run_in_threadpool``) so it
  does not stall the event loop while the request is in flight.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import dotenv_values, set_key, unset_key
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

import ai_extract
import exporter


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ENV_PATH = BASE_DIR / ".env"

# Keys used to persist the API configuration inside the project ``.env`` file.
CFG_PROVIDER = "AI_PROVIDER"
CFG_BASE_URL = "AI_BASE_URL"
CFG_MODEL = "AI_MODEL"
CFG_API_KEY = "AI_API_KEY"

# Columns expected in the AI result (in order) plus the QA flag column.
STANDARD_COLUMNS = list(ai_extract.STANDARD_COLUMNS)          # 13 invoice columns
REVIEW_COLUMN = "Review"
TABLE_COLUMNS = STANDARD_COLUMNS + [REVIEW_COLUMN]            # shown in the UI
EXPORT_COLUMNS = STANDARD_COLUMNS + ["EGP Converter"]         # written to Excel

XLSX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Short-lived in-memory cache of generated workbooks: download_id -> bytes.
_EXCEL_CACHE: dict[str, bytes] = {}
_EXCEL_CACHE_MAX = 32


app = FastAPI(title="Flexible AI Invoice Extractor", version="2.0.0")


# ---------------------------------------------------------------------------
# Configuration helpers (.env persistence)
# ---------------------------------------------------------------------------
def _read_config() -> dict:
    """Read the stored API configuration from ``.env``.

    Returns a dict with ``provider``, ``base_url``, ``model``, ``api_key`` and a
    derived ``has_key`` boolean. The raw key is kept internal and is never sent
    to the client.
    """
    values = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
    api_key = (values.get(CFG_API_KEY) or "").strip()
    return {
        "provider": (values.get(CFG_PROVIDER) or "").strip(),
        "base_url": (values.get(CFG_BASE_URL) or "").strip(),
        "model": (values.get(CFG_MODEL) or "").strip(),
        "api_key": api_key,
        "has_key": bool(api_key),
    }


def _public_config(cfg: dict) -> dict:
    """Return the config safe to send to the browser (never the raw key)."""
    return {
        "provider": cfg["provider"],
        "base_url": cfg["base_url"],
        "model": cfg["model"],
        "has_key": cfg["has_key"],
    }


def _resolve_runtime_config(provider, base_url, model, api_key) -> dict:
    """Merge request-supplied config with the saved ``.env`` config.

    Values provided in the request take precedence; anything left blank falls
    back to what is stored on the server so the UI can "save once, then just
    upload".
    """
    saved = _read_config()
    provider = (provider or "").strip() or saved["provider"]
    base_url = (base_url or "").strip() or saved["base_url"]
    model = (model or "").strip() or saved["model"]
    api_key = (api_key or "").strip() or saved["api_key"]
    return {
        "provider": provider,
        "base_url": base_url,
        "base_url_or_none": base_url or None,
        "model": model,
        "api_key": api_key,
    }


# ---------------------------------------------------------------------------
# Workbook building
# ---------------------------------------------------------------------------
def _build_export_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine per-file frames into one export-ready DataFrame.

    Each AI frame carries the 13 standard columns plus a ``Review`` flag. The
    export keeps only the invoice columns and appends an empty ``EGP Converter``
    column; ``exporter`` then fills that column with the live conversion
    formulas and writes the ``SUM`` subtotal row.
    """
    export_frames = []
    for df in frames:
        part = df.reindex(columns=STANDARD_COLUMNS, fill_value="").copy()
        part["EGP Converter"] = pd.NA
        export_frames.append(part)

    if export_frames:
        combined = pd.concat(export_frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=EXPORT_COLUMNS)
    return combined[EXPORT_COLUMNS]


def _store_workbook(combined: pd.DataFrame) -> str:
    """Render ``combined`` to an ``.xlsx`` in memory and cache it.

    Returns a ``download_id`` that can be handed to ``/api/download/{id}``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "invoices.xlsx"
        exporter.export_to_excel(
            combined, out_path, sheet_name="Invoices", total_label="Total"
        )
        data = out_path.read_bytes()

    download_id = uuid.uuid4().hex
    _EXCEL_CACHE[download_id] = data
    # Bound the cache so it cannot grow without limit.
    while len(_EXCEL_CACHE) > _EXCEL_CACHE_MAX:
        _EXCEL_CACHE.pop(next(iter(_EXCEL_CACHE)))
    return download_id


def json_safe_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-serialisable records (NaN/NA -> None)."""
    clean = df.astype(object).where(pd.notnull(df), None)
    records = []
    for _, row in clean.iterrows():
        record = {}
        for col in df.columns:
            value = row[col]
            # numpy scalar types -> native python types.
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except (ValueError, AttributeError):
                    pass
            record[str(col)] = value
        records.append(record)
    return records



# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page frontend."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=404, detail="Frontend not built (index.html missing)."
        )
    return FileResponse(index_path)


@app.get("/api/config")
def get_config() -> dict:
    """Return the stored configuration status (without the API key value)."""
    return _public_config(_read_config())


@app.post("/api/config")
def save_config(
    provider: str = Form(...),
    api_key: str = Form(...),
    base_url: str = Form(""),
    model: str = Form(""),
) -> dict:
    """Persist the API configuration to ``.env`` (individual keys only)."""
    provider = provider.strip()
    api_key = api_key.strip()
    if not provider:
        raise HTTPException(
            status_code=400, detail="Please enter a provider name before saving."
        )
    if not api_key:
        raise HTTPException(
            status_code=400, detail="Please enter an API key before saving."
        )

    # set_key updates (or adds) a single key and leaves all other .env entries
    # untouched, so unrelated secrets are preserved.
    set_key(ENV_PATH, CFG_PROVIDER, provider)
    set_key(ENV_PATH, CFG_BASE_URL, base_url.strip())
    set_key(ENV_PATH, CFG_MODEL, model.strip())
    set_key(ENV_PATH, CFG_API_KEY, api_key)

    return {"ok": True, "key_name": CFG_API_KEY, "provider": provider}


@app.delete("/api/config")
def clear_config() -> dict:
    """Remove the stored configuration keys (only the ones this app manages)."""
    existed = ENV_PATH.exists()
    if existed:
        for key in (CFG_PROVIDER, CFG_BASE_URL, CFG_MODEL, CFG_API_KEY):
            unset_key(ENV_PATH, key)
    return {"ok": True, "existed": existed}

@app.post("/api/extract")
async def extract_invoices(
    files: list[UploadFile] = File(...),
    provider: Optional[str] = Form(None),
    base_url: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
) -> dict:
    """Extract invoices from the uploaded files and build an Excel workbook.

    Files are processed one at a time so a single failure does not abort the
    rest (matching the previous UI's per-file error handling). The generated
    workbook is cached and a ``download_id`` is returned.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    cfg = _resolve_runtime_config(provider, base_url, model, api_key)
    if not cfg["provider"]:
        raise HTTPException(status_code=400, detail="Please configure a provider name.")
    if not cfg["model"]:
        raise HTTPException(status_code=400, detail="Please configure a model name.")
    if not cfg["api_key"]:
        raise HTTPException(status_code=400, detail="Please configure an API key.")

    # Read every upload into memory up front (keeps the worker thread simple).
    payloads: list[tuple[str, bytes]] = []
    for upload in files:
        name = upload.filename or "upload"
        content = await upload.read()
        payloads.append((name, content))

    def _process() -> dict:
        results = []
        ok_frames: list[pd.DataFrame] = []
        for name, content in payloads:
            try:
                df = ai_extract.extract_invoice_to_dataframe(
                    content,
                    name,
                    base_url=cfg["base_url_or_none"],
                    model=cfg["model"],
                    api_key=cfg["api_key"],
                )
                review = str(df[REVIEW_COLUMN].iloc[0]) if not df.empty else ""
                results.append(
                    {"name": name, "ok": True, "review": review, "error": None}
                )
                ok_frames.append(df)
            except ValueError as exc:
                # Per-file failure: do not stop the others.
                results.append(
                    {"name": name, "ok": False, "review": None, "error": str(exc)}
                )
            except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
                results.append(
                    {
                        "name": name,
                        "ok": False,
                        "review": None,
                        "error": f"Unexpected error: {exc}",
                    }
                )

        ok = [r for r in results if r["ok"]]
        flagged = [
            {"name": r["name"], "review": r["review"]}
            for r in ok
            if r.get("review") in ("MISMATCH", "NO FOOTER")
        ]

        download_id = None
        columns: list[str] = []
        rows: list[dict] = []
        if ok_frames:
            combined_export = _build_export_frame(ok_frames)
            download_id = _store_workbook(combined_export)

            # Table shown in the UI keeps the Review flag per row.
            display = pd.concat(ok_frames, ignore_index=True)
            columns = [c for c in TABLE_COLUMNS if c in display.columns]
            rows = json_safe_records(display[columns])

        return {
            "total": len(results),
            "succeeded": len(ok),
            "failed": len(results) - len(ok),
            "results": results,
            "flagged": flagged,
            "columns": columns,
            "rows": rows,
            "download_id": download_id,
        }

    return await run_in_threadpool(_process)


@app.get("/api/download/{download_id}")
def download_workbook(download_id: str) -> Response:
    """Stream a previously generated workbook back to the browser."""
    data = _EXCEL_CACHE.get(download_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="Download not found or already used. Please extract again.",
        )
    return Response(
        content=data,
        media_type=XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": 'attachment; filename="invoices.xlsx"'},
    )


# ---------------------------------------------------------------------------
# Static files (mounted last so API routes take precedence)
# ---------------------------------------------------------------------------
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)


