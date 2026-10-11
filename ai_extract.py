"""ai_extract.py

Send invoices (PDF, JPG, JPEG, PNG) to an AI API using the openai package.

No provider or model names are hardcoded here: the caller passes the base
URL, model name, and API key.

Usage:
    from ai_extract import extract_invoice_to_dataframe

    df = extract_invoice_to_dataframe(file_bytes, filename, base_url, model, api_key)
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd

# File types accepted by the app.
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

# The exact columns the extracted table must have, in this order.
STANDARD_COLUMNS = [
    "Date", "Hotel", "V.NO", "Adult", "Child", "Inf",
    "EGP", "USD", "Euros", "GBP", "Total", "Net", "Notes",
]

# Same prompt for every provider; it fixes the JSON output format:
# dates as YYYY-MM-DD, plain numbers, empty strings for unknown values.
SYSTEM_PROMPT = (
    "You are an invoice extraction assistant. For every invoice you receive, "
    "reply with ONLY a JSON array of objects. No narration, no markdown, no "
    "code fences. Each object MUST have exactly these keys (in any order): "
    "Date, Hotel, V.NO, Adult, Child, Inf, EGP, USD, Euros, GBP, Total, Net, Notes. "
    "The model returns one object per invoice line/row.\n\n"
    "Rules:\n"
    "- Date: YYYY-MM-DD (e.g. 2026-09-15).\n"
    "- Numbers (Adult, Child, Inf, EGP, USD, Euros, GBP, Total, Net): plain numbers "
    "without commas, currency symbols, or spaces. Use 0 for missing amounts.\n"
    '- Unknown text values (Hotel, V.NO, Notes): empty string "".\n'
    "- If a value is missing, use an empty string for text fields and 0 for numbers.\n"
    "- Return valid JSON only."
)


def _extension(filename: str) -> str:
    """Return the lowercase file extension, e.g. '.pdf'."""
    return Path(filename).suffix.lower()


def _build_user_content(file_bytes: bytes, filename: str) -> list[dict]:
    """Build the user message content for the API call.

    Images (jpg/jpeg/png) are sent as an image_url block with a base64 data URL.
    PDFs are sent as a file block.
    """
    ext = _extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Use PDF, JPG, JPEG, or PNG."
        )

    encoded = base64.b64encode(file_bytes).decode("ascii")

    if ext == ".pdf":
        return [
            {
                "type": "file",
                "file": {
                    "filename": Path(filename).name,
                    "file_data": f"data:application/pdf;base64,{encoded}",
                },
            }
        ]

    media = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext[1:]}"
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media};base64,{encoded}"},
        }
    ]


def _parse_json(text: str) -> list[dict]:
    """Strip markdown code fences (if present) and parse the JSON payload.

    Raises ValueError with a clear message if parsing fails.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Drop the opening fence (``` or ```json) and the closing fence.
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The AI did not return valid JSON, so the table could not be built. "
            f"Raw answer was: {cleaned[:300]}"
        ) from exc

    if not isinstance(data, list):
        raise ValueError(
            "Expected a JSON array of invoice rows from the AI, "
            f"but got {type(data).__name__}."
        )

    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i + 1} of the AI answer is not an object.")
        missing = set(STANDARD_COLUMNS) - set(row)
        extra = set(row) - set(STANDARD_COLUMNS)
        if missing or extra:
            raise ValueError(
                f"Row {i + 1} of the AI answer has the wrong keys. "
                f"Missing: {sorted(missing) or 'none'}. "
                f"Unexpected: {sorted(extra) or 'none'}."
            )
    return data


def extract_invoice_to_dataframe(
    file_bytes: bytes,
    filename: str,
    base_url: str | None,
    model: str,
    api_key: str,
) -> pd.DataFrame:
    """Extract an invoice into a standardized pandas DataFrame.

    Args:
        file_bytes: Raw bytes of the uploaded PDF/image.
        filename: Original filename (chooses image block vs. file block).
        base_url: Optional API base URL; pass None for the default endpoint.
        model: Model name typed by the user.
        api_key: API key typed by the user.

    Returns:
        DataFrame with the STANDARD_COLUMNS columns, in that order.

    Raises:
        ValueError: On empty inputs, on API failures (including a clear
            "may not support PDF" message when a PDF upload is rejected),
            or when the AI answer cannot be parsed.
    """
    if not model:
        raise ValueError("Model name is empty.")
    if not api_key:
        raise ValueError("API key is empty.")

    is_pdf = _extension(filename) == ".pdf"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_content(file_bytes, filename)},
    ]

    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url or None)

    try:
        response = client.chat.completions.create(model=model, messages=messages)
    except Exception as exc:
        if is_pdf:
            raise ValueError(
                "This provider rejected the PDF file, so it may not support PDF "
                "uploads. Please upload the invoice as an image (JPG/JPEG/PNG) "
                f"instead. Details: {exc}"
            ) from exc
        raise ValueError(f"The API call failed: {exc}") from exc

    if not response.choices:
        raise ValueError("The API returned an empty response.")
    text = response.choices[0].message.content or ""
    if not text.strip():
        raise ValueError("The API returned an empty response.")

    rows = _parse_json(text)
    normalized = [
        {key: row.get(key, "") for key in STANDARD_COLUMNS} for row in rows
    ]
    return pd.DataFrame(normalized, columns=STANDARD_COLUMNS)
