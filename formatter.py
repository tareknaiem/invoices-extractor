"""formatter.py

Map extracted invoice columns to the standardized schema defined in
``suppliers_rules.yaml``.

Also performs data coercion (count columns -> integers), date normalization
(-> ``YYYY-MM-DD``), folds extra informative columns into the "Notes" column,
and can auto-detect the supplier from raw PDF text.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "suppliers_rules.yaml"

# Standard columns that hold whole-number counts.
COUNT_COLUMNS = {"Adult", "Child", "Inf"}

# Standard columns that hold monetary values.
CURRENCY_COLUMNS = {"EGP Converter", "EGP", "GBP", "Euros", "USD", "Total", "Net"}


def load_rules(path=DEFAULT_RULES_PATH):
    """Load the supplier rules YAML and return it as a dict.

    Args:
        path (str or Path): Path to the YAML rules file.

    Returns:
        dict: Parsed rules (``standard_columns`` and ``suppliers``).
    """
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def detect_supplier(text, rules=None):
    """Return the supplier key that best matches the identifiers in ``text``.

    Each supplier may declare an ``identifiers`` list (and/or a legacy singular
    ``identifier``). The supplier with the *most* matching identifiers wins
    (case-insensitive), so a shared name like "Egypt Gate" can be disambiguated
    by more specific identifiers. Ties resolve to the first declared supplier.

    Args:
        text (str): Raw PDF text to search.
        rules (dict, optional): Parsed rules; loaded from YAML when omitted.

    Returns:
        str or None: The matched supplier key, or ``None`` if no match.
    """
    if rules is None:
        rules = load_rules()

    text_lower = (text or "").lower()
    best_key = None
    best_score = 0

    for key, supplier in rules.get("suppliers", {}).items():
        identifiers = supplier.get("identifiers", [])
        if isinstance(identifiers, str):
            identifiers = [identifiers]

        # Backward compatibility with a singular ``identifier`` value.
        if supplier.get("identifier"):
            identifiers = list(identifiers) + [supplier["identifier"]]

        score = sum(
            1 for ident in identifiers if ident and str(ident).lower() in text_lower
        )
        if score > best_score:
            best_key = key
            best_score = score

    return best_key


def _coerce_int(series):
    """Coerce a count column to integers, treating empty/invalid as 0.

    Extracts the leading number so values like ``"2 (2MOTO)"`` become ``2``.
    """
    digits = series.astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(digits, errors="coerce").fillna(0).astype(int)


_DAY_PREFIX_RE = re.compile(r"^(mon|tues?|wed|thurs?|fri|sat|sun)[a-z]*\s+", re.IGNORECASE)
_MONTH_NAME_DATE_RE = re.compile(
    r"^(\d{1,2})[-/\s.]+([A-Za-z]{3,9})(?:[-/\s.]+(\d{2,4}))?$"
)


def _parse_date_value(value):
    """Parse a single date string to ``YYYY-MM-DD`` (or keep it on failure)."""
    text = str(value).strip()
    if not text:
        return ""
    text = _DAY_PREFIX_RE.sub("", text).strip()

    # Day + month name (optional year): "1-Sep", "1-Sep-2026", "1 Sep 2026".
    match = _MONTH_NAME_DATE_RE.match(text)
    if match:
        day = int(match.group(1))
        month = datetime.strptime(match.group(2)[:3], "%b").month
        year_raw = match.group(3)
        # No year in the source ("1-Sep"): assume the current year.
        year = int(year_raw) if year_raw else datetime.now().year
        if year < 100:
            year += 2000
        return datetime(year, month, day).strftime("%Y-%m-%d")

    # Numeric date: normalize separators/whitespace and parse day-first.
    cleaned = re.sub(r"[-./]", "/", text)
    cleaned = re.sub(r"\s+", "", cleaned)
    for fmt in ("%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # 2-part numeric date without a year ("09/01", "15/09"): append the year.
    two = re.match(r"^(\d{1,2})/(\d{1,2})$", cleaned)
    if two:
        first, second = int(two.group(1)), int(two.group(2))
        if first > 12 and second <= 12:
            day, month = first, second  # DD/MM
        else:
            month, day = first, second  # MM/DD (default when ambiguous)
        if 1 <= month <= 12 and 1 <= day <= 31:
            return datetime(datetime.now().year, month, day).strftime("%Y-%m-%d")

    return text  # unparseable -> keep the original string


def _normalize_dates(series):
    """Normalize date strings to ``YYYY-MM-DD`` (day-first for numeric dates)."""
    return series.map(_parse_date_value)


def _build_notes(df, notes_columns):
    """Fold several columns into a single labelled "Notes" string column."""
    n = len(df)
    if not notes_columns:
        return pd.Series([""] * n, index=range(n), dtype=object)

    present = [header for header in notes_columns if header in df.columns]
    if not present:
        return pd.Series([""] * n, index=range(n), dtype=object)

    def _row_notes(row):
        parts = []
        for header in present:
            label = notes_columns.get(header, header)
            value = row.get(header)
            text = "" if value is None else str(value).strip()
            if text:
                parts.append(f"{label}: {text}" if label else text)
        return " | ".join(parts)

    return df[present].apply(_row_notes, axis=1).reset_index(drop=True)


def format_transactions(df, supplier_key, rules=None):
    """Map a supplier's extracted columns to the standard schema.

    The returned DataFrame has exactly the ``standard_columns`` (in the order
    declared in the YAML):

    * columns are populated via the supplier's ``columns_mapping``;
    * the ``Notes`` column is assembled from ``notes_columns``;
    * count columns (Adult/Child/Inf) are coerced to integers;
    * the ``Date`` column is normalized to ``YYYY-MM-DD``;
    * standard columns with no matching source are left blank (NaN).

    Args:
        df (pd.DataFrame): Extracted transactions (e.g. from ``extractor_pdf``).
        supplier_key (str): Key of the supplier under ``suppliers``.
        rules (dict, optional): Parsed rules; loaded from YAML when omitted.

    Returns:
        pd.DataFrame: DataFrame whose columns are ``standard_columns``.
    """
    if rules is None:
        rules = load_rules()

    standard_columns = list(rules["standard_columns"])
    supplier = rules["suppliers"][supplier_key]
    mapping = supplier.get("columns_mapping", {})
    notes_columns = supplier.get("notes_columns", {})

    formatted = pd.DataFrame(index=range(len(df)))

    for standard_col in standard_columns:
        if standard_col == "Notes":
            formatted[standard_col] = _build_notes(df, notes_columns)
            continue

        source = next(
            (
                header
                for header, target in mapping.items()
                if target == standard_col and header in df.columns
            ),
            None,
        )

        if source is not None:
            formatted[standard_col] = df[source].reset_index(drop=True)
        else:
            formatted[standard_col] = np.nan

    # Data coercion: count columns -> integers.
    for col in COUNT_COLUMNS:
        if col in formatted.columns:
            formatted[col] = _coerce_int(formatted[col])

    # Date normalization.
    if "Date" in formatted.columns:
        formatted["Date"] = _normalize_dates(formatted["Date"])

    # Currency normalization: coerce to numeric (empty -> NaN) so the columns
    # keep a consistent dtype (Arrow/Excel friendly, no int/str mixing).
    for col in CURRENCY_COLUMNS:
        if col in formatted.columns:
            formatted[col] = pd.to_numeric(formatted[col], errors="coerce")

    return formatted
