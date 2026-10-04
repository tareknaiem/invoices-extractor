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
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Optional fast fuzzy-matching backend. ``rapidfuzz`` is used when available;
# otherwise we transparently fall back to the standard-library ``difflib`` so
# the dynamic mapper keeps working with zero extra dependencies.
try:  # pragma: no cover - exercised implicitly by whichever backend is present
    from rapidfuzz import fuzz as _rf_fuzz
except ImportError:  # pragma: no cover
    _rf_fuzz = None


DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "suppliers_rules.yaml"

# Standard columns that hold whole-number counts.
COUNT_COLUMNS = {"Adult", "Child", "Inf"}

# Standard columns that hold monetary values.
CURRENCY_COLUMNS = {"EGP Converter", "EGP", "GBP", "Euros", "USD", "Total", "Net"}

# ---------------------------------------------------------------------------
# Dynamic (fuzzy) column mapping
# ---------------------------------------------------------------------------
# Canonical standard-column names plus multilingual / abbreviation aliases.
# Used by the dynamic mapper so unknown-supplier headers such as ``Dt``,
# ``Date of Service`` or ``التاريخ`` all resolve to the ``Date`` column.
STANDARD_COLUMN_ALIASES = {
    "Date": [
        "date", "dt", "day", "date of service", "service date", "invoice date",
        "trip date", "التاريخ", "تاريخ", "اليوم",
    ],
    "Hotel": [
        "hotel", "hotel name", "property", "resort",
        "الفندق", "فندق", "الاوتيل", "الأوتيل",
    ],
    "V.NO": [
        "v.no", "v no", "vno", "voucher", "voucher no", "voucher number",
        "no.", "n.", "ref", "reference", "رقم", "فاوتشر", "رقم الفاوتشر",
    ],
    "Adult": [
        "adult", "adults", "ad.", "pax", "بالغ", "بالغين", "فرد", "أفراد",
    ],
    "Child": [
        "child", "children", "ch.", "chd", "طفل", "أطفال", "اطفال",
    ],
    "Inf": [
        "inf", "infant", "infants", "inf.", "رضيع", "رضع",
    ],
    "Total": [
        "total", "totals", "subtotal", "grand total", "cost", "amount", "price",
        "collect", "collection", "اجمالي", "إجمالي", "الإجمالي", "الاجمالي",
        "التكلفة", "السعر", "المجموع",
    ],
    "Net": [
        "net", "net invoice", "net amount", "صافي", "الصافي", "صافى",
    ],
    "USD": [
        "usd", "us$", "dollar", "dollars", "دولار",
    ],
    "Euros": [
        "eur", "euro", "euros", "eue", "يورو",
    ],
    "GBP": [
        "gbp", "pound", "pounds", "sterling", "£", "استرليني",
    ],
    "EGP": [
        "egp", "l.e", "le", "cash le", "rest le", "egyptian pound",
        "egyptian pounds", "جنيه", "م.ج", "ج.م", "الجنيه المصري",
    ],
    "EGP Converter": [
        "egp converter", "egp conveter", "converter", "conveter", "conversion",
        "conv",
    ],
    "Notes": [
        "notes", "note", "remarks", "remark", "comment", "comments",
        "ملاحظات", "ملاحظة", "ملحوظات",
    ],
}

# Similarity score (0..1) at or above which a fuzzy match is auto-accepted.
FUZZY_MATCH_THRESHOLD = 0.80


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


# Arabic Unicode ranges (used to add reversed-order aliases for PDF text that
# was stored visually reversed; see ``extractor_pdf._fix_arabic``).
_ARABIC_CHARS_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)


def _normalize_header(text):
    """Normalize a column header for fuzzy comparison.

    Lowercases, drops punctuation/currency symbols (which ``\\w`` excludes), and
    collapses whitespace so ``"Dt."``, ``"DT"`` and ``"dt "`` all become ``"dt"``.
    Arabic letters are preserved (``\\w`` matches them in Unicode mode).
    """
    text = str(text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _similarity(a, b):
    """Return a 0..1 similarity between two *normalized* strings.

    Uses ``rapidfuzz`` (``token_set_ratio``, which tolerates word order and extra
    tokens) when installed, otherwise a standard-library ``difflib`` heuristic.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if _rf_fuzz is not None:
        return _rf_fuzz.token_set_ratio(a, b) / 100.0

    # difflib fallback: best of full-string, substring and token-set ratios.
    ratio = SequenceMatcher(None, a, b).ratio()
    if a in b or b in a:
        ratio = max(ratio, min(len(a), len(b)) / max(len(a), len(b)))
    tokens_a, tokens_b = set(a.split()), set(b.split())
    shared = tokens_a & tokens_b
    if shared:
        ratio = max(ratio, len(shared) / max(len(tokens_a), len(tokens_b)))
    return ratio


def _alias_lookup(standard_columns):
    """Return a list of ``(standard_column, normalized_alias)`` pairs.

    Arabic aliases are added in both normal and reversed form: PDF text
    extraction frequently stores Arabic visually reversed (corrected later by
    ``extractor_pdf._fix_arabic``), so matching either orientation lets
    ``التاريخ`` resolve to ``Date`` regardless of the source encoding.
    """
    lookup = []
    for column in standard_columns:
        aliases = set(STANDARD_COLUMN_ALIASES.get(column, ()))
        aliases.add(column)  # the canonical name itself
        for alias in aliases:
            variants = {_normalize_header(alias)}
            if _ARABIC_CHARS_RE.search(str(alias)):
                variants.add(_normalize_header(str(alias)[::-1]))
            for normalized in variants:
                if normalized:
                    lookup.append((column, normalized))
    return lookup


def best_standard_match(header, standard_columns):
    """Return ``(standard_column, score)`` best matching ``header``.

    The score is ``1.0`` for an exact normalized match, otherwise the fuzzy
    similarity against the closest alias. Returns ``(None, 0.0)`` if nothing is
    close. This is the fuzzy core used by :func:`build_column_mapping`.
    """
    normalized = _normalize_header(header)
    if not normalized:
        return None, 0.0

    best_column, best_score = None, 0.0
    for column, alias in _alias_lookup(standard_columns):
        score = _similarity(normalized, alias)
        if score > best_score:
            best_column, best_score = column, score
    return best_column, best_score


_CONVERTER_RE = re.compile(r"convert|conveter|\bconv\b", re.IGNORECASE)
# Headers that name a *total/cost* column rather than a bare currency column.
_TOTAL_KEYWORD_RE = re.compile(
    r"\b(total|sub\s?total|grand|cost|net|amount|price|collect\w*|balance|due)\b",
    re.IGNORECASE,
)

# Ordered so "Egyptian Pounds" resolves to EGP before the generic GBP rule.
_CURRENCY_HEADER_PATTERNS = (
    ("USD", re.compile(r"\$|us\$|\busd\b|\bdollars?\b", re.IGNORECASE)),
    ("Euros", re.compile(r"€|\beur\b|\beuros?\b|\beue\b", re.IGNORECASE)),
    (
        "EGP",
        re.compile(
            r"\begp\b|l\.e\.?|\ble\b|egyptian pounds?|جنيه|م\.ج|ج\.م",
            re.IGNORECASE,
        ),
    ),
    ("GBP", re.compile(r"£|\bgbp\b|\bpounds?\b|sterling", re.IGNORECASE)),
)


def detect_header_currency(header):
    """Return the currency standard-column implied by a header, or ``None``.

    Detects currency symbols/codes (``$``, ``£``, ``€``, ``EGP``, ``L.E`` ...) so
    that headers like ``"$0"``, ``"REST €"`` or ``"cash L.E"`` map onto the USD /
    Euros / GBP / EGP columns. A header that also names a total/cost (``"Cost $"``)
    is left to the normal mapper, so it becomes ``Total`` rather than ``USD``.
    """
    text = str(header or "").strip()
    if not text:
        return None
    if _CONVERTER_RE.search(text):
        return None
    if _TOTAL_KEYWORD_RE.search(text):
        return None
    for column, pattern in _CURRENCY_HEADER_PATTERNS:
        if pattern.search(text):
            return column
    return None


def build_column_mapping(headers, standard_columns=None, rules=None,
                         threshold=FUZZY_MATCH_THRESHOLD):
    """Fuzzy-map unknown headers onto the standard schema.

    Runs two passes, with each standard column claimed at most once:

    1. **Currency detection** - headers carrying a currency symbol/code
       (``$``, ``£``, ``€``, ``EGP``, ``L.E``) map to USD / GBP / Euros / EGP.
    2. **Fuzzy / alias matching** - the remaining headers are matched against the
       canonical names and multilingual aliases in ``STANDARD_COLUMN_ALIASES``
       (so ``Dt`` / ``Date of Service`` / ``التاريخ`` -> ``Date``). Matches below
       ``threshold`` are ignored.

    Args:
        headers: Iterable of source column headers.
        standard_columns: Target schema; defaults to the rules' standard columns.
        rules (dict, optional): Parsed rules, used when ``standard_columns`` is
            omitted.
        threshold (float): Minimum fuzzy score to accept a match.

    Returns:
        dict: ``{source_header: standard_column}`` for the accepted matches.
    """
    if standard_columns is None:
        if rules is None:
            rules = load_rules()
        standard_columns = list(rules["standard_columns"])
    standard_columns = list(standard_columns)

    mapping = {}
    claimed = set()

    # Pass 1: explicit currency detection (highest confidence).
    for header in headers:
        if not header or header in mapping:
            continue
        currency = detect_header_currency(header)
        if currency and currency in standard_columns and currency not in claimed:
            mapping[header] = currency
            claimed.add(currency)

    # Pass 2: fuzzy / alias matching.
    for header in headers:
        if not header or header in mapping:
            continue
        column, score = best_standard_match(header, standard_columns)
        if column and score >= threshold and column not in claimed:
            mapping[header] = column
            claimed.add(column)

    return mapping


def format_with_mapping(df, standard_columns, mapping, notes_columns=None):
    """Build a standard-schema DataFrame from an explicit column mapping.

    Shared engine used by both the profile-based :func:`format_transactions` and
    the unknown-supplier fallback extractor. The returned frame has exactly
    ``standard_columns`` (in order):

    * columns are populated via ``mapping`` (``{source: standard}``);
    * the ``Notes`` column is assembled from ``notes_columns``;
    * count columns (Adult/Child/Inf) are coerced to integers;
    * the ``Date`` column is normalized to ``YYYY-MM-DD``;
    * standard columns with no matching source are left blank (NaN).

    Args:
        df (pd.DataFrame): Extracted transactions (e.g. from ``extractor_pdf``).
        standard_columns (list[str]): Ordered target schema.
        mapping (dict): ``{source_header: standard_column}``.
        notes_columns (dict, optional): ``{source_header: label}`` folded into
            the ``Notes`` standard column.

    Returns:
        pd.DataFrame: DataFrame whose columns are ``standard_columns``.
    """
    notes_columns = notes_columns or {}
    standard_columns = list(standard_columns)
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

    return format_with_mapping(
        df,
        standard_columns,
        supplier.get("columns_mapping", {}),
        supplier.get("notes_columns", {}),
    )


def format_dynamic(df, rules=None, standard_columns=None):
    """Standardize an unknown-supplier frame with the fuzzy column mapper.

    Builds a fuzzy mapping from the frame's own headers to the standard schema
    (see :func:`build_column_mapping`), folds any unmapped columns into
    ``Notes`` so no data is lost, and returns a standard-schema frame. Shared by
    the smart fallback extractor and the OCR pipeline, whose header text is
    noisy/unknown.

    Args:
        df (pd.DataFrame): Raw transactions (source/OCR header names).
        rules (dict, optional): Parsed rules; loaded from YAML when omitted.
        standard_columns (list[str], optional): Target schema override.

    Returns:
        pd.DataFrame: DataFrame whose columns are the standard schema.
    """
    if rules is None:
        rules = load_rules()
    if standard_columns is None:
        standard_columns = list(rules["standard_columns"])
    standard_columns = list(standard_columns)

    headers = list(df.columns)
    mapping = build_column_mapping(headers, standard_columns)
    notes_columns = {header: header for header in headers if header not in mapping}
    return format_with_mapping(df, standard_columns, mapping, notes_columns)
