"""extractor_pdf.py

Extract, clean, and structure tables from text-based PDF invoices.

The module extracts tables with pdfplumber, normalizes currency values,
corrects Arabic text that was stored in visual (reversed) order, and separates
the trailing summary block (Total Invoice / collections / Net) from the main
reservation rows.

It also provides the *Smart Fallback Extractor* (``smart_extract_tables`` /
``smart_fallback_extract``): a dynamic engine for unknown/new suppliers that
auto-detects the header row, separates summary rows, and fuzzy-maps the columns
onto the standard schema (via :mod:`formatter`).

When a PDF's fonts lack a ``ToUnicode`` map, text extracts as ``(cid:NN)``
noise; :func:`has_cid_garbage` detects this so the caller can route the file to
the OCR pipeline (:mod:`extractor_ocr`).

Dependencies:
    - pdfplumber
    - pandas
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pdfplumber

import formatter


# Key reported when transactions were extracted by the dynamic fallback engine.
DYNAMIC_SUPPLIER_KEY = "auto_detected"

# Arabic Unicode blocks (Arabic, Supplement, Extended-A, Presentation Forms A/B).
_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)

# Currency codes and symbols to strip from monetary values.
_CURRENCY_CODE_RE = re.compile(
    r"\b(EGP|USD|EUR|EURO|GBP|POUNDS?|L\.?E)\b", re.IGNORECASE
)
_CURRENCY_SYMBOL_RE = re.compile(r"[$€£¥]")

# Column-name keywords that indicate a monetary column.
_CURRENCY_COLUMN_KEYWORDS = (
    "usd", "egp", "eur", "eue", "euro", "gbp", "pound",
    "collect", "cost", "net", "total", "amount", "price",
    # Header spellings used by some suppliers ("Dollar", "Egy LE", "cash L.E").
    "dollar", "egy le", "cash",
)

# Labels that mark a summary/total block rather than a transaction row.
_SUMMARY_LABEL_RE = re.compile(
    r"\b(total|sub\s?total|grand\s?total|collection|net)\b", re.IGNORECASE
)

# A cell shaped like a summary label ("Rest:", "Cost:", "Total:", ...).
_SUMMARY_CELL_RE = re.compile(
    r"^(total|sub\s?total|grand\s?total|rest|net|collection|cost)\s*:",
    re.IGNORECASE,
)

# A cell starting with a summary keyword ("NET FOR BEBO =", "Total ...", ...).
_SUMMARY_ROW_RE = re.compile(
    r"^(total|sub\s?total|grand\s?total|net)\b", re.IGNORECASE
)

# Summary keywords used to identify multi-column summary tables.
_SUMMARY_WORDS = {
    "total", "totals", "subtotal", "sub total", "grand total", "net",
    "collection", "collected", "cost", "costs", "rest", "price", "amount",
    # Arabic summary labels (also used by the dynamic fallback extractor).
    "اجمالي", "إجمالي", "الإجمالي", "الاجمالي", "صافي", "الصافي",
    "المجموع", "مجموع", "الباقي",
}

# Labels that head the invoice-metadata strip above the main table
# ("INVOICE NO", "INVOICE DATE", "invoice number", ...). "Total Invoice" (a
# data-table column) must NOT match: the ``no/num/number/date`` suffix is
# mandatory, so bare "invoice" never triggers this.
_INVOICE_LABEL_RE = re.compile(
    r"\binvoice\s*(?:no|num|number|date)\b\.?", re.IGNORECASE
)

# A bare identifier 10+ digits long (invoice/booking numbers like 2609131157)
# that must never be coerced into a money value.
_BARE_ID_RE = re.compile(r"\d{10,}")

# The metadata strip is a small table (TOTAL / RECORDS / PERIOD / RECIPIENT /
# INVOICE DATE / INVOICE NO = 6 columns). A table wider than this cannot be the
# metadata strip, even if its first row carries an invoice label.
INVOICE_META_MAX_COLS = 10

# Unmapped-glyph markers emitted when a PDF font lacks a ``ToUnicode`` map,
# e.g. ``(cid:123)``. They carry no readable text, so they are stripped and used
# as a signal that the page should be OCR'd instead.
_CID_RE = re.compile(r"\(cid:\d+\)")


def has_cid_garbage(text):
    """Return True when ``text`` contains unmapped glyphs like ``(cid:123)``.

    PDFs whose fonts lack a ``ToUnicode`` CMap extract as ``(cid:NN)`` noise
    (common with Arabic and scanned documents). Such files are best routed to
    the OCR pipeline (:mod:`extractor_ocr`).
    """
    return bool(_CID_RE.search(text or ""))


def _strip_cid(text):
    """Remove ``(cid:NN)`` glyph markers and tidy the surrounding whitespace."""
    return _CID_RE.sub(" ", text)


def _clean_cell(value):
    """Normalize a single table cell.

    - Converts ``None`` to an empty string.
    - Converts any other value to ``str``.
    - Removes unmapped ``(cid:NN)`` glyph markers.
    - Removes newline characters and collapses all runs of whitespace into a
      single space, so the final DataFrame is clean and spreadsheet-friendly.
    """
    if value is None:
        return ""
    # ``str.split()`` with no arguments splits on any whitespace (including
    # newlines, tabs, and repeated spaces); joining with a single space both
    # removes newlines and normalizes the remaining whitespace.
    return " ".join(_strip_cid(str(value)).split())


def _clean_currency(value):
    """Normalize a currency-formatted value into a plain number.

    Examples:
        "$ 7 5"        -> 75
        "63 €"         -> 63
        "EGP 4,600"    -> 4600
        "€ 310"        -> 310
        "( 23,756.00)" -> -23756.0
        "915-"         -> -915          (trailing-minus negative)
        "2609131157"   -> "2609131157"  (bare 10+ digit identifier -> text)

    Bare identifiers 10+ digits long (invoice/booking numbers) are returned
    unchanged as text so they can never surface as a huge amount (e.g.
    ``2,609,131,157.00``) inside a currency column. Non-numeric text is
    returned unchanged (whitespace-normalized).
    """
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text in ("-", "--", "—"):
        return ""

    original = text
    had_currency_context = bool(
        _CURRENCY_CODE_RE.search(original) or _CURRENCY_SYMBOL_RE.search(original)
    )

    # Accounting-style parentheses denote a negative amount.
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]

    # Strip currency codes and symbols, then thousands separators.
    text = _CURRENCY_CODE_RE.sub(" ", text)
    text = _CURRENCY_SYMBOL_RE.sub(" ", text)
    text = text.replace(",", "")

    # If what remains is a plain number (internal spaces allowed, e.g. "$ 7 5"),
    # collapse the spaces and return a numeric value.
    candidate = re.sub(r"\s+", "", text)

    # Trailing minus ("915-") is another negative convention.
    trailing_minus = candidate.endswith("-") and len(candidate) > 1
    if trailing_minus:
        candidate = candidate[:-1]

    if re.fullmatch(r"-?\d+(\.\d+)?", candidate):
        number = float(candidate) if "." in candidate else int(candidate)
        if (
            not had_currency_context
            and isinstance(number, int)
            and abs(number) >= 10**9
        ):
            # A bare invoice/booking number masquerading as an amount.
            return " ".join(original.split())
        if trailing_minus:
            number = -abs(number)
        if negative:
            number = -abs(number)
        return number

    return " ".join(text.split())


def _fix_arabic(value):
    """Correct Arabic text that was extracted in visual (reversed) order.

    PDF generators often store Arabic glyphs left-to-right, so extraction
    yields strings like ``'ىدايعلا رون'``. Reversing recovers the intended
    reading (``'نور العيادى'``). Only predominantly-Arabic strings are reversed
    so mixed content (e.g. ``"1 ,300 . م.ج"``) is not scrambled.
    """
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    if not _ARABIC_RE.search(text):
        return text
    if len(_ARABIC_RE.findall(text)) / len(text) < 0.5:
        return text
    return text[::-1]


def _is_currency_column(name):
    """Return True when a column name looks like a monetary column."""
    lowered = str(name).lower()
    # A currency symbol in the header (e.g. "REST $") marks a monetary column.
    if _CURRENCY_SYMBOL_RE.search(name):
        return True
    return any(keyword in lowered for keyword in _CURRENCY_COLUMN_KEYWORDS)


def _dedupe_columns(columns):
    """Make a list of column names unique.

    Blank names are replaced with ``Unnamed_<index>``; any remaining
    duplicates get a numeric suffix. This keeps ``pd.concat`` (which requires
    uniquely-labelled columns) from raising ``InvalidIndexError``.
    """
    seen = {}
    result = []
    for idx, col in enumerate(columns):
        name = col if col else f"Unnamed_{idx}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        result.append(name)
    return result


def detect_header_row(table):
    """Return the index of the table's header row (text-density heuristic).

    The header is the *fullest* row that is at least half non-empty and mostly
    word-like. Requiring alphabetic content keeps numeric rate/config rows from
    being mistaken for the column header. Falls back to the single densest row
    when no row satisfies the word-like rule.
    """
    if not table:
        return 0

    cleaned = [[_clean_cell(cell) for cell in row] for row in table]
    n_cols = max((len(row) for row in cleaned), default=0)

    def _non_empty(row):
        return sum(1 for cell in row if cell)

    def _wordy_count(row):
        return sum(1 for cell in row if cell and any(c.isalpha() for c in str(cell)))

    threshold = max(2, n_cols // 2)
    candidates = [
        i
        for i, row in enumerate(cleaned)
        if _non_empty(row) >= threshold
        and _wordy_count(row) >= _non_empty(row) // 2
    ]
    if candidates:
        return max(candidates, key=lambda i: _non_empty(cleaned[i]))

    # Fallback: the row with the most non-empty cells.
    return max(range(len(cleaned)), key=lambda i: _non_empty(cleaned[i]))


def _table_to_frame(table, fix_arabic=True):
    """Convert a single pdfplumber table (list of rows) into a DataFrame.

    The header row is auto-detected from text density (see
    :func:`detect_header_row`). Arabic header text is corrected (reversed
    storage), and a second, sparse header row (e.g. currency sub-labels) is
    merged in when it fills empty header cells.

    Args:
        table (list): Raw table (list of rows).
        fix_arabic (bool): Reverse visually-reversed Arabic (text-PDF mode).
            Disable for OCR output, which already returns logical-order text.
    """
    if not table:
        return None

    cleaned = [[_clean_cell(cell) for cell in row] for row in table]
    n_cols = len(cleaned[0]) if cleaned else 0

    def _non_empty(row):
        return sum(1 for cell in row if cell)

    # Header row is auto-detected from text density (see ``detect_header_row``).
    header_idx = detect_header_row(table)
    header = [
        (_fix_arabic(cell) if fix_arabic else cell) for cell in cleaned[header_idx]
    ]
    data_start = header_idx + 1

    # Merge a second, sparse header row that fills empty/stacked labels
    # (e.g. "cash" + "L.E" -> "cash L.E", plus "euro" and "dollar").
    if header_idx + 1 < len(cleaned):
        next_row = cleaned[header_idx + 1]
        header_non_empty = _non_empty(cleaned[header_idx])
        next_non_empty = _non_empty(next_row)
        fills_gap = any(
            next_row[i] and not cleaned[header_idx][i] for i in range(n_cols)
        )
        if 0 < next_non_empty < header_non_empty and fills_gap:
            for i, raw in enumerate(next_row):
                value = _fix_arabic(raw) if fix_arabic else raw
                if not value:
                    continue
                header[i] = f"{header[i]} {value}".strip() if header[i] else value
            data_start = header_idx + 2

    header = _dedupe_columns(header)
    data = cleaned[data_start:]

    if not data:
        return None

    return pd.DataFrame(data, columns=header)


def _is_summary_table(table):
    """Detect a key/value or multi-column summary block.

    Recognises a 2-column block whose first label looks like a summary metric
    (``Total Invoice``, ``USD Collection``), or a multi-column block whose first
    column is mostly summary keywords (``Total``, ``COLLECTED``, ``COST``,
    ``Rest``, ``Net`` ...).
    """
    if not table or not table[0]:
        return False

    # 2-column key/value summary ("Total Invoice", "USD Collection", ...).
    if len(table[0]) == 2:
        first_label = _clean_cell(table[0][0])
        if _SUMMARY_LABEL_RE.search(first_label):
            return True

    # Multi-column summary: first column cells are summary keywords.
    first_col = [_clean_cell(row[0]) if row else "" for row in table]
    labels = [v.lower() for v in first_col if v]
    if labels:
        hits = sum(1 for v in labels if v in _SUMMARY_WORDS)
        if hits >= max(1, len(labels) // 2):
            return True

    return False


def _parse_summary_table(table):
    """Convert a key/value summary table into a ``{label: value}`` dict."""
    summary = {}
    for row in table:
        if not row:
            continue
        label = _clean_cell(row[0])
        if not label:
            continue
        summary[label] = _clean_currency(row[1])
    return summary


def _invoice_label_row(row):
    """Return True when ``row`` heads the invoice-metadata label strip.

    The strip labels the page (TOTAL / RECORDS / PERIOD / RECIPIENT /
    INVOICE DATE / INVOICE NO). At least two cells must look like an
    ``INVOICE ...`` label so a single data column named "Invoice No" never
    triggers this.
    """
    if not row:
        return False
    matches = sum(
        1 for cell in row if _INVOICE_LABEL_RE.search(_clean_cell(cell))
    )
    return matches >= 2


def _split_invoice_meta(table, summary):
    """Pull the invoice-metadata strip out of ``table`` into ``summary``.

    The strip is the label row plus the value row directly beneath it. Its
    values — most importantly the invoice number — are page metadata and must
    never leak into the transaction table (where the invoice number would
    otherwise surface as a huge amount in a currency column). Returns the
    table with both rows removed.
    """
    if not table:
        return table

    label_indices = set(i for i, row in enumerate(table) if _invoice_label_row(row))
    if not label_indices:
        return table

    kept = []
    for i, row in enumerate(table):
        if i in label_indices:
            value_row = table[i + 1] if i + 1 < len(table) else []
            for j, cell in enumerate(row):
                label = _clean_cell(cell)
                if not label:
                    continue
                value = _clean_cell(value_row[j]) if j < len(value_row) else ""
                if value:
                    summary.setdefault(label, _clean_currency(value))
            continue
        if i - 1 in label_indices:
            # The value row directly beneath a label row is metadata too.
            continue
        kept.append(row)
    return kept


def _drop_summary_rows(frame, min_filled=5, arabic_labels_are_summary=True):
    """Remove repeated-header, total/subtotal and summary rows from a frame.

    A row is treated as summary when:
    * its first cell is a total keyword (``Total``, ``Subtotal``, ...);
    * any cell is a ``Label:`` summary cell (``Rest:``, ``Cost:``, ...);
    * any cell starts with a summary keyword (``NET FOR BEBO =``, ...);
    * its first cell is a summary word (``price``, ``t.cost``, ``COLLECTED``,
      ``COST``, ``Rest``, ...) or, when ``arabic_labels_are_summary``, an Arabic
      label;
    * its first cell repeats the header (a second page's header row); or
    * the row is sparse (fewer than ``min_filled`` cells filled).

    Args:
        frame (pd.DataFrame): The table to filter.
        min_filled (int): Rows with fewer filled cells are treated as summary.
            The dynamic fallback lowers this for narrow tables.
        arabic_labels_are_summary (bool): Treat any Arabic first-cell label as a
            summary row. The fallback disables this so Arabic *data* is retained.
    """
    if frame.empty:
        return frame

    total_labels = {"total", "totals", "grand total", "subtotal", "sub total"}
    first_col = frame.columns[0]

    first_is_total = frame[first_col].map(
        lambda v: str(v).strip().lower() in total_labels
    )

    # A repeated header row (e.g. a second page) matches the column header.
    first_is_header = frame[first_col].map(
        lambda v: str(v).strip().lower() == str(first_col).strip().lower()
    )

    def _row_is_summary(row):
        non_empty = 0
        for value in row:
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            non_empty += 1
            if _SUMMARY_CELL_RE.search(text) or _SUMMARY_ROW_RE.search(text):
                return True
            # A row carrying an "INVOICE NO / DATE" label is metadata, not data.
            if _INVOICE_LABEL_RE.search(text):
                return True
        first_value = row[first_col]
        first_text = str(first_value).strip() if first_value is not None else ""
        first_lower = first_text.lower()
        if (
            first_lower in _SUMMARY_WORDS
            or first_lower.startswith("t.cost")
            or first_lower.startswith("t cost")
            or (arabic_labels_are_summary and _ARABIC_RE.search(first_text))
        ):
            return True
        # Sparse rows (very few filled cells) are placeholders/summaries.
        return non_empty < min_filled

    row_is_summary = frame.apply(_row_is_summary, axis=1)

    return frame[~(first_is_total | first_is_header | row_is_summary)].reset_index(drop=True)


def _clean_dataframe(df, fix_arabic=True):
    """Apply (optional) Arabic correction and currency cleaning to a frame."""
    if df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        if fix_arabic:
            df[col] = df[col].map(_fix_arabic)
        if _is_currency_column(col):
            df[col] = df[col].map(_clean_currency)
    return df


def get_pdf_text(pdf_path):
    """Return the concatenated text of every page in a PDF.

    Args:
        pdf_path (str or pathlib.Path): Path to the PDF file.

    Returns:
        str: All page text joined by newlines, with ``(cid:NN)``
        unmapped-glyph markers stripped. Use :func:`get_raw_pdf_text` when the
        markers themselves must be preserved (e.g. OCR auto-routing).
    """
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(_strip_cid(page.extract_text() or ""))
    return "\n".join(parts)


def get_raw_pdf_text(pdf_path):
    """Return the concatenated text of every page, preserving markers.

    Unlike :func:`get_pdf_text`, unmapped-glyph ``(cid:NN)`` markers are kept
    so :func:`extractor_pdf.has_cid_garbage` and the OCR auto-routing heuristic
    can detect font-encoding noise.

    Args:
        pdf_path (str or pathlib.Path): Path to the PDF file.

    Returns:
        str: All raw page text joined by newlines.
    """
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def count_pdf_pages(pdf_path):
    """Return the number of pages in ``pdf_path``."""
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def extract_tables_from_pdf(pdf_path):
    """Read a text-based PDF and return ``(transactions, summary)``.

    The function extracts every table, separates the trailing summary block
    (Total Invoice, collections, Net Invoice) from the reservation rows, and
    applies Arabic correction plus currency normalization.

    Args:
        pdf_path (str or pathlib.Path): Path to the PDF file.

    Returns:
        tuple[pd.DataFrame, dict]:
            * ``transactions`` - a cleaned DataFrame of reservation rows with
              Arabic corrected and currency values normalized.
            * ``summary`` - a dict of summary metrics (e.g. ``Total Invoice``,
              ``USD Collection``, ``Net Invoice``).
    """
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables.extend([t for t in (page.extract_tables() or []) if t])

    main_frames = []
    summary = {}

    for table in tables:
        # The invoice-metadata strip (INVOICE NO / INVOICE DATE / ...) is page
        # metadata, not transaction data: split it out first so the invoice
        # number can never become a table value.
        table = _split_invoice_meta(table, summary)
        if not table:
            continue

        if _is_summary_table(table):
            summary.update(_parse_summary_table(table))
            continue

        frame = _table_to_frame(table)
        if frame is None or frame.empty:
            continue

        frame = _drop_summary_rows(frame)
        if frame.empty:
            continue

        main_frames.append(frame)

    transactions = (
        pd.concat(main_frames, ignore_index=True) if main_frames else pd.DataFrame()
    )
    transactions = _clean_dataframe(transactions)

    return transactions, summary


def tables_to_transactions(tables, fix_arabic=True):
    """Rebuild *raw* (unmapped) transactions + summary from a list of tables.

    Shared by the smart fallback and the OCR pipeline: it applies the same
    header detection and summary separation, but leaves column naming/mapping to
    the caller.

    Args:
        tables (list): Raw tables (each a list of rows of cell strings).
        fix_arabic (bool): Reverse visually-reversed Arabic (text-PDF mode).
            Pass ``False`` for OCR output, which is already logical order.

    Returns:
        tuple[pd.DataFrame, dict]: (raw transactions, summary metrics). The
        transactions frame keeps the source/OCR header names as its columns.
    """
    main_frames = []
    summary = {}

    for table in tables:
        # The invoice-metadata strip (INVOICE NO / INVOICE DATE / ...) is page
        # metadata, not transaction data: split it out first so the invoice
        # number can never become a table value.
        table = _split_invoice_meta(table, summary)
        if not table:
            continue

        if _is_summary_table(table):
            summary.update(_parse_summary_table(table))
            continue

        frame = _table_to_frame(table, fix_arabic=fix_arabic)
        if frame is None or frame.empty:
            continue

        # Sparsity threshold for placeholder rows. This scales DOWN (never up)
        # as tables get narrower, matching the fallback's "narrow-table
        # friendly" intent: a 26-column landscape invoice whose rows only fill
        # ~8 cells must not be held to 13 filled cells (which would discard
        # every row and silently yield an empty workbook). Capping at the
        # _drop_summary_rows default of 5 also keeps this path consistent with
        # extract_tables_from_pdf, which uses the default threshold.
        min_filled = min(5, max(2, len(frame.columns) // 2))
        frame = _drop_summary_rows(
            frame, min_filled=min_filled, arabic_labels_are_summary=False
        )
        if frame.empty:
            continue

        main_frames.append(frame)

    if not main_frames:
        return pd.DataFrame(), summary

    raw = pd.concat(main_frames, ignore_index=True)
    return _clean_dataframe(raw, fix_arabic=fix_arabic), summary


def smart_extract_tables(tables, rules=None, fix_arabic=True):
    """Smart fallback: standardize tables from an unknown supplier.

    Used when :func:`formatter.detect_supplier` cannot match a profile. Reuses
    the same table cleaning and summary separation as
    :func:`extract_tables_from_pdf`, but replaces the fixed profile mapping with
    automatic detection:

    * the header row is auto-detected from text density (``_table_to_frame``);
    * summary/total rows are separated with ``_drop_summary_rows`` (narrow-table
      friendly settings);
    * column headers are fuzzy-mapped onto the standard schema with
      :func:`formatter.build_column_mapping` (including currency-symbol
      detection), so ``Dt`` / ``Date of Service`` / ``التاريخ`` all become
      ``Date`` and ``$`` / ``£`` / ``€`` / ``L.E`` headers become currency
      columns;
    * any leftover columns are folded into ``Notes`` so no data is lost.

    Args:
        tables (list): Raw pdfplumber tables (each a list of rows).
        rules (dict, optional): Parsed rules; loaded from YAML when omitted.
        fix_arabic (bool): Reverse visually-reversed Arabic (text-PDF mode).
            Pass ``False`` for OCR output, which is already logical order.

    Returns:
        tuple[pd.DataFrame, dict]: A standard-schema DataFrame (every
        ``standard_columns`` entry present, ready for :mod:`exporter`) and the
        summary metrics dict.
    """
    if rules is None:
        rules = formatter.load_rules()
    standard_columns = list(rules["standard_columns"])

    raw, summary = tables_to_transactions(tables, fix_arabic=fix_arabic)
    if raw.empty:
        return pd.DataFrame(columns=standard_columns), summary

    return formatter.format_dynamic(raw, standard_columns=standard_columns), summary


def smart_fallback_extract(pdf_path, rules=None):
    """Extract an unknown-supplier PDF with the smart fallback engine.

    A thin wrapper around :func:`smart_extract_tables` that first pulls every
    table from ``pdf_path`` with pdfplumber.

    Args:
        pdf_path (str or pathlib.Path): Path to the PDF file.
        rules (dict, optional): Parsed rules; loaded from YAML when omitted.

    Returns:
        tuple[pd.DataFrame, dict]: Standard-schema transactions + summary dict.
    """
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables.extend([t for t in (page.extract_tables() or []) if t])
    return smart_extract_tables(tables, rules)


if __name__ == "__main__":
    import sys

    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "egypt gate (1).pdf"

    pdf_path = Path(pdf_file)
    if not pdf_path.is_absolute():
        pdf_path = Path(__file__).resolve().parent / pdf_path

    if not pdf_path.exists():
        # Fallback: use the first PDF found in the workspace directory.
        candidates = sorted(Path(__file__).resolve().parent.glob("*.pdf"))
        if candidates:
            print(
                f"Note: '{pdf_file}' was not found; "
                f"using '{candidates[0].name}' instead.\n"
            )
            pdf_path = candidates[0]
        else:
            raise SystemExit(
                f"PDF file not found: {pdf_path}\n"
                f"Please place '{pdf_file}' in the workspace directory and run again."
            )

    transactions, summary = extract_tables_from_pdf(pdf_path)

    print(f"Extracted {len(transactions)} transaction rows from '{pdf_path.name}'.")
    print("First 10 transaction rows:")
    print(transactions.head(10).to_string(index=False))

    print("\nSummary metrics:")
    if summary:
        for label, value in summary.items():
            print(f"  {label}: {value}")
    else:
        print("  (none found)")
