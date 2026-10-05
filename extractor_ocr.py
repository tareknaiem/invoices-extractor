"""extractor_ocr.py

OCR extraction pipeline for scanned / image-based invoice PDFs.

Pages are rendered to images with pdfplumber's built-in renderer (default
``pypdfium2`` backend, so no external poppler binary is required), recognised
with PaddleOCR configured for Arabic (``lang='ar'``), and the recognised text
boxes are clustered back into a table grid. That grid is handed to the shared
dynamic pipeline (:func:`extractor_pdf.tables_to_transactions` +
:func:`formatter.format_dynamic`), so the result is a standard-schema DataFrame
that flows straight into :mod:`exporter` (Subtotal row + live EGP formulas
included).

The heavy PaddleOCR / PaddlePaddle stack is imported lazily, so this module is
cheap to import and the text-based pipeline keeps working when OCR is not
installed. :func:`is_ocr_available` reports whether the OCR backend is usable.

Dependencies (optional):
    - paddleocr + paddlepaddle (CPU build)  -> the OCR backend
    - pdfplumber                             -> page rendering (already used)

PaddleOCR 3.x is distributed as ``paddleocr`` on top of ``paddlex``; CPU
inference additionally needs the ``paddlepaddle`` framework (Windows wheels are
available for CPython 3.9-3.13).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
import pdfplumber

import extractor_pdf
import formatter


# Rendering / recognition defaults.
DEFAULT_RESOLUTION = 300
OCR_LANG = "ar"

# Direct image files (jpg/jpeg/png) accepted alongside PDFs. They have no text
# layer, so they always route through the OCR pipeline (see ``is_image_path``).
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Row grouping: two boxes belong to the same row when their vertical centres are
# within ``ROW_Y_TOLERANCE`` * the median text-line height.
ROW_Y_TOLERANCE = 0.6
# Minimum blank vertical band (as a fraction of page width) that separates two
# table columns.
MIN_GUTTER_RATIO = 0.012
# Below this many characters per page, a PDF is treated as scanned.
MIN_CHARS_PER_PAGE = 40


class OcrUnavailableError(RuntimeError):
    """Raised when the PaddleOCR backend cannot be imported or initialised."""


# Exact import failures recorded by ocr_dependency_status():
# {package: "ExcType: message"} — e.g. ImportError: libGL.so.1: cannot open
# shared object file. Kept for diagnostics so deploy logs show WHY the OCR
# backend is unavailable instead of a generic "not installed".
_OCR_IMPORT_ERRORS = {}


def _missing_ocr_message():
    """Return the actionable installation message shown when OCR is unusable.

    Includes the exact import exception for every package that failed to
    import (see :func:`ocr_import_errors`) so the root cause (e.g. a missing
    system library) is visible in Streamlit Cloud logs.
    """
    ready, missing = ocr_dependency_status()
    missing_text = ", ".join(missing) if missing and not ready else "paddleocr/paddlepaddle"
    message = (
        f"PaddleOCR is not available (missing: {missing_text}). Install the "
        "OCR extras and try again: paddleocr==3.7.0 (pulls paddlex>=3.7.0) "
        "plus the CPU runtime paddlepaddle==3.3.1 (Windows wheels for Python "
        "3.9-3.13). Text-based PDFs continue to work without OCR."
    )
    details = "; ".join(
        f"{name}: {err}" for name, err in sorted(_OCR_IMPORT_ERRORS.items())
    )
    if details:
        message += f" Import errors: {details}"
    return message


def require_ocr_engine(engine=None, lang=OCR_LANG):
    """Return a ready OCR engine, raising a clear error when unavailable.

    Args:
        engine (object, optional): Caller-supplied engine with
            ``recognize(image)``. Useful for tests and dependency injection.
        lang (str): Language passed to the default Arabic PaddleOCR engine.
    """
    if engine is not None:
        required = ("recognize",)
        missing = [name for name in required if not hasattr(engine, name)]
        if missing:
            raise OcrUnavailableError(
                f"The supplied OCR engine lacks: {', '.join(missing)}."
            )
        return engine
    if not is_ocr_available():
        raise OcrUnavailableError(_missing_ocr_message())
    return PaddleOcrEngine(lang=lang)


@dataclass
class OcrLine:
    """A single recognised text box.

    Attributes:
        text (str): Recognised text.
        box (tuple): Axis-aligned bounding box ``(x0, y0, x1, y1)`` in pixels.
        confidence (float): Recognition confidence (0..1).
    """

    text: str
    box: tuple
    confidence: float = 1.0


def _pil_to_array(image):
    """Convert a PIL image to a contiguous ``(H, W, 3)`` uint8 numpy array."""
    return np.asarray(image.convert("RGB"))


def render_pdf_pages(pdf_path, resolution=DEFAULT_RESOLUTION, max_pages=None):
    """Render PDF pages to numpy RGB arrays using pdfplumber's image backend.

    Args:
        pdf_path (str or pathlib.Path): Path to the PDF.
        resolution (int): Rendering DPI (higher = better OCR, slower).
        max_pages (int, optional): Limit the number of pages rendered.

    Returns:
        list[numpy.ndarray]: One ``(H, W, 3)`` array per rendered page.
    """
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
        for page in pages:
            rendered = page.to_image(resolution=resolution)
            images.append(_pil_to_array(rendered.original))
    return images


def is_ocr_available():
    """Return ``True`` when the PaddleOCR backend (and PaddlePaddle) imports."""
    ready, _ = ocr_dependency_status()
    return ready


def ocr_dependency_status():
    """Return ``(ready, missing)`` for the OCR stack.

    ``ready`` is True only when both ``paddleocr`` and the ``paddle``
    inference runtime (``paddlepaddle``) import. ``missing`` names the packages
    that could not be imported. The exact exception raised by each failed
    import is recorded in ``_OCR_IMPORT_ERRORS`` and exposed by
    :func:`ocr_import_errors` for diagnostics.
    """
    missing = []
    for module in ("paddleocr", "paddle"):
        package = "paddleocr" if module == "paddleocr" else "paddlepaddle"
        try:
            __import__(module)
        except Exception as exc:  # noqa: BLE001 - keep probing the remaining packages
            missing.append(package)
            _OCR_IMPORT_ERRORS[package] = f"{type(exc).__name__}: {exc}"
        else:
            _OCR_IMPORT_ERRORS.pop(package, None)
    return (not missing, tuple(missing))


def ocr_import_errors():
    """Return ``{package: "ExcType: message"}`` for failed OCR imports.

    Re-probes the imports so the recorded details always reflect the current
    environment — e.g. ``ImportError: libGL.so.1: cannot open shared object
    file`` when OpenCV's system libraries are missing on a headless box, or
    ``ModuleNotFoundError`` when the wheels were never installed. Returns an
    empty dict when the whole OCR stack imports.
    """
    ocr_dependency_status()
    return dict(_OCR_IMPORT_ERRORS)


class PaddleOcrEngine:
    """Lazy wrapper around PaddleOCR configured for Arabic (``lang='ar'``).

    Targets the PaddleOCR 3.x pipeline API (``PaddleOCR(lang=..., device="cpu",
    use_textline_orientation=...)`` + ``predict(input=<numpy array>)``) and
    degrades gracefully to the legacy 2.x spellings (``use_angle_cls`` +
    ``ocr(...)``). Instantiation is deferred until the first
    :meth:`recognize` call so the (slow, model-loading) backend is only built
    when OCR is actually used.
    """

    def __init__(self, lang=OCR_LANG, device="cpu", use_textline_orientation=True):
        self.lang = lang
        self.device = device
        self.use_textline_orientation = use_textline_orientation
        # Legacy PaddleOCR 2.x spelling of the orientation flag (fallback).
        self.use_angle_cls = use_textline_orientation
        self._ocr = None

    @staticmethod
    def _constructor_candidates(lang, device, orient):
        """Candidate ``PaddleOCR(...)`` kwargs, newest API first.

        The PaddleOCR 3.x (paddlex pipeline) constructor takes ``lang``,
        ``device`` (``"cpu"`` for CPU-only deployment) and
        ``use_textline_orientation``; the legacy 2.x spellings are kept as
        fallbacks so older installs still initialise.
        """
        return [
            {
                "lang": lang,
                "device": device,
                "use_textline_orientation": orient,
                "show_log": False,
            },
            {"lang": lang, "device": device, "use_textline_orientation": orient},
            {"lang": lang, "device": device},
            {"lang": lang, "use_angle_cls": orient},
            {"lang": lang},
        ]

    def _ensure_engine(self):
        if self._ocr is not None:
            return
        if not is_ocr_available():
            raise OcrUnavailableError(_missing_ocr_message())
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # noqa: BLE001
            raise OcrUnavailableError(
                "PaddleOCR is not installed. Install the OCR extras "
                "(paddleocr + paddlepaddle) to use the scanned-invoice mode. "
                f"Original error: {exc}"
            ) from exc

        # A test or user seam may patch availability without supplying the real
        # PaddleOCR package. In that case fail loudly before attempting to use
        # a lightweight stand-in as if it were the trained backend.
        if PaddleOCR is None:
            raise OcrUnavailableError(_missing_ocr_message())

        # The constructor signature changed across PaddleOCR releases; try the
        # richer 3.x form first and degrade gracefully to the 2.x spellings.
        for kwargs in self._constructor_candidates(
            self.lang, self.device, self.use_textline_orientation
        ):
            try:
                self._ocr = PaddleOCR(**kwargs)
                return
            except TypeError:
                continue
            except Exception as exc:  # noqa: BLE001
                raise OcrUnavailableError(
                    f"Could not initialise PaddleOCR (lang={self.lang!r}): {exc}"
                ) from exc
        raise OcrUnavailableError("Could not initialise PaddleOCR.")

    @staticmethod
    def _predict(ocr, image):
        """Call the newest supported prediction entry point on the backend.

        PaddleOCR 3.x accepts the image as keyword input via
        ``predict(input=<numpy array>)``; the legacy 2.x entry point is
        ``ocr(<image>)``. Both are tried positionally and by keyword so the
        pipeline works whichever spelling the installed backend supports.
        """
        errors = []
        predict = getattr(ocr, "predict", None)
        if callable(predict):
            for call in (lambda: predict(input=image), lambda: predict(image)):
                try:
                    return call()
                except TypeError as exc:
                    errors.append(exc)
        legacy = getattr(ocr, "ocr", None)
        if callable(legacy):
            for call in (lambda: legacy(image), lambda: legacy(input=image)):
                try:
                    return call()
                except TypeError as exc:
                    errors.append(exc)
        raise OcrUnavailableError(
            "The installed PaddleOCR backend exposes neither a usable "
            f"predict() nor ocr() entry point ({errors!r})."
        )

    def recognize(self, image):
        """Run OCR on ``image`` and return a list of :class:`OcrLine`."""
        self._ensure_engine()
        return _parse_paddle_result(self._predict(self._ocr, image))


# ---------------------------------------------------------------------------
# PaddleOCR output normalisation
# ---------------------------------------------------------------------------
def _field(obj, name):
    """Read ``name`` from a dict-like object or an attribute, or ``None``."""
    value = getattr(obj, name, None)
    if value is None and hasattr(obj, "get"):
        try:
            value = obj.get(name)
        except Exception:  # noqa: BLE001 - exotic result objects
            value = None
    return value


def _box_to_xyxy(box):
    """Collapse a 4-point polygon into an axis-aligned ``(x0, y0, x1, y1)``."""
    points = [tuple(map(float, p)) for p in box]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _parse_v2_page(page):
    """Parse a classic PaddleOCR 2.x page: ``[[box, (text, score)], ...]``."""
    lines = []
    for entry in page or []:
        if not entry or len(entry) < 2:
            continue
        box, payload = entry[0], entry[1]
        if isinstance(payload, (list, tuple)) and payload:
            text, score = payload[0], (payload[1] if len(payload) > 1 else 1.0)
        else:
            text, score = payload, 1.0
        text = str(text).strip()
        if text:
            lines.append(OcrLine(text, _box_to_xyxy(box), float(score)))
    return lines


def _parse_v3_page(page):
    """Parse a PaddleOCR 3.x result dict: ``rec_texts``/``rec_scores``/``dt_polys``."""
    texts = _field(page, "rec_texts") or []
    scores = _field(page, "rec_scores") or []
    polys = _field(page, "dt_polys") or _field(page, "rec_polys") or []

    lines = []
    for i, text in enumerate(texts):
        text = str(text).strip()
        if not text:
            continue
        box = _box_to_xyxy(polys[i]) if i < len(polys) else (0.0, 0.0, 0.0, 0.0)
        score = float(scores[i]) if i < len(scores) else 1.0
        lines.append(OcrLine(text, box, score))
    return lines


def _parse_paddle_result(raw):
    """Normalise PaddleOCR 2.x/3.x output into :class:`OcrLine` items."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        return _parse_v3_page(raw)

    lines = []
    for page in raw:
        if page is None:
            continue
        if isinstance(page, dict) or _field(page, "rec_texts") is not None:
            lines.extend(_parse_v3_page(page))
        elif isinstance(page, (list, tuple)):
            lines.extend(_parse_v2_page(page))
    return lines


# ---------------------------------------------------------------------------
# Table reconstruction
# ---------------------------------------------------------------------------
def lines_to_rows(lines, y_tolerance=ROW_Y_TOLERANCE):
    """Group recognised lines into visual rows (each sorted left-to-right).

    Args:
        lines (list[OcrLine]): Recognised text boxes.
        y_tolerance (float): Row-merge tolerance as a fraction of the median
            text-line height.

    Returns:
        list[list[OcrLine]]: Rows top-to-bottom; items within a row ordered by x.
    """
    items = [line for line in lines if line.text]
    if not items:
        return []

    heights = [max(1.0, line.box[3] - line.box[1]) for line in items]
    tolerance = max(1.0, median(heights) * y_tolerance)

    rows = []  # list of [y_center, [lines]]
    for line in sorted(items, key=lambda item: (item.box[1] + item.box[3]) / 2.0):
        y_center = (line.box[1] + line.box[3]) / 2.0
        if rows and abs(y_center - rows[-1][0]) <= tolerance:
            rows[-1][1].append(line)
            centres = [(item.box[1] + item.box[3]) / 2.0 for item in rows[-1][1]]
            rows[-1][0] = sum(centres) / len(centres)
        else:
            rows.append([y_center, [line]])

    result = []
    for _, row_items in rows:
        row_items.sort(key=lambda item: (item.box[0] + item.box[2]) / 2.0)
        result.append(row_items)
    return result


def column_bands(lines, page_width, min_gutter_ratio=MIN_GUTTER_RATIO):
    """Return the x-intervals (bands) that hold content - the table columns.

    A vertical whitespace "gutter" (an x-range no text box overlaps) separates
    two columns; each band is a maximal content region between consecutive
    gutters.

    Args:
        lines (list[OcrLine]): Text boxes on the page.
        page_width (float): Page width in pixels.
        min_gutter_ratio (float): Minimum gutter width as a fraction of the page.

    Returns:
        list[tuple[int, int]]: ``(x0, x1)`` content bands, left-to-right.
    """
    width = max(1, int(round(page_width)))
    covered = np.zeros(width, dtype=bool)
    for line in lines:
        x0 = int(max(0, min(width, round(line.box[0]))))
        x1 = int(max(0, min(width, round(line.box[2]))))
        if x1 > x0:
            covered[x0:x1] = True

    min_gutter = max(1, int(width * min_gutter_ratio))

    bands = []
    start = None
    gap = 0
    for x in range(width):
        if covered[x]:
            if start is None:
                start = x
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gutter:
                bands.append((start, x - gap + 1))
                start = None
                gap = 0
    if start is not None:
        bands.append((start, width))
    return bands


def _assign_column(bands, line):
    """Return the index of the band that best covers ``line``."""
    x0, x1 = line.box[0], line.box[2]
    best_index, best_overlap = 0, -1.0
    for index, (band_x0, band_x1) in enumerate(bands):
        overlap = min(x1, band_x1) - max(x0, band_x0)
        if overlap > best_overlap:
            best_index, best_overlap = index, overlap
    if best_overlap <= 0:
        centre = (x0 + x1) / 2.0
        best_index = min(
            range(len(bands)),
            key=lambda i: abs(centre - (bands[i][0] + bands[i][1]) / 2.0),
        )
    return best_index


def lines_to_table(lines, page_width=None):
    """Rebuild a table (list of rows of cell strings) from OCR lines.

    Args:
        lines (list[OcrLine]): Recognised text boxes for one page.
        page_width (float, optional): Page width; inferred from the boxes when
            omitted.

    Returns:
        list[list[str]]: A rectangular grid (one list per row). Empty when no
        text was recognised.
    """
    rows = lines_to_rows(lines)
    if not rows:
        return []

    flat = [line for row in rows for line in row]
    if page_width is None:
        page_width = max((line.box[2] for line in flat), default=0)
    bands = column_bands(flat, page_width)
    if not bands:
        return []

    table = []
    for row in rows:
        cells = [""] * len(bands)
        for line in row:
            index = _assign_column(bands, line)
            cells[index] = (
                f"{cells[index]} {line.text}".strip() if cells[index] else line.text
            )
        table.append(cells)
    return table


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------
def extract_tables_from_images(pdf_path, engine=None, resolution=DEFAULT_RESOLUTION,
                               max_pages=None):
    """Render a PDF and OCR each page into a reconstructed table + full text.

    Args:
        pdf_path (str or pathlib.Path): Path to the PDF.
        engine (object, optional): OCR engine exposing ``recognize(image)``.
            Defaults to :class:`PaddleOcrEngine` (Arabic).
        resolution (int): Rendering DPI.
        max_pages (int, optional): Limit the number of pages processed.

    Returns:
        tuple[list, str]: ``(tables, ocr_text)`` where ``tables`` is a list of
        reconstructed grids (one per page with text) and ``ocr_text`` is all
        recognised text joined by newlines.
    """
    engine = require_ocr_engine(engine)
    images = render_pdf_pages(pdf_path, resolution=resolution, max_pages=max_pages)

    tables = []
    texts = []
    for image in images:
        lines = engine.recognize(image)
        if not lines:
            continue
        texts.extend(line.text for line in lines)
        table = lines_to_table(lines, page_width=image.shape[1])
        if table:
            tables.append(table)
    return tables, "\n".join(texts)


def extract_transactions_from_pdf(pdf_path, rules=None, engine=None,
                                  resolution=DEFAULT_RESOLUTION, max_pages=None):
    """OCR a scanned PDF and return ``(raw_transactions, summary, ocr_text)``.

    The returned transactions use the OCR-detected header names; map them with
    :func:`formatter.format_dynamic` (see :func:`main.run`).

    Args:
        pdf_path (str or pathlib.Path): Path to the PDF.
        rules (dict, optional): Parsed rules; loaded from YAML when omitted.
        engine (object, optional): OCR engine (defaults to Arabic PaddleOCR).
        resolution (int): Rendering DPI.
        max_pages (int, optional): Limit the number of pages processed.

    Returns:
        tuple[pd.DataFrame, dict, str]: (raw transactions, summary, OCR text).
    """
    if rules is None:
        rules = formatter.load_rules()
    tables, ocr_text = extract_tables_from_images(
        pdf_path, engine=engine, resolution=resolution, max_pages=max_pages
    )
    raw, summary = extractor_pdf.tables_to_transactions(tables, fix_arabic=False)
    return raw, summary, ocr_text


def is_image_path(path):
    """Return True when ``path`` points at a direct image file (jpg/jpeg/png)."""
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def extract_tables_from_image(image_path, engine=None):
    """OCR a single image file into ``(tables, ocr_text)``.

    Unlike :func:`extract_tables_from_images`, the file is loaded directly (no
    page rendering / DPI options) and recognised as one table grid.

    Args:
        image_path (str or pathlib.Path): Path to a jpg/jpeg/png file.
        engine (object, optional): OCR engine exposing ``recognize(image)``.
            Defaults to :class:`PaddleOcrEngine` (Arabic).

    Returns:
        tuple[list, str]: ``(tables, ocr_text)`` - a list holding at most one
        reconstructed grid, and all recognised text joined by newlines.
    """
    engine = require_ocr_engine(engine)
    from PIL import Image

    with Image.open(image_path) as image:
        array = _pil_to_array(image)
    lines = engine.recognize(array)
    table = lines_to_table(lines, page_width=array.shape[1])
    return ([table] if table else []), "\n".join(line.text for line in lines)


def extract_transactions_from_image(image_path, engine=None):
    """OCR a single image file and return ``(raw_transactions, summary, ocr_text)``.

    Mirrors :func:`extract_transactions_from_pdf` for direct image uploads
    (jpg/jpeg/png): recognised lines are rebuilt into a table and run through
    the shared dynamic pipeline (``fix_arabic=False`` - Paddle already returns
    logical-order text). Map the result with :func:`formatter.format_dynamic`
    (see :func:`main.run`).

    Args:
        image_path (str or pathlib.Path): Path to a jpg/jpeg/png file.
        engine (object, optional): OCR engine (defaults to Arabic PaddleOCR).

    Returns:
        tuple[pd.DataFrame, dict, str]: (raw transactions, summary, OCR text).
    """
    tables, ocr_text = extract_tables_from_image(image_path, engine=engine)
    raw, summary = extractor_pdf.tables_to_transactions(tables, fix_arabic=False)
    return raw, summary, ocr_text


def should_use_ocr(text, pdf_path=None, min_chars_per_page=MIN_CHARS_PER_PAGE):
    """Heuristic: OCR when text extraction is empty/tiny or contains ``(cid:...)``.

    IMPORTANT: pass text with ``(cid:...)`` markers preserved (see
    :func:`extractor_pdf.get_raw_pdf_text`); :func:`extractor_pdf.get_pdf_text`
    strips those markers, which would silently disable CID-noise detection.

    Args:
        text (str): Text extracted from the PDF by pdfplumber.
        pdf_path (str or pathlib.Path, optional): Used to scale the threshold by
            page count.
        min_chars_per_page (int): Pages whose average text is shorter than this
            are treated as scanned.

    Returns:
        bool: ``True`` when the OCR pipeline should be used.
    """
    if extractor_pdf.has_cid_garbage(text):
        return True

    pages = 1
    if pdf_path is not None:
        try:
            pages = max(1, extractor_pdf.count_pdf_pages(pdf_path))
        except Exception:  # noqa: BLE001 - unreadable file -> assume single page
            pages = 1
    return len((text or "").strip()) < min_chars_per_page * pages


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if target is None:
        candidates = sorted(Path(__file__).resolve().parent.glob("*.pdf"))
        if not candidates:
            raise SystemExit("No PDF provided and none found in the workspace.")
        target = candidates[0]

    if not is_ocr_available():
        raise SystemExit(
            "PaddleOCR is not available. Install the OCR extras "
            "(paddleocr + paddlepaddle) to run the OCR pipeline."
        )

    tables, ocr_text = extract_tables_from_images(target)
    print(f"OCR'd {target.name}: {len(tables)} page table(s).")
    print("--- recognised text (first 500 chars) ---")
    print(ocr_text[:500])