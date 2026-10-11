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

import re
import threading
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

# Image preprocessing before OCR. Upscale is set to 1.0 (no resize) to avoid
# Out-Of-Memory (OOM) kills during PaddleOCR inference in memory-constrained
# environments (Streamlit / local). Contrast enhancement (autocontrast) is kept
# to make faint grey text crisp black-on-white.
OCR_UPSCALE_FACTOR = 1.0
OCR_CONTRAST_CUTOFF = 1

# Row grouping. Boxes are separated into strict horizontal rows on their
# vertical *centre* axis: after sorting by y-centre, a new row begins wherever
# two adjacent centres are farther apart than ``ROW_Y_TOLERANCE`` * the median
# box height. This is a 1-D DBSCAN (eps = that tolerance, min_samples = 1):
# it is drift-free and anchored to the robust median height, so neither a
# running mean nor a single tall Arabic box can chain-merge closely-stacked
# rows and read a whole column vertically into one concatenated cell.
ROW_Y_TOLERANCE = 0.5
# Minimum blank vertical band (as a fraction of page width) that separates two
# table columns.
MIN_GUTTER_RATIO = 0.012
# Below this many characters per page, a PDF is treated as scanned.
MIN_CHARS_PER_PAGE = 40

# Header-driven column bands for wide borderless tables: when the densest row
# (the header) spans at least HEADER_BAND_MIN_SPAN of the page and holds at
# least HEADER_BAND_MIN_BOXES boxes, its inter-box midpoints become the column
# boundaries (gutters narrower than MIN_GUTTER_RATIO would otherwise collapse
# several columns into one band). HEADER_MERGE_GAP_RATIO merges word-level
# splits of a single header label ("Notes /" + "Details").
HEADER_BAND_MIN_BOXES = 4
HEADER_BAND_MIN_SPAN = 0.6
HEADER_MERGE_GAP_RATIO = 0.005


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
        "plus the CPU runtime paddlepaddle>=3.3.1,<4 (Windows wheels for "
        "Python 3.9-3.13). The 2.x paddlepaddle line must not be used: it caps "
        "protobuf<=3.20.2, which conflicts with streamlit's protobuf>=5.26.1. "
        "Verify with `pip check` and `python -c \"import paddle\"`. Text-based "
        "PDFs continue to work without OCR."
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


def preprocess_for_ocr(array, scale=OCR_UPSCALE_FACTOR, contrast_cutoff=OCR_CONTRAST_CUTOFF):
    """Contrast-enhance an image before OCR, bypassing resize when scale <= 1.0.

    When scale > 1.0, the image is upscaled with Lanczos interpolation.
    Otherwise, resizing is bypassed to avoid high memory allocation and Out-Of-Memory
    (OOM) kills during PaddleOCR inference. The autocontrast step is preserved.

    Args:
        array (numpy.ndarray): ``(H, W, 3)`` uint8 image.
        scale (float): Upscale factor. Defaults to :data:`OCR_UPSCALE_FACTOR` (1.0).
        contrast_cutoff (float): Autocontrast percentile cutoff (0..49).

    Returns:
        numpy.ndarray: The enhanced ``(H, W, 3)`` uint8 image.
    """
    from PIL import Image, ImageOps

    image = Image.fromarray(array)
    if scale and scale > 1.0:
        width = max(1, int(image.width * scale))
        height = max(1, int(image.height * scale))
        image = image.resize((width, height), resample=Image.LANCZOS)

    # Stretch the histogram so faint grey text becomes crisp black-on-white.
    if contrast_cutoff:
        image = ImageOps.autocontrast(image, cutoff=contrast_cutoff)

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


# ---------------------------------------------------------------------------
# Process-wide OCR backend singleton
# ---------------------------------------------------------------------------
# PaddleX (the framework behind PaddleOCR 3.x) allows exactly ONE pipeline
# initialization per process; a second ``PaddleOCR(...)`` aborts with "PDX has
# already been initialized. Reinitialization is not supported." Streamlit
# re-runs the app script on every interaction and processes several uploads per
# run, so a per-object engine would re-initialize and fail. The backend is
# therefore built exactly once per worker process and shared; construction and
# inference are serialized because the pipeline is not thread-safe.
_OCR_ENGINE_LOCK = threading.RLock()
_OCR_BACKENDS: dict = {}

# PaddlePaddle 3.x on some Windows CPUs builds the pipeline successfully but
# then dies on the first predict() inside the oneDNN (MKL-DNN) kernels:
#   NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute
#   not support [pir::ArrayAttribute<pir::DoubleAttribute>]
#   (paddle/fluid/framework/new_executor/instruction/onednn/...)
# The cure is a backend built with enable_mkldnn=False (verified against
# paddlepaddle 3.4.0). It is applied as a one-shot rebuild rather than a
# blanket default so the fast oneDNN path is kept wherever it works
# (e.g. the Linux deploy target). Substring match on the marker above.
_ONEDNN_UNIMPLEMENTED = "ConvertPirAttribute2RuntimeAttribute"


def _mkldnn_off_candidates(lang, device, orient):
    """Constructor kwargs for the oneDNN-disabled rebuild, newest API first."""
    return [
        {
            "lang": lang,
            "device": device,
            "use_textline_orientation": orient,
            "enable_mkldnn": False,
        },
        {"lang": lang, "device": device, "enable_mkldnn": False},
        {"lang": lang, "enable_mkldnn": False},
    ]


def reset_ocr_backends():
    """Drop every cached OCR backend (test/maintenance helper)."""
    with _OCR_ENGINE_LOCK:
        _OCR_BACKENDS.clear()


def _load_paddleocr_class():
    """Import and return the ``PaddleOCR`` class, or ``None`` when missing."""
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # noqa: BLE001
        raise OcrUnavailableError(
            "PaddleOCR is not installed. Install the OCR extras "
            "(paddleocr + paddlepaddle) to use the scanned-invoice mode. "
            f"Original error: {exc}"
        ) from exc
    return PaddleOCR


def _create_ocr_backend(lang, device, orient):
    """Build a new PaddleOCR backend, degrading across constructor spellings."""
    if not is_ocr_available():
        raise OcrUnavailableError(_missing_ocr_message())
    PaddleOCR = _load_paddleocr_class()
    # A test or user seam may patch availability without supplying the real
    # PaddleOCR package. In that case fail loudly before attempting to use
    # a lightweight stand-in as if it were the trained backend.
    if PaddleOCR is None:
        raise OcrUnavailableError(_missing_ocr_message())

    # The constructor signature changed across PaddleOCR releases; try the
    # richer 3.x form first and degrade gracefully to the 2.x spellings.
    last_error = None
    for kwargs in PaddleOcrEngine._constructor_candidates(lang, device, orient):
        try:
            return PaddleOCR(**kwargs)
        except TypeError as exc:
            last_error = exc
            continue
        except Exception as exc:  # noqa: BLE001
            raise OcrUnavailableError(
                f"Could not initialise PaddleOCR (lang={lang!r}): {exc}"
            ) from exc
    raise OcrUnavailableError(f"Could not initialise PaddleOCR: {last_error!r}")


def get_ocr_backend(lang=OCR_LANG, device="cpu", use_textline_orientation=True):
    """Return the shared PaddleOCR backend, creating it exactly once.

    Keyed by ``(lang, device, use_textline_orientation)``: identical
    configurations reuse the single instance PaddleX allows per process
    (Streamlit reruns and multi-file runs included), while genuinely different
    configurations still get their own pipeline.

    Args:
        lang (str): PaddleOCR language code.
        device (str): Inference device (``"cpu"`` for CPU-only deployment).
        use_textline_orientation (bool): Text-line orientation detection flag.

    Returns:
        object: The shared ``PaddleOCR`` instance.

    Raises:
        OcrUnavailableError: When the stack cannot be imported or initialised.
    """
    key = (lang, device, use_textline_orientation)
    with _OCR_ENGINE_LOCK:
        backend = _OCR_BACKENDS.get(key)
        if backend is None:
            backend = _create_ocr_backend(lang, device, use_textline_orientation)
            _OCR_BACKENDS[key] = backend
        return backend


class PaddleOcrEngine:
    """Lazy wrapper around PaddleOCR configured for Arabic (``lang='ar'``).

    Targets the PaddleOCR 3.x pipeline API (``PaddleOCR(lang=..., device="cpu",
    use_textline_orientation=...)`` + ``predict(input=<numpy array>)``) and
    degrades gracefully to the legacy 2.x spellings (``use_angle_cls`` +
    ``ocr(...)``). Instantiation is deferred until the first :meth:`recognize`
    call, and the resulting backend is the process-wide singleton returned by
    :func:`get_ocr_backend` — PaddleX permits only one pipeline initialization
    per process, which Streamlit's script reruns would otherwise violate.
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
            {"lang": lang, "device": device, "use_textline_orientation": orient,
             "use_doc_orientation_classify": False, "use_doc_unwarping": False},
            {"lang": lang, "device": device},
            {"lang": lang, "use_angle_cls": orient},
            {"lang": lang},
        ]

    def _ensure_engine(self):
        """Return the shared backend, building it once per process."""
        if self._ocr is None:
            self._ocr = get_ocr_backend(
                self.lang, self.device, self.use_textline_orientation
            )
        return self._ocr

    def _rebuild_without_mkldnn(self):
        """Replace the cached backend with a oneDNN-free rebuild.

        Only used when :data:`_ONEDNN_UNIMPLEMENTED` is raised: the pipeline
        constructs fine but every predict() dies inside the oneDNN kernels, so
        the only cure is a backend built with ``enable_mkldnn=False``. The
        rebuild replaces the process-wide cache entry, so this happens at most
        once per worker and later pages reuse the working backend.
        """
        key = (self.lang, self.device, self.use_textline_orientation)
        PaddleOCR = _load_paddleocr_class()
        last_error = None
        for kwargs in _mkldnn_off_candidates(
            self.lang, self.device, self.use_textline_orientation
        ):
            try:
                backend = PaddleOCR(**kwargs)
            except TypeError as exc:  # backend without enable_mkldnn
                last_error = exc
                continue
            except Exception as exc:  # noqa: BLE001
                raise OcrUnavailableError(
                    "PaddleOCR failed to re-initialise with oneDNN disabled "
                    f"after a {_ONEDNN_UNIMPLEMENTED} error: {exc}"
                ) from exc
            with _OCR_ENGINE_LOCK:
                _OCR_BACKENDS[key] = backend
            return backend
        raise OcrUnavailableError(
            "PaddleOCR could not be re-initialised with oneDNN disabled "
            f"({last_error!r})."
        )

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
        """Run OCR on ``image`` and return a list of :class:`OcrLine`.

        The backend is the process-wide singleton (PaddleX allows a single
        pipeline initialization per process); inference is serialized because
        the pipeline is not thread-safe.

        A known PaddlePaddle 3.x oneDNN failure (see
        :data:`_ONEDNN_UNIMPLEMENTED`) is retried once against a rebuilt
        ``enable_mkldnn=False`` backend; any other error propagates.
        """
        backend = self._ensure_engine()
        with _OCR_ENGINE_LOCK:
            try:
                result = self._predict(backend, image)
            except NotImplementedError as exc:
                if _ONEDNN_UNIMPLEMENTED not in str(exc):
                    raise
                # One-shot, self-healing: the rebuilt backend replaces the
                # cached one, so this never repeats for later pages.
                backend = self._rebuild_without_mkldnn()
                result = self._predict(backend, image)
        return _parse_paddle_result(result)


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
def _otsu_threshold(values):
    """Return the 1-D Otsu threshold that best splits ``values`` into two.

    Maximises the between-class variance over every possible cut between two
    adjacent sorted values. Returns ``None`` when there are fewer than two
    values (no meaningful split exists).
    """
    values = sorted(values)
    n = len(values)
    if n < 2:
        return None

    total = sum(values)
    best_threshold, best_between = None, -1.0
    prefix_sum = 0.0
    for i in range(n - 1):
        prefix_sum += values[i]
        w0 = (i + 1) / n
        w1 = 1.0 - w0
        if w0 <= 0.0 or w1 <= 0.0:
            continue
        mu0 = prefix_sum / (i + 1)
        mu1 = (total - prefix_sum) / (n - i - 1)
        between = w0 * w1 * (mu0 - mu1) ** 2
        if between > best_between:
            best_between = between
            best_threshold = (values[i] + values[i + 1]) / 2.0
    return best_threshold


def _row_epsilon(centres, heights, y_tolerance):
    """Estimate the max within-row centre gap (the DBSCAN ``eps``).

    Two estimates are combined, keeping the *tighter* (smaller) one so that
    closely-stacked rows are split aggressively rather than read vertically:

    1. ``y_tolerance * median_height`` — the scale of a single text line;
    2. the 1-D Otsu threshold of the sorted centre gaps — the value that best
       separates the small within-row jitter from the larger line pitch. This
       is only trusted when the gap distribution is clearly bimodal (the
       largest gap ≫ the median gap), i.e. a real row structure exists.
    """
    base = max(1.0, median(heights) * y_tolerance)

    gaps = sorted(b - a for a, b in zip(centres, centres[1:]) if b - a > 1e-6)
    if len(gaps) < 2:
        return base

    otsu = _otsu_threshold(gaps)
    if otsu is None:
        return base

    median_gap = gaps[len(gaps) // 2]
    if gaps[-1] <= median_gap * 1.5 + 1.0:
        return base  # no clear bimodal structure -> keep the height-based value

    return max(1.0, min(base, otsu))


def lines_to_rows(lines, y_tolerance=ROW_Y_TOLERANCE):
    """Group recognised lines into strict horizontal rows (sorted left-to-right).

    Rows are separated on the boxes' vertical *centre* axis. Every box is
    sorted by y-centre and split into a new row wherever two adjacent centres
    are farther apart than ``eps`` (see :func:`_row_epsilon`: a fraction of the
    median box height, tightened by an Otsu split of the centre gaps when the
    rows are closely stacked). This is a 1-D DBSCAN (``eps`` neighbourhood,
    ``min_samples = 1``) over the centre axis, and it fixes the vertical-read
    failure on wide borderless tables:

    * boxes on the same visual line share a baseline, so their centres sit
      within a few pixels of each other — well under ``eps`` — while two
      stacked lines are a whole line-pitch apart (well over ``eps``);
    * the tolerance is anchored to the *median* height, so a single tall
      Arabic box cannot inflate it (unlike a global mean);
    * there is no running anchor or overlap test, so nothing drifts: the
      decision at each boundary is made from that pair of adjacent boxes alone.

    Args:
        lines (list[OcrLine]): Recognised text boxes.
        y_tolerance (float): Max within-row centre gap, as a fraction of the
            median box height.

    Returns:
        list[list[OcrLine]]: Rows top-to-bottom; items within a row ordered by x.
    """
    items = [line for line in lines if line.text]
    if not items:
        return []

    items.sort(key=lambda item: (item.box[1] + item.box[3]) / 2.0)
    centres = [(item.box[1] + item.box[3]) / 2.0 for item in items]
    heights = [max(1.0, item.box[3] - item.box[1]) for item in items]
    eps = _row_epsilon(centres, heights, y_tolerance)

    rows = []
    current = []
    last_centre = None
    for item in items:
        centre = (item.box[1] + item.box[3]) / 2.0
        if current and centre - last_centre > eps:
            rows.append(current)
            current = []
        current.append(item)
        last_centre = centre
    if current:
        rows.append(current)

    for row in rows:
        row.sort(key=lambda item: (item.box[0] + item.box[2]) / 2.0)
    return rows


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


def _header_bands(rows, page_width):
    """Column bands derived from the header row's own box gaps.

    On wide borderless invoices the gutters between neighbouring columns can
    be narrower than :data:`MIN_GUTTER_RATIO`, so page-wide gutter detection
    (:func:`column_bands`) collapses several columns into one band and the cell
    concatenates their values. The header row names every column, so its
    inter-box midpoints give exact column boundaries instead.

    Word-level splits of one label (``Notes /`` + ``Details``) are merged when
    their gap is below :data:`HEADER_MERGE_GAP_RATIO` of the page width.

    Args:
        rows (list[list[OcrLine]]): Rows from :func:`lines_to_rows`.
        page_width (float): Page width in pixels.

    Returns:
        list[tuple[float, float]]: ``(x0, x1)`` bands, or [] when the layout is
        not a wide table (caller then falls back to :func:`column_bands`).
    """
    if not rows or page_width <= 0:
        return []

    header = max(rows, key=len)  # the densest row is the column header
    if len(header) < HEADER_BAND_MIN_BOXES:
        return []

    boxes = sorted((float(line.box[0]), float(line.box[2])) for line in header)
    if boxes[-1][1] - boxes[0][0] < HEADER_BAND_MIN_SPAN * page_width:
        return []

    merge_gap = max(2.0, HEADER_MERGE_GAP_RATIO * page_width)
    merged = []
    for x0, x1 in boxes:
        if merged and x0 <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])

    if len(merged) < 2:
        return []

    bounds = [0.0]
    for i in range(len(merged) - 1):
        bounds.append((merged[i][1] + merged[i + 1][0]) / 2.0)
    bounds.append(float(page_width))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


# ---------------------------------------------------------------------------
# Coordinate-template reconstruction (known suppliers)
# ---------------------------------------------------------------------------
# A known supplier's header labels define the column X-template. Boxes are then
# sliced strictly by X onto those columns, which is far more robust than
# generic row clustering when the OCR returns multi-line / multi-column boxes
# (which no amount of Y-axis clustering can separate).
HEADER_MATCH_SIMILARITY = 0.50
# A box is snapped to a row if its y-centre is within this fraction of the
# median box height of the row backbone's centre.
SNAP_Y_TOLERANCE = 0.6


def _header_similarity(norm_box, norm_header):
    """Loose similarity matching between an OCR box text and an expected header label.

    Accommodates 1x OCR where font noise, missing accents, diacritics, or split
    tokens slightly alter header labels. Combines exact match, RapidFuzz token/partial
    similarity, substring containment, prefix/suffix matching, and token overlap.
    """
    if not norm_box or not norm_header:
        return 0.0
    if norm_box == norm_header:
        return 1.0

    scores = [formatter._similarity(norm_box, norm_header)]

    # Substring containment (e.g. "hot" in "hotel", "adult" in "adults", "date" in "date.")
    if len(norm_box) >= 2 and len(norm_header) >= 2:
        if norm_box in norm_header or norm_header in norm_box:
            sub_ratio = min(len(norm_box), len(norm_header)) / max(len(norm_box), len(norm_header))
            scores.append(max(0.70, sub_ratio))
        if norm_box.startswith(norm_header) or norm_header.startswith(norm_box):
            scores.append(0.80)

    # Word/token overlap (e.g. "notes" from "notes details", "egy" from "egy le")
    box_tokens = set(norm_box.split())
    target_tokens = set(norm_header.split())
    shared = box_tokens & target_tokens
    if shared:
        scores.append(len(shared) / min(len(box_tokens), len(target_tokens)))

    # RapidFuzz partial_ratio if available
    rf_fuzz = getattr(formatter, "_rf_fuzz", None)
    if rf_fuzz is not None:
        try:
            scores.append(rf_fuzz.partial_ratio(norm_box, norm_header) / 100.0)
        except Exception:
            pass

    return max(scores)


def _find_header_row(lines, header_names):
    """Locate the header row using a few expected header labels.

    Expected labels are fuzzy-matched against every box using loose similarity
    (_header_similarity); the tightest Y-band holding the most matches is the
    header row. We then return *every* box in that band (not just the matched
    ones), so its X-coordinates define the full column template even when some
    labels collide ("Total" / "Total Invoice" / "Total REST") or are slightly
    misread by 1x OCR.
    """
    norm_names = [(n, formatter._normalize_header(n)) for n in header_names]
    norm_names = [(n, nn) for n, nn in norm_names if nn]

    hits = []  # (name, line, score, y_centre)
    for line in lines:
        norm = formatter._normalize_header(line.text or "")
        if not norm:
            continue
        best_name, best_score = None, 0.0
        for name, nn in norm_names:
            score = _header_similarity(norm, nn)
            if score > best_score:
                best_name, best_score = name, score
        if best_name is not None and best_score >= HEADER_MATCH_SIMILARITY:
            hits.append((best_name, line, best_score, (line.box[1] + line.box[3]) / 2.0))

    if not hits:
        return []

    heights = [line.box[3] - line.box[1] for _, line, _, _ in hits]
    eps = max(1.0, median(heights) * 0.5)

    hits.sort(key=lambda h: h[3])
    bands, current, last_y = [], [], None
    for name, line, score, y in hits:
        if current and last_y is not None and y - last_y > eps:
            bands.append(current)
            current = []
        current.append((name, line, y))
        last_y = y
    if current:
        bands.append(current)

    best_band = max(bands, key=len)
    y_centres = [y for _, _, y in best_band]
    y_lo, y_hi = min(y_centres), max(y_centres)
    span = max(1.0, y_hi - y_lo, median(heights) * 0.5)

    return [
        line for line in lines
        if y_lo - span <= (line.box[1] + line.box[3]) / 2.0 <= y_hi + span
    ]


def _column_bounds(boxes, page_width):
    """Return per-column ``(x0, x1)`` strips from header boxes (sorted by x)."""
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = []
    for x0, _, x1, _ in boxes:
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])

    if not merged:
        return []
    if len(merged) == 1:
        return [(0.0, max(float(page_width), merged[0][1]))]

    right_edge = max(float(page_width), merged[-1][1])
    bounds = []
    for i in range(len(merged)):
        left = 0.0 if i == 0 else (merged[i - 1][1] + merged[i][0]) / 2.0
        right = right_edge if i == len(merged) - 1 else (merged[i][1] + merged[i + 1][0]) / 2.0
        bounds.append((left, right))
    return bounds


def _is_date_string(text):
    """Return True when text resembles an ISO or localized date."""
    t = (text or "").strip()
    return bool(
        re.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$", t)
        or re.match(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$", t)
        or re.match(r"^\d{1,2}[-/.]\w{3,}[-/.]\d{2,4}$", t)
        or re.match(r"^\w{3,}\s+\d{1,2},?\s+\d{4}$", t)
    )


def _is_numeric_column_header(header):
    """Return True when a column header indicates a numeric or monetary column."""
    if not header:
        return False
    lowered = str(header).lower().strip()
    # Explicitly exclude non-numeric column types
    non_numeric = (
        "date", "hotel", "v.no", "voucher", "room", "guest", "name",
        "tour", "trip", "nat", "info", "note", "contact", "phone", "tel",
    )
    if any(k in lowered for k in non_numeric):
        # Allow monetary sub-keywords e.g. "Rest$" or "Cost LE"
        if not any(c in lowered for c in ("$", "€", "£", "usd", "egp", "eur", "gbp", "cost", "total", "rest")):
            return False

    if extractor_pdf._is_currency_column(header):
        return True

    numeric_keywords = (
        "rest", "rate", "paid", "balance", "price", "amount", "cost", "total", "net",
        "adult", "adults", "child", "children", "ch", "inf", "infant", "infants",
        "pax", "count", "qty", "quantity", "dive", "intro",
        # Arabic headers
        "اجمالي", "إجمالي", "الإجمالي", "الاجمالي", "صافي", "الصافي",
        "المجموع", "مجموع", "الباقي", "باقي", "سعر", "مبلغ", "تكلفة",
        "مدفوع", "كبار", "أطفال", "اطفال", "رضع", "عدد",
    )
    return any(k in lowered for k in numeric_keywords)


def _split_discrete_numeric_values(text):
    """Force-split text containing multiple discrete numeric/currency values.

    Handles dense OCR boxes where lack of upscale causes numbers in a numeric
    column to be grouped into a single box without newlines (e.g.
    ``"888- 681- 1,493- 991-"``, ``"888 681 1,493 991"``, or
    ``"888-681-1493-991-"``).

    Returns:
        list[str]: Discrete values if multiple numeric values are found;
        otherwise ``[text]``.
    """
    text = (text or "").strip()
    if not text or _is_date_string(text):
        return [text] if text else []

    token_pattern = re.compile(
        r"^[+-]?\s*[$€£¥]?\s*-?\d[\d,]*(?:\.\d+)?\s*-?(?:\s*(?:LE|EGP|USD|EUR|GBP))?$"
    )

    # Strategy 1: Whitespace separation (with optional trailing dashes or detached '-' separators)
    space_tokens = text.split()
    if len(space_tokens) > 1:
        filtered = [t for t in space_tokens if t not in ("-", "—", "--")]
        if len(filtered) > 1 and all(token_pattern.match(t) for t in filtered):
            return filtered

    # Strategy 2: Dash-separated or attached tokens (e.g. '888-681-1493-991-')
    val_regex = re.compile(r"[$€£¥]?\s*-?\d[\d,]*(?:\.\d+)?-?")
    matches = [m.group(0).strip() for m in val_regex.finditer(text)]
    if len(matches) > 1:
        remainder = val_regex.sub("", text)
        if all(c in " -—,\t\r\n" for c in remainder):
            all_had_trailing = all(m.endswith("-") for m in matches)
            if not all_had_trailing:
                cleaned = [m[:-1] if m.endswith("-") else m for m in matches]
                return cleaned
            return matches

    return [text]


def _slice_line_by_x(line, bounds, header_names=None):
    """Slice one OCR box onto the column template.

    Returns a list of ``(column_index, text, y_centre)`` pieces:

    * a box whose text contains newlines (a multi-line OCR box) is split into
      one piece per line, its y-centres interpolated across the box's height;
    * a box in a numeric/currency column that lacks newlines but contains
      multiple discrete values (e.g. "888- 681- 1,493- 991-") is force-split
      vertically across the box's height;
    * a box that straddles several columns has its whitespace-separated values
      distributed left-to-right across the columns it spans.
    """
    x0, x1 = line.box[0], line.box[2]
    text = (line.text or "").strip()
    if not text:
        return []

    spanned = [i for i, (bx0, bx1) in enumerate(bounds) if x1 > bx0 and x0 < bx1]
    if not spanned:
        centre_x = (x0 + x1) / 2.0
        spanned = [min(
            range(len(bounds)),
            key=lambda i: abs(centre_x - (bounds[i][0] + bounds[i][1]) / 2.0),
        )]

    y0, y1 = line.box[1], line.box[3]
    y_centre = (y0 + y1) / 2.0
    height = max(1.0, y1 - y0)

    # Multi-line box: split into rows first.
    chunks = [c.strip() for c in re.split(r"[\n\r]+", text)]
    chunks = [c for c in chunks if c]
    if not chunks:
        return []
    if len(chunks) > 1:
        step = height / len(chunks)
        return [
            (spanned[0], c, y0 + (i + 0.5) * step)
            for i, c in enumerate(chunks)
        ]

    # Single-line box without newlines:
    # Determine the primary column this box belongs to (maximum horizontal overlap).
    primary_col = max(
        spanned,
        key=lambda i: min(x1, bounds[i][1]) - max(x0, bounds[i][0]),
    )
    header = (
        header_names[primary_col]
        if header_names and primary_col < len(header_names)
        else ""
    )
    is_numeric = _is_numeric_column_header(header)

    if is_numeric or not header:
        discrete_vals = _split_discrete_numeric_values(chunks[0])
        if len(discrete_vals) > 1:
            # Force-split vertically across the bounding box's height
            if len(spanned) == 1 or len(discrete_vals) > len(spanned) or is_numeric:
                step = height / len(discrete_vals)
                return [
                    (primary_col, val, y0 + (i + 0.5) * step)
                    for i, val in enumerate(discrete_vals)
                ]

    # Single-line box in a single column.
    if len(spanned) == 1:
        return [(spanned[0], chunks[0], y_centre)]

    # Straddles several columns: distribute values left-to-right.
    tokens = [t for t in re.split(r"\s+", chunks[0]) if t]
    if len(tokens) == len(spanned):
        return [(col, tok, y_centre) for col, tok in zip(spanned, tokens)]
    if len(tokens) > len(spanned):
        return [(col, tok, y_centre) for col, tok in zip(spanned, tokens[:len(spanned)])]
    # Fewer tokens than columns: keep the whole text in the first column.
    return [(spanned[0], " ".join(tokens), y_centre)]


def template_lines_to_table(lines, page_width, header_names):
    """Rebuild a table grid from a known supplier's header column template.

    The header labels in ``header_names`` are matched to OCR boxes; their
    X-coordinates become the column boundaries. Every remaining box is then
    sliced strictly by X onto those columns (splitting boxes that straddle a
    boundary, multi-line boxes, and unnewline numeric columns). Rows are
    recovered from the densest column (the "row backbone") by snapping the
    other columns' cells to its y-centres, so no generic row clustering is needed.

    Args:
        lines (list[OcrLine]): Recognised text boxes for one page.
        page_width (float): Page width in pixels.
        header_names (list[str]): Expected header labels for this supplier.

    Returns:
        list[list[str]]: A rectangular grid (header row first), or ``None``
        when the header template cannot be anchored reliably (the caller then
        falls back to the generic reconstruction).
    """
    matched = _find_header_row(lines, header_names)
    if len(matched) < 2:
        return None

    ordered = sorted(matched, key=lambda line: line.box[0])
    header_ids = {id(line) for line in ordered}
    bounds = _column_bounds([line.box for line in ordered], page_width)
    if len(bounds) < 2:
        return None
    header_row = [(line.text or "").strip() for line in ordered]

    # The table body starts below the header row: anything above it (the
    # invoice-metadata strip, titles) is page metadata, not transaction data.
    header_bottom = max(line.box[3] for line in ordered)

    heights = [max(1.0, ln.box[3] - ln.box[1]) for ln in lines if ln.text]
    median_h = median(heights) if heights else 1.0

    col_cells = {i: [] for i in range(len(bounds))}
    for line in lines:
        if not line.text or id(line) in header_ids:
            continue
        if (line.box[1] + line.box[3]) / 2.0 < header_bottom:
            continue  # above the table header -> metadata strip
        for col, text, y in _slice_line_by_x(line, bounds, header_row):
            col_cells.setdefault(col, []).append((text, y))

    if not any(col_cells.values()):
        return [header_row]

    # Row backbone: the densest column defines the row y-positions.
    backbone = max(col_cells, key=lambda c: len(col_cells[c]))
    backbone_cells = sorted(col_cells[backbone], key=lambda t: t[1])

    pitch = (
        (backbone_cells[-1][1] - backbone_cells[0][1]) / max(1, len(backbone_cells) - 1)
        if len(backbone_cells) > 1
        else median_h
    )
    snap = max(1.0, median_h * SNAP_Y_TOLERANCE, pitch * SNAP_Y_TOLERANCE)
    data_rows = [[""] * len(bounds) for _ in backbone_cells]
    for i, (row_text, _) in enumerate(backbone_cells):
        data_rows[i][backbone] = row_text

    # Assign every other column's cells one-to-one to the backbone rows (each
    # cell used once), so a value can never satisfy two rows.
    for col, cells in col_cells.items():
        if col == backbone:
            continue
        cells = sorted(cells, key=lambda t: t[1])
        if len(cells) == len(backbone_cells) and len(cells) > 0:
            if abs(cells[0][1] - backbone_cells[0][1]) <= snap * 3:
                for i in range(len(backbone_cells)):
                    data_rows[i][col] = cells[i][0]
                continue

        used = [False] * len(cells)
        for i, (_, y) in enumerate(backbone_cells):
            best_idx, best_diff = None, None
            for j, (_, cy) in enumerate(cells):
                if used[j]:
                    continue
                diff = abs(cy - y)
                if diff <= snap and (best_diff is None or diff < best_diff):
                    best_idx, best_diff = j, diff
            if best_idx is not None:
                used[best_idx] = True
                data_rows[i][col] = cells[best_idx][0]

    return [header_row] + data_rows


def _expand_tall_numeric_lines(lines):
    """Fallback for generic reconstruction: expand bounding boxes containing
    multiple discrete numeric values into separate vertical OcrLine items."""
    heights = [ln.box[3] - ln.box[1] for ln in lines if ln.text]
    med_h = median(heights) if heights else 15.0

    expanded = []
    for line in lines:
        if not line.text or "\n" in line.text or "\r" in line.text:
            expanded.append(line)
            continue
        vals = _split_discrete_numeric_values(line.text)
        if len(vals) > 1:
            x0, y0, x1, y1 = line.box
            h = max(1.0, y1 - y0)
            if h >= med_h * 1.3 or len(vals) >= 3:
                step = h / len(vals)
                for i, v in enumerate(vals):
                    expanded.append(
                        OcrLine(v, (x0, y0 + i * step, x1, y0 + (i + 1) * step), line.confidence)
                    )
                continue
        expanded.append(line)
    return expanded


def lines_to_table(lines, page_width=None, header_names=None):
    """Rebuild a table (list of rows of cell strings) from OCR lines.

    When ``header_names`` is supplied (a known supplier), the grid is rebuilt
    from the header column X-template (:func:`template_lines_to_table`) and the
    generic row clustering is bypassed entirely. Otherwise wide borderless
    tables are columned from the header row's own box gaps
    (:func:`_header_bands`) and every other layout uses page-wide gutter
    detection (:func:`column_bands`).

    Args:
        lines (list[OcrLine]): Recognised text boxes for one page.
        page_width (float, optional): Page width; inferred from the boxes when
            omitted.
        header_names (list[str], optional): Expected header labels; enables the
            coordinate-template reconstruction.

    Returns:
        list[list[str]]: A rectangular grid (one list per row). Empty when no
        text was recognised.
    """
    if header_names:
        table = template_lines_to_table(lines, page_width, header_names)
        if table is not None:
            return table

    lines = _expand_tall_numeric_lines(lines)
    rows = lines_to_rows(lines)
    if not rows:
        return []

    flat = [line for row in rows for line in row]
    if page_width is None:
        page_width = max((line.box[2] for line in flat), default=0)
    bands = _header_bands(rows, page_width)
    if not bands:
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
                               max_pages=None, rules=None, supplier=None):
    """Render a PDF and OCR each page into a reconstructed table + full text.

    When ``supplier`` or ``rules`` is provided and identifies a known supplier,
    the grid is rebuilt from that supplier's header column template
    (see :func:`template_lines_to_table`) instead of generic row clustering.

    Args:
        pdf_path (str or pathlib.Path): Path to the PDF.
        engine (object, optional): OCR engine exposing ``recognize(image)``.
            Defaults to :class:`PaddleOcrEngine` (Arabic).
        resolution (int): Rendering DPI.
        max_pages (int, optional): Limit the number of pages processed.
        rules (dict, optional): Parsed supplier rules, used to detect the
            supplier and anchor the column template.
        supplier (str, optional): Explicit supplier key to anchor the column
            template, overriding auto-detection from text.

    Returns:
        tuple[list, str]: ``(tables, ocr_text)`` where ``tables`` is a list of
        reconstructed grids (one per page with text) and ``ocr_text`` is all
        recognised text joined by newlines.
    """
    engine = require_ocr_engine(engine)
    images = render_pdf_pages(pdf_path, resolution=resolution, max_pages=max_pages)
    # Upscale + contrast-stretch each page so the detector emits one box per
    # value instead of one tall box spanning a whole column.
    images = [preprocess_for_ocr(image) for image in images]

    # Recognise every page first so supplier detection sees the full text.
    pages_lines = []
    texts = []
    for image in images:
        lines = engine.recognize(image)
        pages_lines.append(lines)
        texts.extend(line.text for line in lines)
    ocr_text = "\n".join(texts)

    header_names = None
    if rules is not None:
        target_supplier = supplier or formatter.detect_supplier(ocr_text, rules)
        if target_supplier and target_supplier in rules.get("suppliers", {}):
            header_names = formatter.profile_header_names(target_supplier, rules)

    tables = []
    for image, lines in zip(images, pages_lines):
        if not lines:
            continue
        table = lines_to_table(
            lines, page_width=image.shape[1], header_names=header_names
        )
        if table:
            tables.append(table)
    return tables, ocr_text


def extract_transactions_from_pdf(pdf_path, rules=None, engine=None,
                                  resolution=DEFAULT_RESOLUTION, max_pages=None,
                                  supplier=None):
    """OCR a scanned PDF and return ``(raw_transactions, summary, ocr_text)``.

    The returned transactions use the OCR-detected header names; map them with
    :func:`formatter.format_dynamic` or :func:`formatter.format_transactions`
    (see :func:`main.run`).

    Args:
        pdf_path (str or pathlib.Path): Path to the PDF.
        rules (dict, optional): Parsed rules; loaded from YAML when omitted.
        engine (object, optional): OCR engine (defaults to Arabic PaddleOCR).
        resolution (int): Rendering DPI.
        max_pages (int, optional): Limit the number of pages processed.
        supplier (str, optional): Explicit supplier key override.

    Returns:
        tuple[pd.DataFrame, dict, str]: (raw transactions, summary, OCR text).
    """
    if rules is None:
        rules = formatter.load_rules()
    tables, ocr_text = extract_tables_from_images(
        pdf_path, engine=engine, resolution=resolution, max_pages=max_pages,
        rules=rules, supplier=supplier,
    )
    raw, summary = extractor_pdf.tables_to_transactions(tables, fix_arabic=False)
    return raw, summary, ocr_text


def is_image_path(path):
    """Return True when ``path`` points at a direct image file (jpg/jpeg/png)."""
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def extract_tables_from_image(image_path, engine=None, rules=None, supplier=None):
    """OCR a single image file into ``(tables, ocr_text)``.

    Unlike :func:`extract_tables_from_images`, the file is loaded directly (no
    page rendering / DPI options) and recognised as one table grid.

    Args:
        image_path (str or pathlib.Path): Path to a jpg/jpeg/png file.
        engine (object, optional): OCR engine exposing ``recognize(image)``.
            Defaults to :class:`PaddleOcrEngine` (Arabic).
        rules (dict, optional): Parsed supplier rules, used to detect the
            supplier and anchor the column template.
        supplier (str, optional): Explicit supplier key override.

    Returns:
        tuple[list, str]: ``(tables, ocr_text)`` - a list holding at most one
        reconstructed grid, and all recognised text joined by newlines.
    """
    engine = require_ocr_engine(engine)
    from PIL import Image

    with Image.open(image_path) as image:
        array = _pil_to_array(image)
    # Upscale + contrast-stretch so the detector emits one box per value.
    array = preprocess_for_ocr(array)
    lines = engine.recognize(array)
    ocr_text = "\n".join(line.text for line in lines)

    header_names = None
    if rules is not None:
        target_supplier = supplier or formatter.detect_supplier(ocr_text, rules)
        if target_supplier and target_supplier in rules.get("suppliers", {}):
            header_names = formatter.profile_header_names(target_supplier, rules)

    table = lines_to_table(
        lines, page_width=array.shape[1], header_names=header_names
    )
    return ([table] if table else []), ocr_text


def extract_transactions_from_image(image_path, engine=None, rules=None, supplier=None):
    """OCR a single image file and return ``(raw_transactions, summary, ocr_text)``.

    Mirrors :func:`extract_transactions_from_pdf` for direct image uploads
    (jpg/jpeg/png): recognised lines are rebuilt into a table and run through
    the shared dynamic pipeline (``fix_arabic=False`` - Paddle already returns
    logical-order text). Map the result with :func:`formatter.format_dynamic`
    or :func:`formatter.format_transactions` (see :func:`main.run`).

    Args:
        image_path (str or pathlib.Path): Path to a jpg/jpeg/png file.
        engine (object, optional): OCR engine (defaults to Arabic PaddleOCR).
        rules (dict, optional): Parsed supplier rules, used to detect the
            supplier and anchor the column template.
        supplier (str, optional): Explicit supplier key override.

    Returns:
        tuple[pd.DataFrame, dict, str]: (raw transactions, summary, OCR text).
    """
    if rules is None:
        rules = formatter.load_rules()
    tables, ocr_text = extract_tables_from_image(
        image_path, engine=engine, rules=rules, supplier=supplier
    )
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