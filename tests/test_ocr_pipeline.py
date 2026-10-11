"""tests/test_ocr_pipeline.py

Tests for the OCR (scanned-invoice) pipeline:

* ``(cid:...)`` detection/cleanup and the auto-routing heuristic;
* pdfplumber-based page rendering;
* OCR result parsing (PaddleOCR v2 and v3 shapes);
* box -> rows -> columns -> table reconstruction;
* the end-to-end OCR pipeline mapping to the standard schema and exporting with
  the Subtotal row + live EGP formulas intact.

A fake OCR engine is injected so the pipeline is verified without PaddleOCR
(which is not installable on every Python version). Runnable directly::

    python tests/test_ocr_pipeline.py

or with pytest.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from openpyxl import load_workbook  # noqa: E402

import exporter  # noqa: E402
import extractor_ocr  # noqa: E402
import extractor_pdf  # noqa: E402
import formatter  # noqa: E402
import main  # noqa: E402
from extractor_ocr import OcrLine  # noqa: E402


STANDARD_COLUMNS = list(formatter.load_rules()["standard_columns"])
PAGE_WIDTH = 900


class _FakeEngine:
    """OCR engine stub returning a fixed set of lines for any image."""

    def __init__(self, lines):
        self._lines = lines
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        return list(self._lines)


def _fake_lines():
    """OCR lines for a mock unknown-supplier invoice (5 columns, 3 data rows)."""
    return [
        # header row
        OcrLine("Dt", (10, 10, 60, 30)),
        OcrLine("Hotel", (200, 10, 320, 30)),
        OcrLine("Ad.", (450, 10, 500, 30)),
        OcrLine("ch", (550, 10, 590, 30)),
        OcrLine("Cost", (700, 10, 760, 30)),
        # data row 1
        OcrLine("1-Sep", (10, 40, 90, 60)),
        OcrLine("Hurghada", (200, 40, 360, 60)),
        OcrLine("2", (460, 40, 480, 60)),
        OcrLine("1", (555, 40, 575, 60)),
        OcrLine("100", (720, 40, 760, 60)),
        # data row 2
        OcrLine("2-Sep", (10, 70, 90, 90)),
        OcrLine("Cairo", (200, 70, 320, 90)),
        OcrLine("1", (460, 70, 480, 90)),
        OcrLine("0", (555, 70, 575, 90)),
        OcrLine("500", (720, 70, 760, 90)),
        # summary row
        OcrLine("Total", (10, 100, 80, 120)),
        OcrLine("600", (720, 100, 760, 120)),
    ]


def _first_pdf():
    pdfs = sorted(ROOT.glob("*.pdf")) + sorted(ROOT.glob("*.PDF"))
    return pdfs[0] if pdfs else None


# ---------------------------------------------------------------------------
# 1. cid garbage + routing heuristic
# ---------------------------------------------------------------------------
def test_has_cid_garbage_and_cleanup():
    assert extractor_pdf.has_cid_garbage("hello (cid:12) world") is True
    assert extractor_pdf.has_cid_garbage("clean readable text") is False
    # (cid:NN) markers are stripped during cell cleanup.
    assert extractor_pdf._clean_cell("(cid:12) 5 (cid:34)") == "5"


def test_should_use_ocr_heuristic():
    assert extractor_ocr.should_use_ocr("") is True
    assert extractor_ocr.should_use_ocr("(cid:12) (cid:345)") is True
    long_text = "This is a normal digital invoice with plenty of readable text. " * 5
    assert extractor_ocr.should_use_ocr(long_text) is False


def test_resolve_engine_modes():
    pdf = _first_pdf()
    if pdf is None:
        print("      (skipped: no sample PDF in the workspace)")
        return
    assert main._resolve_engine("ocr", pdf) is True
    assert main._resolve_engine("text", pdf) is False
    assert isinstance(main._resolve_engine("auto", pdf), bool)
    try:
        main._resolve_engine("bogus", pdf)
        assert False, "expected ValueError for an unknown engine"
    except ValueError:
        pass


def _patch_get_raw_pdf_text(fake_text):
    original = extractor_pdf.get_raw_pdf_text
    extractor_pdf.get_raw_pdf_text = lambda path: fake_text
    return original


def test_auto_routes_cid_noise_to_ocr():
    pdf = _first_pdf()
    if pdf is None:
        print("      (skipped: no sample PDF in the workspace)")
        return
    original = _patch_get_raw_pdf_text("(cid:12) (cid:345)")
    try:
        assert main._resolve_engine("auto", pdf) is True
    finally:
        extractor_pdf.get_raw_pdf_text = original


def test_auto_uses_text_when_layer_is_readable():
    pdf = _first_pdf()
    if pdf is None:
        print("      (skipped: no sample PDF in the workspace)")
        return
    original = _patch_get_raw_pdf_text("x" * 500)
    try:
        assert main._resolve_engine("auto", pdf) is False
    finally:
        extractor_pdf.get_raw_pdf_text = original


# ---------------------------------------------------------------------------
# 2. Page rendering
# ---------------------------------------------------------------------------
def test_render_pdf_pages_produces_rgb_arrays():
    pdf = _first_pdf()
    if pdf is None:
        print("      (skipped: no sample PDF in the workspace)")
        return
    images = extractor_ocr.render_pdf_pages(pdf, resolution=72, max_pages=1)
    assert len(images) == 1
    array = images[0]
    assert array.ndim == 3 and array.shape[2] == 3
    assert array.shape[0] > 0 and array.shape[1] > 0


def test_preprocess_for_ocr_upscales_and_stays_uint8_rgb():
    try:
        import numpy as np
        from PIL import Image
    except ImportError:  # pragma: no cover
        print("      (skipped: Pillow/numpy not installed)")
        return

    # A faint grey-on-white strip (the typical dense numeric column).
    base = np.full((60, 200, 3), 255, dtype=np.uint8)
    base[10:50, 40:160] = 128  # low-contrast text band
    out = extractor_ocr.preprocess_for_ocr(base, scale=3.0)

    assert out.shape[0] == 180 and out.shape[1] == 600 and out.shape[2] == 3
    assert out.dtype == np.uint8
    # Contrast stretch pushes the grey band towards black (centre of the band,
    # away from the interpolated edge).
    assert int(out[90, 300, 0]) < 128


def test_preprocess_for_ocr_noop_at_scale_one():
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        print("      (skipped: numpy not installed)")
        return
    base = np.full((40, 40, 3), 200, dtype=np.uint8)
    out = extractor_ocr.preprocess_for_ocr(base, scale=1.0, contrast_cutoff=0)
    assert out.shape == (40, 40, 3)
    assert out.dtype == np.uint8


def test_default_upscale_factor_stays_under_paddle_limit():
    """The default upscale must keep tall invoices under PaddleOCR's ~4000px
    max-side limit (a 3x upscale crashed the Streamlit worker with OOM)."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        print("      (skipped: numpy not installed)")
        return

    # A tall photographed invoice (iPhone-ish portrait), the shape that
    # previously became 2211x4002 at 3x and blew the 4000px limit.
    tall = np.full((1334, 737, 3), 255, dtype=np.uint8)
    out = extractor_ocr.preprocess_for_ocr(tall)
    assert max(out.shape[0], out.shape[1]) < 4000
    assert out.shape[0] == int(1334 * extractor_ocr.OCR_UPSCALE_FACTOR)
    assert extractor_ocr.OCR_UPSCALE_FACTOR <= 2.0


# ---------------------------------------------------------------------------
# 3. PaddleOCR result parsing (v2 and v3 shapes)
# ---------------------------------------------------------------------------
def test_parse_paddle_v2_result():
    raw = [
        [
            [[[10, 10], [60, 10], [60, 30], [10, 30]], ("Dt", 0.99)],
            [[[200, 10], [320, 10], [320, 30], [200, 30]], ("Hotel", 0.98)],
        ]
    ]
    lines = extractor_ocr._parse_paddle_result(raw)
    assert [line.text for line in lines] == ["Dt", "Hotel"]
    assert lines[0].box == (10.0, 10.0, 60.0, 30.0)
    assert lines[1].confidence == 0.98


def test_parse_paddle_v3_result():
    raw = [
        {
            "rec_texts": ["Dt", "Hotel"],
            "rec_scores": [0.99, 0.98],
            "dt_polys": [
                [[10, 10], [60, 10], [60, 30], [10, 30]],
                [[200, 10], [320, 10], [320, 30], [200, 30]],
            ],
        }
    ]
    lines = extractor_ocr._parse_paddle_result(raw)
    assert [line.text for line in lines] == ["Dt", "Hotel"]
    assert lines[0].box == (10.0, 10.0, 60.0, 30.0)


def test_paddle_engine_prefers_modern_constructor():
    candidates = extractor_ocr.PaddleOcrEngine._constructor_candidates(
        "ar", "cpu", True
    )
    # PaddleOCR 3.x takes lang/device/use_textline_orientation. The removed 2.x
    # `show_log` flag is NOT passed: paddleocr 3.7.0 funnels unknown kwargs into
    # parse_common_args(), which raises ValueError("Unknown argument: show_log")
    # rather than a TypeError. _create_ocr_backend only falls through to the
    # next candidate on TypeError, so show_log would abort the whole backend
    # with OcrUnavailableError instead of degrading.
    assert candidates[0] == {
        "lang": "ar",
        "device": "cpu",
        "use_textline_orientation": True,
    }
    assert {"lang": "ar"} in candidates


def test_constructor_candidates_avoid_rejected_kwargs():
    """The candidate used against PaddleOCR 3.x passes no rejected kwarg.

    Regression guard: paddleocr 3.7.0 funnels unknown kwargs into
    parse_common_args(), which raises ValueError("Unknown argument: ...")
    rather than a TypeError. _create_ocr_backend only falls through to the next
    candidate on TypeError, so a stray 2.x-only flag (the original `show_log`)
    would abort the whole backend with OcrUnavailableError instead of
    degrading.

    Only the FIRST candidate is checked: it is the one tried against a 3.x
    install. The remaining `use_angle_cls` / bare-`lang` entries are deliberate
    2.x fallbacks, reached only after a 3.x backend has already raised
    TypeError, so validating them against 3.x would be wrong.
    """
    import inspect

    try:
        from paddleocr import PaddleOCR

        params = set(inspect.signature(PaddleOCR.__init__).parameters)
    except Exception:  # noqa: BLE001 - PaddleOCR not installed here
        return

    first = extractor_ocr.PaddleOcrEngine._constructor_candidates(
        "ar", "cpu", True
    )[0]
    # Explicit 3.x parameters, plus the common_args forwarded via **kwargs.
    common_args = {
        "device", "engine", "engine_config", "enable_hpi", "use_tensorrt",
        "precision", "enable_mkldnn", "mkldnn_cache_capacity", "cpu_threads",
        "enable_cinn",
    }
    unknown = set(first) - (params | common_args)
    assert not unknown, f"first candidate {first} passes rejected kwargs: {unknown}"


# ---------------------------------------------------------------------------
# 2b. oneDNN recovery (paddlepaddle 3.x on some Windows CPUs)
# ---------------------------------------------------------------------------
def test_recognize_recovers_from_onednn_failure():
    """A oneDNN NotImplementedError triggers one enable_mkldnn=False rebuild."""
    onednn_error = NotImplementedError(
        "(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support "
        "[pir::ArrayAttribute<pir::DoubleAttribute>]"
    )

    class _OneDnnBackend:
        """Fails like the real oneDNN kernels on every predict()."""

        def predict(self, *args, **kwargs):
            raise onednn_error

    class _WorkingBackend:
        def __init__(self):
            self.calls = 0

        def predict(self, *args, **kwargs):
            self.calls += 1
            return {
                "rec_texts": ["Total", "EGP"],
                "rec_scores": [0.9, 0.8],
                "dt_polys": [
                    np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float),
                    np.array(
                        [[0, 20], [10, 20], [10, 30], [0, 30]], dtype=float
                    ),
                ],
            }

    rebuilt = _WorkingBackend()
    seen_kwargs = []

    class _FakePaddleOcr:
        """Stands in for the PaddleOCR class; records the constructor kwargs."""

        def __new__(cls, **kwargs):
            seen_kwargs.append(kwargs)
            return rebuilt

    original_loader = extractor_ocr._load_paddleocr_class
    original_backend = extractor_ocr.get_ocr_backend
    extractor_ocr._load_paddleocr_class = lambda: _FakePaddleOcr
    extractor_ocr.get_ocr_backend = lambda *a, **k: _OneDnnBackend()
    try:
        engine = extractor_ocr.PaddleOcrEngine()
        lines = engine.recognize(np.zeros((40, 80, 3), dtype=np.uint8))
    finally:
        extractor_ocr._load_paddleocr_class = original_loader
        extractor_ocr.get_ocr_backend = original_backend

    # The rebuild happened and disabled oneDNN explicitly.
    assert seen_kwargs, "expected a oneDNN-free rebuild"
    assert seen_kwargs[0]["enable_mkldnn"] is False
    assert seen_kwargs[0]["lang"] == "ar"
    # The retry succeeded and returned parsed text.
    assert rebuilt.calls == 1
    assert [line.text for line in lines] == ["Total", "EGP"]


def test_recognize_does_not_swallow_other_notimplemented():
    """An unrelated NotImplementedError must propagate, not trigger a rebuild."""
    class _OtherBackend:
        def predict(self, *args, **kwargs):
            raise NotImplementedError("something else entirely")

    original_loader = extractor_ocr._load_paddleocr_class
    original_backend = extractor_ocr.get_ocr_backend
    calls = []
    extractor_ocr._load_paddleocr_class = lambda **kwargs: calls.append(kwargs)
    extractor_ocr.get_ocr_backend = lambda *a, **k: _OtherBackend()
    try:
        engine = extractor_ocr.PaddleOcrEngine()
        try:
            engine.recognize(np.zeros((40, 80, 3), dtype=np.uint8))
            assert False, "expected the original NotImplementedError"
        except NotImplementedError as exc:
            assert "something else entirely" in str(exc)
    finally:
        extractor_ocr._load_paddleocr_class = original_loader
        extractor_ocr.get_ocr_backend = original_backend

    assert not calls, "must not rebuild for an unrelated error"


def test_mkldnn_off_candidates_disable_onednn():
    """Every rebuild candidate pins enable_mkldnn=False and keeps lang."""
    candidates = extractor_ocr._mkldnn_off_candidates("ar", "cpu", True)
    assert candidates
    for kwargs in candidates:
        assert kwargs["enable_mkldnn"] is False
        assert kwargs["lang"] == "ar"


class _PredictRecordingBackend:
    """Mimics the PaddleOCR 3.x predict(input=...) entry point."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def predict(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.payload


class _LegacyOcrBackend:
    """Mimics the legacy 2.x ocr(image) entry point."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def ocr(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.payload


# ---------------------------------------------------------------------------
# 4. Table reconstruction
# ---------------------------------------------------------------------------
def test_predict_prefers_keyword_input():
    payload = [
        [
            [[[10, 10], [60, 10], [60, 30], [10, 30]], ("Dt", 0.99)],
        ]
    ]
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    backend = _PredictRecordingBackend(payload)
    lines = extractor_ocr.PaddleOcrEngine._predict(backend, image)
    assert backend.calls and backend.calls[0][1] == {"input": image}
    assert [line.text for line in extractor_ocr._parse_paddle_result(lines)] == ["Dt"]


def test_predict_falls_back_to_legacy_ocr():
    payload = [
        [
            [[[10, 10], [60, 10], [60, 30], [10, 30]], ("Dt", 0.99)],
        ]
    ]
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    backend = _LegacyOcrBackend(payload)
    raw = extractor_ocr.PaddleOcrEngine._predict(backend, image)
    assert backend.calls and backend.calls[0][0] == (image,)
    assert [line.text for line in extractor_ocr._parse_paddle_result(raw)] == ["Dt"]


def test_lines_to_rows_groups_by_y():
    rows = extractor_ocr.lines_to_rows(_fake_lines())
    assert len(rows) == 4
    assert [line.text for line in rows[1]] == ["1-Sep", "Hurghada", "2", "1", "100"]


def test_column_bands_detects_gutters():
    bands = extractor_ocr.column_bands(_fake_lines(), PAGE_WIDTH)
    assert len(bands) == 5
    # Bands are ordered left-to-right and non-overlapping.
    assert all(bands[i][1] <= bands[i + 1][0] for i in range(len(bands) - 1))


def test_lines_to_table_builds_grid():
    table = extractor_ocr.lines_to_table(_fake_lines(), page_width=PAGE_WIDTH)
    assert len(table) == 4
    assert table[0] == ["Dt", "Hotel", "Ad.", "ch", "Cost"]
    assert table[1] == ["1-Sep", "Hurghada", "2", "1", "100"]
    assert table[3] == ["Total", "", "", "", "600"]


def test_reconstruction_handles_empty_and_full_width_input():
    assert extractor_ocr.lines_to_rows([]) == []
    assert extractor_ocr.lines_to_table([]) == []
    bands = extractor_ocr.column_bands(
        [OcrLine("whole row text", (0, 10, PAGE_WIDTH, 30))], PAGE_WIDTH
    )
    assert len(bands) == 1


def test_predict_reports_unusable_backend():
    try:
        extractor_ocr.PaddleOcrEngine._predict(object(), None)
        assert False, "expected OcrUnavailableError"
    except extractor_ocr.OcrUnavailableError as exc:
        assert "predict()" in str(exc)


# ---------------------------------------------------------------------------
# 5. End-to-end OCR pipeline -> standard schema
# ---------------------------------------------------------------------------
def test_ocr_pipeline_maps_to_standard_schema():
    pdf = _first_pdf()
    if pdf is None:
        print("      (skipped: no sample PDF in the workspace)")
        return
    engine = _FakeEngine(_fake_lines())
    raw, summary, ocr_text = extractor_ocr.extract_transactions_from_pdf(
        pdf, engine=engine, resolution=72, max_pages=1
    )
    assert engine.calls == 1
    assert "Dt" in ocr_text and "Hurghada" in ocr_text

    formatted = formatter.format_dynamic(raw)
    assert list(formatted.columns) == STANDARD_COLUMNS
    assert len(formatted) == 2  # the "Total" row is separated out
    assert list(formatted["Hotel"]) == ["Hurghada", "Cairo"]
    assert list(formatted["Adult"]) == [2, 1]
    assert list(formatted["Child"]) == [1, 0]
    assert list(formatted["Total"]) == [100, 500]
    assert formatted["Date"].iloc[0].endswith("-09-01")


def test_main_run_ocr_exports_with_subtotal_and_egp_formulas():
    pdf = _first_pdf()
    if pdf is None:
        print("      (skipped: no sample PDF in the workspace)")
        return
    engine = _FakeEngine(_fake_lines())

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "ocr_out.xlsx"
        out, formatted, summary, supplier_key = main.run(
            pdf,
            output_path=out_path,
            engine="ocr",
            ocr_engine=engine,
            ocr_resolution=72,
            ocr_max_pages=1,
        )

        assert supplier_key == extractor_pdf.DYNAMIC_SUPPLIER_KEY
        assert list(formatted.columns) == STANDARD_COLUMNS
        n = len(formatted)
        assert n == 2

        ws = load_workbook(out)["Invoices"]
        header_row = exporter.HEADER_ROW
        headers = {str(c.value): c.column_letter for c in ws[header_row]}
        assert "EGP Converter" in headers

        conv = headers["EGP Converter"]
        first_data = header_row + 1
        last_data = header_row + n
        total_row = last_data + 1

        # The live EGP converter formula on individual rows is intact.
        assert str(ws[f"{conv}{first_data}"].value).startswith("=")
        assert "$B$2" in ws[f"{conv}{first_data}"].value

        # The Subtotal row carries a SUM over the exact data range.
        assert ws[f"{conv}{total_row}"].value == (
            f"=SUM({conv}{first_data}:{conv}{last_data})"
        )


def test_main_run_unavailable_ocr_reports_actionable_error():
    pdf = _first_pdf()
    if pdf is None:
        print("      (skipped: no sample PDF in the workspace)")
        return
    original = extractor_ocr.is_ocr_available
    extractor_ocr.is_ocr_available = lambda: False
    try:
        main.run(pdf, output_path=Path("unused.xlsx"), engine="ocr")
        assert False, "expected OcrUnavailableError"
    except extractor_ocr.OcrUnavailableError as exc:
        message = str(exc)
        assert "PaddleOCR" in message
        assert "paddlepaddle" in message.lower()
        assert "Text-based PDFs continue to work" in message
    finally:
        extractor_ocr.is_ocr_available = original


def test_require_ocr_engine_validates_custom_engines():
    fake = _FakeEngine(_fake_lines())
    assert extractor_ocr.require_ocr_engine(fake) is fake
    try:
        extractor_ocr.require_ocr_engine(object())
        assert False, "expected OcrUnavailableError for an invalid engine"
    except extractor_ocr.OcrUnavailableError as exc:
        assert "recognize" in str(exc)


def test_ocr_dependency_status_matches_availability():
    ready, missing = extractor_ocr.ocr_dependency_status()
    assert ready == extractor_ocr.is_ocr_available()
    assert set(missing) <= {"paddleocr", "paddlepaddle"}
    assert ready == (len(missing) == 0)


def test_should_use_ocr_scales_exactly_per_page():
    assert extractor_ocr.should_use_ocr("x" * 39) is True
    assert extractor_ocr.should_use_ocr("x" * 40) is False
    assert extractor_ocr.should_use_ocr("x" * 100) is False
    assert extractor_ocr.should_use_ocr("", pdf_path=None) is True


def test_empty_ocr_pipeline_yields_empty_standard_frame():
    pdf = _first_pdf()
    if pdf is None:
        print("      (skipped: no sample PDF in the workspace)")
        return

    class _SilentEngine:
        def __init__(self):
            self.calls = 0

        def recognize(self, image):
            self.calls += 1
            return []

    engine = _SilentEngine()
    tables, ocr_text = extractor_ocr.extract_tables_from_images(
        pdf, engine=engine, resolution=72, max_pages=1
    )
    assert engine.calls == 1
    assert tables == []
    assert ocr_text == ""

    raw, summary, _ = extractor_ocr.extract_transactions_from_pdf(
        pdf, engine=_SilentEngine(), resolution=72, max_pages=1
    )
    assert raw.empty
    assert summary == {}

    formatted = formatter.format_dynamic(raw)
    assert list(formatted.columns) == STANDARD_COLUMNS
    assert formatted.empty


# ---------------------------------------------------------------------------
# 6. Graceful degradation when PaddleOCR is unavailable
# ---------------------------------------------------------------------------
def test_ocr_unavailable_reports_cleanly():
    if extractor_ocr.is_ocr_available():
        print("      (skipped: PaddleOCR is installed)")
        return
    engine = extractor_ocr.PaddleOcrEngine()
    try:
        engine.recognize(np.zeros((8, 8, 3), dtype=np.uint8))
        assert False, "expected OcrUnavailableError"
    except extractor_ocr.OcrUnavailableError:
        pass


# ---------------------------------------------------------------------------
# 6b. Direct image uploads (jpg/jpeg/png)
# ---------------------------------------------------------------------------
def test_is_image_path_detects_images():
    assert extractor_ocr.is_image_path("invoice.jpg") is True
    assert extractor_ocr.is_image_path("invoice.JPEG") is True
    assert extractor_ocr.is_image_path("scan.png") is True
    assert extractor_ocr.is_image_path("invoice.pdf") is False
    assert extractor_ocr.is_image_path("invoice.PDF") is False
    assert extractor_ocr.is_image_path("report.xlsx") is False


def test_image_pipeline_maps_to_standard_schema_and_exports():
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow ships with Streamlit
        print("      (skipped: Pillow not installed)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "invoice.png"
        Image.new("RGB", (900, 200), "white").save(image_path)

        engine = _FakeEngine(_fake_lines())
        raw, summary, ocr_text = extractor_ocr.extract_transactions_from_image(
            image_path, engine=engine
        )
        assert engine.calls == 1
        assert "Hurghada" in ocr_text

        formatted = formatter.format_dynamic(raw)
        assert list(formatted.columns) == STANDARD_COLUMNS
        assert len(formatted) == 2
        assert list(formatted["Hotel"]) == ["Hurghada", "Cairo"]

        # End-to-end: main.run routes the image to OCR and exports live formulas.
        out_path = Path(tmp) / "image_out.xlsx"
        out, formatted, summary, supplier_key = main.run(
            image_path, output_path=out_path, engine="auto", ocr_engine=engine
        )
        assert supplier_key == extractor_pdf.DYNAMIC_SUPPLIER_KEY
        assert list(formatted.columns) == STANDARD_COLUMNS
        assert out.exists()

        ws = load_workbook(out)["Invoices"]
        header_row = exporter.HEADER_ROW
        headers = {str(c.value): c.column_letter for c in ws[header_row]}
        assert "EGP Converter" in headers
        n = len(formatted)
        first_data, last_data = header_row + 1, header_row + n
        total_row = last_data + 1
        conv = headers["EGP Converter"]
        # Live per-row converter formula + SUM subtotal are intact.
        assert str(ws[f"{conv}{first_data}"].value).startswith("=")
        assert "$B$2" in ws[f"{conv}{first_data}"].value
        assert ws[f"{conv}{total_row}"].value == (
            f"=SUM({conv}{first_data}:{conv}{last_data})"
        )


# ---------------------------------------------------------------------------
# 6c. Exact import-failure diagnostics (ocr_import_errors)
# ---------------------------------------------------------------------------
def test_ocr_import_errors_captures_exact_exception():
    """A failed import is recorded as "ExcType: message", not swallowed."""
    import sys

    sentinel = object()
    saved = sys.modules.pop("paddle", sentinel)
    # ``None`` in sys.modules makes ``__import__("paddle")`` raise ImportError.
    sys.modules["paddle"] = None
    try:
        ready, missing = extractor_ocr.ocr_dependency_status()
        assert "paddlepaddle" in missing
        assert ready is False

        errors = extractor_ocr.ocr_import_errors()
        detail = errors["paddlepaddle"]
        # Python raises ImportError or its subclass ModuleNotFoundError here;
        # the exact type + message must be preserved verbatim.
        assert detail.startswith(("ImportError", "ModuleNotFoundError")), detail
        assert "paddle" in detail.lower()
    finally:
        if saved is sentinel:
            sys.modules.pop("paddle", None)
        else:
            sys.modules["paddle"] = saved
        # Re-probe so recorded errors match the real environment again.
        extractor_ocr.ocr_dependency_status()


def test_ocr_import_errors_reflect_environment():
    errors = extractor_ocr.ocr_import_errors()
    if extractor_ocr.is_ocr_available():
        assert errors == {}
    else:
        assert errors, "unavailable OCR must expose exact import errors"
        for detail in errors.values():
            exc_name = detail.split(":", 1)[0].strip()
            assert exc_name and ":" in detail, detail


def test_missing_ocr_message_includes_import_details():
    if extractor_ocr.is_ocr_available():
        print("      (skipped: PaddleOCR is installed)")
        return
    message = extractor_ocr._missing_ocr_message()
    # The actionable base message stays intact for existing consumers...
    assert "Text-based PDFs continue to work" in message
    assert "PaddleOCR" in message
    # ...and the exact import failures are appended.
    assert "Import errors:" in message


# ---------------------------------------------------------------------------
# 6d. Process-wide singleton backend (Streamlit rerun safety)
# ---------------------------------------------------------------------------
def test_ocr_backend_created_once_across_reruns():
    """Two engines (== two Streamlit reruns) must share ONE backend instance.

    PaddleX aborts with "PDX has already been initialized" whenever a second
    pipeline is constructed in the same process.
    """
    created = []

    class _FakePaddleOcr:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def predict(self, *args, **kwargs):
            return []

    original_available = extractor_ocr.is_ocr_available
    original_loader = extractor_ocr._load_paddleocr_class
    extractor_ocr.is_ocr_available = lambda: True
    extractor_ocr._load_paddleocr_class = lambda: _FakePaddleOcr
    try:
        extractor_ocr.reset_ocr_backends()

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        first = extractor_ocr.PaddleOcrEngine()
        second = extractor_ocr.PaddleOcrEngine()  # e.g. the next Streamlit rerun
        assert first.recognize(image) == []
        assert second.recognize(image) == []

        assert first._ensure_engine() is second._ensure_engine()
        assert len(created) == 1, f"PaddleOCR must be built once, got {len(created)}"
        assert created[0]["lang"] == extractor_ocr.OCR_LANG
    finally:
        extractor_ocr.is_ocr_available = original_available
        extractor_ocr._load_paddleocr_class = original_loader
        extractor_ocr.reset_ocr_backends()


def test_ocr_backend_cache_is_keyed_by_configuration():
    """Different OCR configurations get their own pipeline; identical ones share."""
    created = []

    class _FakePaddleOcr:
        def __init__(self, **kwargs):
            created.append(kwargs)

    original_available = extractor_ocr.is_ocr_available
    original_loader = extractor_ocr._load_paddleocr_class
    extractor_ocr.is_ocr_available = lambda: True
    extractor_ocr._load_paddleocr_class = lambda: _FakePaddleOcr
    try:
        extractor_ocr.reset_ocr_backends()
        arabic = extractor_ocr.get_ocr_backend("ar", "cpu", True)
        english = extractor_ocr.get_ocr_backend("en", "cpu", True)
        again = extractor_ocr.get_ocr_backend("ar", "cpu", True)
        assert arabic is again
        assert arabic is not english
        assert [call["lang"] for call in created] == ["ar", "en"]
    finally:
        extractor_ocr.is_ocr_available = original_available
        extractor_ocr._load_paddleocr_class = original_loader
        extractor_ocr.reset_ocr_backends()


# ---------------------------------------------------------------------------
# 7. Strict horizontal clustering + header bands (wide borderless tables)
# ---------------------------------------------------------------------------
def test_strict_row_clustering_blocks_vertical_chaining():
    """Tall boxes on a tight pitch must stay one row each.

    The old global tolerance chain-merged such rows (reading whole columns
    vertically into one concatenated cell). The strict overlap rule keeps them
    apart.
    """
    lines = [OcrLine(f"r{i}", (10, 23 * i, 60, 23 * i + 40)) for i in range(6)]
    rows = extractor_ocr.lines_to_rows(lines)
    assert len(rows) == 6
    assert [[line.text for line in row] for row in rows] == [[f"r{i}"] for i in range(6)]


def test_row_clustering_keeps_same_row_boxes_together():
    lines = [
        OcrLine("اسم عربي طويل", (100, 40, 300, 70)),   # tall Arabic box
        OcrLine("2", (10, 48, 50, 64)),                  # short number, same row
        OcrLine("فندق", (100, 73, 300, 103)),            # next row
        OcrLine("2400", (10, 81, 60, 97)),               # next row number
    ]
    rows = extractor_ocr.lines_to_rows(lines)
    assert len(rows) == 2
    assert [line.text for line in rows[0]] == ["2", "اسم عربي طويل"]
    assert [line.text for line in rows[1]] == ["2400", "فندق"]


def test_header_bands_split_wide_borderless_table():
    """Columns with gutters narrower than MIN_GUTTER must not collapse."""
    width = 1386
    header = [OcrLine(f"H{i}", (198 * i, 10, 198 * i + 186, 34)) for i in range(7)]
    data = [
        OcrLine(str(100 + i), (198 * i + 60, 50, 198 * i + 120, 74))
        for i in range(7)
    ]
    lines = header + data
    # The page-wide gutter detector collapses all seven columns (12px gutters
    # < 1.2% of the page) — exactly the bug the header bands fix.
    assert len(extractor_ocr.column_bands(lines, width)) == 1
    table = extractor_ocr.lines_to_table(lines, page_width=width)
    assert len(table) == 2
    assert table[0] == [f"H{i}" for i in range(7)]
    assert table[1] == [str(100 + i) for i in range(7)]


def test_closely_stacked_numbers_are_not_merged_vertically():
    """Closely-stacked numbers must each become their own row.

    Regression: the Hero "Total Invoice" values ("888-", "681-", "1,493-", ...)
    were read vertically into one concatenated cell. Even when the OCR boxes
    overlap vertically, distinct y-centres must stay in distinct rows.
    """
    lines = [
        OcrLine("888-", (10, 100, 60, 122)),
        OcrLine("681-", (10, 116, 60, 138)),
        OcrLine("1,493-", (10, 132, 60, 154)),
        OcrLine("0", (10, 148, 60, 170)),
    ]
    rows = extractor_ocr.lines_to_rows(lines)
    assert len(rows) == 4
    assert [[line.text for line in row] for row in rows] == [
        ["888-"], ["681-"], ["1,493-"], ["0"],
    ]


def test_row_clustering_ignores_tall_arabic_boxes():
    """A tall Arabic cell must not drag two adjacent rows together.

    The old median-anchor overlap rule let a tall Arabic box widen the row's
    anchor box so the next row's numbers overlapped it and merged. The
    centre-gap clustering keeps the two lines apart regardless of box height.
    """
    lines = [
        OcrLine("888-", (10, 100, 60, 114)),
        OcrLine("فندق", (200, 90, 320, 130)),
        OcrLine("شرم", (340, 90, 420, 130)),
        OcrLine("الغردقة", (440, 90, 560, 130)),
        OcrLine("681-", (10, 118, 60, 132)),
        OcrLine("هيلتون", (200, 108, 320, 148)),
        OcrLine("سيناء", (340, 108, 420, 148)),
        OcrLine("الرحلة", (440, 108, 560, 148)),
    ]
    rows = extractor_ocr.lines_to_rows(lines)
    assert len(rows) == 2
    assert [line.text for line in rows[0]] == ["888-", "فندق", "شرم", "الغردقة"]
    assert [line.text for line in rows[1]] == ["681-", "هيلتون", "سيناء", "الرحلة"]


def test_stacked_numbers_reconstruct_to_separate_table_rows():
    """A vertical run of numbers in one column must yield one grid row each."""
    lines = [
        OcrLine("Notes", (20, 40, 120, 62)),
        OcrLine("888-", (20, 100, 120, 122)),
        OcrLine("681-", (20, 120, 120, 142)),
        OcrLine("1,493-", (20, 140, 120, 162)),
        OcrLine("0", (20, 160, 120, 182)),
    ]
    table = extractor_ocr.lines_to_table(lines, page_width=400)
    assert len(table) == 5
    assert [row[0] for row in table] == ["Notes", "888-", "681-", "1,493-", "0"]


# ---------------------------------------------------------------------------
# 8b. Coordinate-template reconstruction (known suppliers)
# ---------------------------------------------------------------------------
def test_template_splits_wide_box_across_columns():
    """A box straddling several columns is sliced at the column boundaries."""
    width = 600
    lines = [
        OcrLine("A", (0, 10, 140, 30)),
        OcrLine("B", (150, 10, 290, 30)),
        OcrLine("C", (300, 10, 440, 30)),
        OcrLine("D", (450, 10, 590, 30)),
        OcrLine("22 33", (150, 50, 440, 70)),  # spans columns B and C
    ]
    table = extractor_ocr.template_lines_to_table(lines, width, ["A", "B", "C", "D"])
    assert table is not None
    assert table[0] == ["A", "B", "C", "D"]
    assert table[1][1] == "22"   # B column
    assert table[1][2] == "33"   # C column


def test_template_splits_multiline_box_into_rows():
    """A multi-line OCR box (newline-separated) is split into its rows."""
    width = 600
    lines = [
        OcrLine("A", (0, 10, 200, 30)),
        OcrLine("B", (300, 10, 500, 30)),
        OcrLine("x\ny\nz", (0, 50, 200, 110)),  # three lines in one tall box
    ]
    table = extractor_ocr.template_lines_to_table(lines, width, ["A", "B"])
    assert table is not None
    assert table[0] == ["A", "B"]
    assert [row[0] for row in table[1:]] == ["x", "y", "z"]


def test_template_bypasses_generic_row_clustering():
    """A known supplier's header anchors columns even when rows overlap."""
    width = 600
    header_names = ["Date", "Hotel", "Total"]
    lines = [
        OcrLine("Date", (0, 10, 190, 30)),
        OcrLine("Hotel", (200, 10, 390, 30)),
        OcrLine("Total", (400, 10, 590, 30)),
        # Two data rows whose boxes overlap vertically (generic clustering
        # would merge them) but which sit in distinct columns.
        OcrLine("1-Sep", (0, 50, 190, 90)),
        OcrLine("Hilton", (200, 60, 390, 100)),
        OcrLine("100", (400, 70, 590, 110)),
        OcrLine("2-Sep", (0, 80, 190, 120)),
        OcrLine("Cairo", (200, 90, 390, 130)),
        OcrLine("200", (400, 100, 590, 140)),
    ]
    table = extractor_ocr.template_lines_to_table(lines, width, header_names)
    assert table is not None
    assert table[0] == ["Date", "Hotel", "Total"]
    # Two rows, each with all three cells populated (no vertical concat).
    assert len(table) == 3
    assert table[1] == ["1-Sep", "Hilton", "100"]
    assert table[2] == ["2-Sep", "Cairo", "200"]


def test_split_discrete_numeric_values():
    """Discrete values separated by spaces or dashes are split; dates/text are preserved."""
    split = extractor_ocr._split_discrete_numeric_values
    assert split("888- 681- 1,493- 991-") == ["888-", "681-", "1,493-", "991-"]
    assert split("888 681 1,493 991") == ["888", "681", "1,493", "991"]
    assert split("888-681-1,493-991-") == ["888-", "681-", "1,493-", "991-"]
    assert split("888-681-1493-991") == ["888", "681", "1493", "991"]
    assert split("888 - 681 - 1,493 - 991") == ["888", "681", "1,493", "991"]
    assert split("-888 -681 -1,493") == ["-888", "-681", "-1,493"]
    assert split("100.50 200.75 300.00") == ["100.50", "200.75", "300.00"]
    assert split("888-") == ["888-"]
    assert split("2026-09-10") == ["2026-09-10"]
    assert split("10/09/2026") == ["10/09/2026"]
    assert split("Hilton - Room 101") == ["Hilton - Room 101"]
    assert split("2609131157") == ["2609131157"]


def test_template_splits_unnewline_numeric_box_into_rows():
    """A bounding box in a numeric column lacking newlines is force-split into separate rows."""
    width = 800
    header_names = ["Date", "Hotel", "Total", "Net"]
    lines = [
        OcrLine("Date", (0, 10, 190, 30)),
        OcrLine("Hotel", (200, 10, 390, 30)),
        OcrLine("Total", (400, 10, 590, 30)),
        OcrLine("Net", (600, 10, 790, 30)),
        # 4 distinct Date boxes
        OcrLine("1-Sep", (0, 50, 190, 80)),
        OcrLine("2-Sep", (0, 90, 190, 120)),
        OcrLine("3-Sep", (0, 130, 190, 160)),
        OcrLine("4-Sep", (0, 170, 190, 200)),
        # 4 distinct Hotel boxes
        OcrLine("Hilton", (200, 50, 390, 80)),
        OcrLine("Cairo", (200, 90, 390, 120)),
        OcrLine("Rixos", (200, 130, 390, 160)),
        OcrLine("Savoy", (200, 170, 390, 200)),
        # Single box for Total lacking newlines (e.g. 888- 681- 1,493- 991-)
        OcrLine("888- 681- 1,493- 991-", (400, 50, 590, 200)),
        # Single box for Net lacking newlines
        OcrLine("888- 681- 1,493- 991-", (600, 50, 790, 200)),
    ]
    table = extractor_ocr.template_lines_to_table(lines, width, header_names)
    assert table is not None
    assert table[0] == ["Date", "Hotel", "Total", "Net"]
    assert len(table) == 5
    assert table[1] == ["1-Sep", "Hilton", "888-", "888-"]
    assert table[2] == ["2-Sep", "Cairo", "681-", "681-"]
    assert table[3] == ["3-Sep", "Rixos", "1,493-", "1,493-"]
    assert table[4] == ["4-Sep", "Savoy", "991-", "991-"]


def test_generic_lines_to_table_splits_unnewline_numeric_box():
    """Generic lines_to_table fallback expands unnewline discrete numeric boxes."""
    lines = [
        OcrLine("Date", (0, 10, 190, 30)),
        OcrLine("Hotel", (200, 10, 390, 30)),
        OcrLine("Total", (400, 10, 590, 30)),
        OcrLine("1-Sep", (0, 50, 190, 80)),
        OcrLine("2-Sep", (0, 90, 190, 120)),
        OcrLine("Hilton", (200, 50, 390, 80)),
        OcrLine("Cairo", (200, 90, 390, 120)),
        OcrLine("888- 681-", (400, 50, 590, 120)),
    ]
    table = extractor_ocr.lines_to_table(lines, 600)
    assert len(table) == 3
    assert table[0] == ["Date", "Hotel", "Total"]
    assert table[1] == ["1-Sep", "Hilton", "888-"]
    assert table[2] == ["2-Sep", "Cairo", "681-"]


# ---------------------------------------------------------------------------
# 8. Hero end-to-end via the OCR pipeline
# ---------------------------------------------------------------------------
HERO_OCR_HEADERS = [
    "Total Invoice", "Total REST", "Rest£", "Rest€", "Rest$", "Rest Egy",
    "Total", "Eng$", "Euro", "Dollar", "Egy LE", "Dive2", "Dive1",
    "intro2", "intro1", "Nationality", "Notes / Details", "Room/Tel",
    "Infant", "Child", "Adults", "Hotel", "Trip", "Date",
]


def _hero_ocr_lines(width=1440):
    """Reconstructed Hero OCR output: metadata strip + header + 2 data rows."""
    pitch = width // len(HERO_OCR_HEADERS)
    lines = []

    def add(text, x, y, w, h):
        lines.append(OcrLine(text, (x, y, x + w, y + h)))

    meta_labels = ["TOTAL", "RECORDS", "PERIOD", "RECIPIENT", "INVOICE DATE", "INVOICE NO"]
    meta_values = ["-10,353 LE", "27 items", "2026-09-10 → 2026-09-13",
                   "Ayman Sharm Gate", "Sep 13, 2026", "2609131157"]
    for i, lab in enumerate(meta_labels):
        add(lab, 20 + i * 120, 10, 100, 20)
    for i, val in enumerate(meta_values):
        add(val, 20 + i * 120, 40, 110, 20)

    for i, head in enumerate(HERO_OCR_HEADERS):
        add(head, i * pitch + 4, 90, pitch - 8, 22)

    row1 = ["915-", "3315", "", "56", "", "2400", "2600", "", "", "3600", "2400",
            "", "", "", "", "ايطاليا", "TG2026-5315", "393895769302+", "0", "1",
            "2", "فندق نعمه ساتر", "الرحلة الرئيسية", "2026-09-10"]
    row2 = ["980-", "4080", "", "", "1495", "5400", "3100", "", "122", "", "3100",
            "", "", "", "", "المانيا", "TG2026-6474", "31692313415+", "1", "0",
            "4", "فرشاز هيلتون", "رحلة غطس", "2026-09-11"]
    for y, row in ((130, row1), (170, row2)):
        for i, val in enumerate(row):
            if val:
                add(val, i * pitch + 6, y, pitch - 12, 22)

    # Footer conversion labels (also used for supplier auto-detection).
    add("T-InEgy", 20, 300, 80, 20)
    add("T-InEng", 120, 300, 80, 20)
    add("T-InEuro", 220, 300, 90, 20)
    return lines


def test_hero_profile_applied_via_ocr_pipeline():
    pdf = _first_pdf()
    if pdf is None:
        print("      (skipped: no sample PDF in the workspace)")
        return
    engine = _FakeEngine(_hero_ocr_lines())

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "hero_out.xlsx"
        out, formatted, summary, supplier_key = main.run(
            pdf, output_path=out_path, engine="ocr", ocr_engine=engine,
            ocr_resolution=72, ocr_max_pages=1,
        )

        # The wide borderless layout was recognised and mapped via its profile.
        assert supplier_key == "hero"
        assert list(formatted.columns) == STANDARD_COLUMNS
        assert len(formatted) == 2
        assert list(formatted["V.NO"]) == ["TG2026-5315", "TG2026-6474"]
        assert formatted["Net"].iloc[0] == -915
        assert formatted["EGP"].iloc[0] == 2400
        assert formatted["USD"].iloc[0] == 3600

        # The invoice number (metadata strip above the header) is never a
        # money value or a data cell.
        assert formatted["EGP Converter"].isna().all()
        assert not (formatted.astype(str) == "2609131157").any().any()

        # The live EGP converter formula survives the OCR path.
        ws = load_workbook(out)["Invoices"]
        headers = {str(c.value): c.column_letter for c in ws[exporter.HEADER_ROW]}
        assert "EGP Converter" in headers
        conv = headers["EGP Converter"]
        assert str(ws[f"{conv}{exporter.HEADER_ROW + 1}"].value).startswith("=")


def test_hero_profile_forced_via_manual_override():
    """Manual supplier_key='hero' forces the Hero profile even when identifiers are absent."""
    pdf = _first_pdf()
    if pdf is None:
        print("      (skipped: no sample PDF in the workspace)")
        return

    # Lines without any Hero identifiers (no 'Hero', 'Ayman', 'T-InEgy')
    lines = [ln for ln in _hero_ocr_lines() if not any(
        ident in ln.text for ident in ("Ayman", "Hero", "T-InEgy", "T-InEng", "T-InEuro")
    )]
    # Auto-detection from this text returns None
    assert formatter.detect_supplier("\n".join(ln.text for ln in lines), formatter.load_rules()) is None

    engine = _FakeEngine(lines)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "hero_forced.xlsx"
        out, formatted, summary, supplier_key = main.run(
            pdf, supplier_key="hero", output_path=out_path, engine="ocr",
            ocr_engine=engine, ocr_resolution=72, ocr_max_pages=1,
        )

        # Forced profile used the 24-column Hero template and mapped to standard columns.
        assert supplier_key == "hero"
        assert list(formatted.columns) == STANDARD_COLUMNS
        assert len(formatted) == 2
        assert list(formatted["V.NO"]) == ["TG2026-5315", "TG2026-6474"]
        assert formatted["Net"].iloc[0] == -915
        assert formatted["EGP"].iloc[0] == 2400
        assert formatted["USD"].iloc[0] == 3600


def test_find_header_row_fuzzy_matches_noisy_1x_headers():
    """_find_header_row matches headers when text is noisy or partial from 1x OCR."""
    header_names = ["Date", "Hotel", "Total", "Adults", "Euro", "Dollar", "Notes / Details"]
    lines = [
        # Slightly noisy or truncated 1x OCR boxes
        OcrLine("Dat", (10, 50, 80, 70)),
        OcrLine("Hotl", (100, 50, 180, 70)),
        OcrLine("Totl", (200, 50, 270, 70)),
        OcrLine("Adult", (290, 50, 360, 70)),
        OcrLine("Eur", (380, 50, 440, 70)),
        OcrLine("Dolar", (460, 50, 530, 70)),
        OcrLine("Notes", (550, 50, 640, 70)),  # token match of "Notes / Details"
        # Data row below
        OcrLine("1-Sep", (10, 100, 80, 120)),
        OcrLine("Hilton", (100, 100, 180, 120)),
        OcrLine("100", (200, 100, 270, 120)),
    ]
    matched = extractor_ocr._find_header_row(lines, header_names)
    assert len(matched) == 7
    texts = {ln.text for ln in matched}
    assert texts == {"Dat", "Hotl", "Totl", "Adult", "Eur", "Dolar", "Notes"}


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL  {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any unexpected error
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(_run_all())
