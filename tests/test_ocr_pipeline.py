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
    assert candidates[0] == {
        "lang": "ar",
        "device": "cpu",
        "use_textline_orientation": True,
        "show_log": False,
    }
    assert {"lang": "ar"} in candidates


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
