"""debug_hero_ocr.py

Diagnostic: constructor-candidate probe + three-orientation OCR on Hero JPEG.

CRITICAL CONSTRAINT: PaddleX allows only ONE PaddleOCR construction per process,
and even importing `from paddleocr import PaddleOCR` before that construction
triggers partial model loading that causes a deadlock on the real init.
Therefore this script NEVER imports PaddleOCR directly. All OCR goes through
extractor_ocr (the same path the app uses). Post-construction introspection is
done via the already-loaded module in sys.modules.
"""

from __future__ import annotations

import sys
import inspect
from pathlib import Path

import numpy as np
from PIL import Image

import extractor_ocr
import formatter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_image_path() -> Path:
    candidates = [
        Path("WhatsApp Image 2026-09-13 at 19.19.38_ Hero.jpeg"),
        Path.home() / "Downloads" / "WhatsApp Image 2026-09-13 at 19.19.38_ Hero.jpeg",
        Path.home() / "Desktop" / "khaled" / "WhatsApp Image 2026-09-13 at 19.19.38_ Hero.jpeg",
        Path.home() / "Desktop" / "WhatsApp Image 2026-09-13 at 19.19.38_ Hero.jpeg",
    ]
    for c in candidates:
        if c.exists():
            return c
    for root in [Path.cwd(), Path.home() / "Downloads", Path.home() / "Desktop"]:
        if root.exists():
            for match in root.rglob("*Hero*.jp*g"):
                return match
    raise FileNotFoundError(
        "Could not find 'WhatsApp Image 2026-09-13 at 19.19.38_ Hero.jpeg'."
    )


def save_boxes(lines, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# text\tx0\ty0\tx1\ty1\tconfidence\n")
        for line in lines:
            x0, y0, x1, y1 = line.box
            f.write(
                f"{line.text}\t{x0:.1f}\t{y0:.1f}\t{x1:.1f}\t{y1:.1f}"
                f"\t{line.confidence:.4f}\n"
            )


def summarise(label: str, lines, out_path: Path) -> None:
    save_boxes(lines, out_path)
    max_x1 = max((l.box[2] for l in lines), default=0.0)
    max_y1 = max((l.box[3] for l in lines), default=0.0)
    has_date  = any("date"  in (l.text or "").lower() for l in lines)
    has_hotel = any("hotel" in (l.text or "").lower() for l in lines)
    print(f"\n--- {label} ---")
    print(f"  Boxes     : {len(lines)}")
    print(f"  Max x1    : {max_x1:.1f}")
    print(f"  Max y1    : {max_y1:.1f}")
    print(f"  'Date'  box found : {has_date}")
    print(f"  'Hotel' box found : {has_hotel}")
    print(f"  Saved to  : {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    img_path = find_image_path()
    print(f"Loading image from: {img_path}")

    with Image.open(img_path) as img:
        orig_array = extractor_ocr._pil_to_array(img)

    print(f"Original array shape: {orig_array.shape}  (H x W x C)")

    # -----------------------------------------------------------------------
    # (a) Part 1: print all candidate dicts BEFORE any construction.
    #     No paddleocr import — just reads the static list.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("(a) Constructor candidate dicts (no import, no construction)")
    print("=" * 60)
    lang, device, orient = "ar", "cpu", True
    candidates = extractor_ocr.PaddleOcrEngine._constructor_candidates(lang, device, orient)
    for i, kw in enumerate(candidates):
        print(f"  [{i}] {kw}")

    # -----------------------------------------------------------------------
    # Build all three arrays before OCR so preprocess overhead is clear
    # -----------------------------------------------------------------------
    arr_orig = extractor_ocr.preprocess_for_ocr(orig_array)
    arr_cw   = extractor_ocr.preprocess_for_ocr(np.rot90(orig_array, k=-1))
    arr_ccw  = extractor_ocr.preprocess_for_ocr(np.rot90(orig_array, k=1))
    print(f"\nArray shapes after preprocess:")
    print(f"  Original : {arr_orig.shape}")
    print(f"  CW  90   : {arr_cw.shape}   (np.rot90 k=-1)")
    print(f"  CCW 90   : {arr_ccw.shape}  (np.rot90 k=1)")

    # -----------------------------------------------------------------------
    # Initialise ONE engine — triggers the one-and-only PaddleOCR construction
    # -----------------------------------------------------------------------
    print("\nInitialising PaddleOcrEngine(lang='ar') ...")
    engine = extractor_ocr.PaddleOcrEngine(lang="ar")

    # -----------------------------------------------------------------------
    # (b) Run 1: original
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("(b) OCR runs")
    print("=" * 60)
    print("\nRunning OCR on ORIGINAL ...")
    lines_orig = engine.recognize(arr_orig)

    # -----------------------------------------------------------------------
    # (a) Part 2: post-construction introspection — safe now, paddleocr is
    #     already fully initialised in sys.modules.
    # -----------------------------------------------------------------------
    print("\n--- Constructor introspection (post-init) ---")
    poc_mod = sys.modules.get("paddleocr")
    if poc_mod is not None:
        ver = getattr(poc_mod, "__version__", "unknown")
        print(f"  paddleocr version : {ver}")
        PaddleOCR_cls = getattr(poc_mod, "PaddleOCR", None)
        if PaddleOCR_cls is not None:
            try:
                sig = inspect.signature(PaddleOCR_cls.__init__)
                params = set(sig.parameters.keys()) - {"self"}
                has_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                if has_var_kw:
                    print("  PaddleOCR.__init__ accepts **kwargs (all keys pass through)")
                    accepted_note = "accepted (via **kwargs)"
                else:
                    print(f"  PaddleOCR.__init__ params: {sorted(params)}")
                    accepted_note = None
                for flag in ("use_doc_orientation_classify", "use_doc_unwarping"):
                    if has_var_kw:
                        print(f"  '{flag}': {accepted_note}")
                    else:
                        status = "ACCEPTED" if flag in params else "NOT accepted (causes TypeError)"
                        print(f"  '{flag}': {status}")
            except Exception as exc:
                print(f"  Could not inspect signature: {exc}")
    else:
        print("  paddleocr module not found in sys.modules")

    # Report which backend was cached
    key = (engine.lang, engine.device, engine.use_textline_orientation)
    backend = extractor_ocr._OCR_BACKENDS.get(key)
    if backend is not None:
        print(f"  _OCR_BACKENDS key used : {key}")
        print(f"  Backend type           : {type(backend).__module__}.{type(backend).__name__}")
        # Infer which candidate succeeded from presence of doc_ori attributes
        has_doc_orient = hasattr(backend, "use_doc_orientation_classify") or \
                         hasattr(backend, "_use_doc_orientation_classify")
        print(f"  Backend has doc_orientation_classify attr: {has_doc_orient}")
    else:
        all_keys = list(extractor_ocr._OCR_BACKENDS.keys())
        print(f"  Key {key!r} not in _OCR_BACKENDS. All keys: {all_keys}")

    summarise("Original", lines_orig, Path("debug_ocr_hero_orig.txt"))

    # -----------------------------------------------------------------------
    # (b) Run 2 & 3: rotated variants
    # -----------------------------------------------------------------------
    print("\nRunning OCR on CW-rotated (90 clockwise, k=-1) ...")
    lines_cw = engine.recognize(arr_cw)
    summarise("Rotated 90 CW (k=-1)", lines_cw, Path("debug_ocr_hero_cw.txt"))

    print("\nRunning OCR on CCW-rotated (90 counter-clockwise, k=1) ...")
    lines_ccw = engine.recognize(arr_ccw)
    summarise("Rotated 90 CCW (k=1)", lines_ccw, Path("debug_ocr_hero_ccw.txt"))

    # Backwards-compat file
    save_boxes(lines_orig, Path("debug_ocr_hero.txt"))
    print(f"\nAlso saved original run to debug_ocr_hero.txt")
    print("\nDone.")


if __name__ == "__main__":
    main()
