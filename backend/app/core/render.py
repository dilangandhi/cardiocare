"""Render an ECG signal onto standard calibrated ECG graph paper.

Follows the international recording convention:

    paper speed      25 mm/s   -> 1 small square (1 mm) = 0.04 s
    gain             10 mm/mV  -> 1 small square (1 mm) = 0.1 mV
    heavy gridline   every 5 mm (0.2 s / 0.5 mV)
    calibration pulse 10 mm tall, 5 mm wide, at the left margin

Used in three places: to build the sample gallery, to convert PhysioNet signal
recordings into training images, and to generate known-input images for the
digitizer round-trip test.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

MM_PER_SEC = 25.0
MM_PER_MV = 10.0

# Sampled from a real Schiller/GE ECG printout.
PAPER = (255, 246, 243)
GRID_FINE = (245, 196, 186)
GRID_BOLD = (228, 134, 116)
TRACE = (20, 24, 28)


class PaperSpec:
    """Geometry of one rendered strip."""

    def __init__(self, px_per_mm=6.0, height_mm=40.0, left_margin_mm=12.0, cal_pulse=True):
        self.px_per_mm = float(px_per_mm)
        self.height_mm = float(height_mm)
        self.left_margin_mm = float(left_margin_mm)
        self.cal_pulse = bool(cal_pulse)

    @property
    def px_per_sec(self) -> float:
        return MM_PER_SEC * self.px_per_mm

    @property
    def px_per_mv(self) -> float:
        return MM_PER_MV * self.px_per_mm

    def width_px(self, duration_s: float) -> int:
        total_mm = self.left_margin_mm + duration_s * MM_PER_SEC
        return int(round(total_mm * self.px_per_mm))

    @property
    def height_px(self) -> int:
        return int(round(self.height_mm * self.px_per_mm))


def _draw_grid(draw: ImageDraw.ImageDraw, w: int, h: int, px_per_mm: float) -> None:
    fine = px_per_mm
    bold = px_per_mm * 5.0

    x = 0.0
    while x < w:
        draw.line([(x, 0), (x, h)], fill=GRID_FINE, width=1)
        x += fine
    y = 0.0
    while y < h:
        draw.line([(0, y), (w, y)], fill=GRID_FINE, width=1)
        y += fine

    x = 0.0
    while x < w:
        draw.line([(x, 0), (x, h)], fill=GRID_BOLD, width=1)
        x += bold
    y = 0.0
    while y < h:
        draw.line([(0, y), (w, y)], fill=GRID_BOLD, width=1)
        y += bold


def render(
    signal: np.ndarray,
    fs: float,
    spec: PaperSpec | None = None,
    *,
    trace_width: int = 2,
    clip_mv: float = 2.0,
) -> Image.Image:
    """Render ``signal`` (in mV) to a PIL image of calibrated ECG paper."""
    spec = spec or PaperSpec()
    sig = np.asarray(signal, dtype=float)
    duration = len(sig) / fs

    w, h = spec.width_px(duration), spec.height_px
    img = Image.new("RGB", (w, h), PAPER)
    draw = ImageDraw.Draw(img)
    _draw_grid(draw, w, h, spec.px_per_mm)

    baseline_y = h / 2.0
    left_px = spec.left_margin_mm * spec.px_per_mm

    if spec.cal_pulse:
        # 10 mm tall, 5 mm wide square wave in the left margin.
        amp = 1.0 * spec.px_per_mv
        pw = 5.0 * spec.px_per_mm
        x0 = left_px - pw - 2.0 * spec.px_per_mm
        pts = [
            (x0 - 2.0 * spec.px_per_mm, baseline_y),
            (x0, baseline_y),
            (x0, baseline_y - amp),
            (x0 + pw, baseline_y - amp),
            (x0 + pw, baseline_y),
            (left_px, baseline_y),
        ]
        draw.line(pts, fill=TRACE, width=trace_width, joint="curve")

    sig = np.clip(sig, -clip_mv, clip_mv)
    xs = left_px + (np.arange(len(sig)) / fs) * spec.px_per_sec
    ys = baseline_y - sig * spec.px_per_mv

    inside = xs < w
    xs, ys = xs[inside], ys[inside]
    ys = np.clip(ys, 1, h - 2)

    if len(xs) > 1:
        draw.line(list(zip(xs.tolist(), ys.tolist(), strict=True)), fill=TRACE, width=trace_width, joint="curve")

    return img


def render_to_png(signal: np.ndarray, fs: float, spec: PaperSpec | None = None, **kw) -> bytes:
    buf = io.BytesIO()
    render(signal, fs, spec, **kw).save(buf, format="PNG")
    return buf.getvalue()


def add_photo_realism(img: Image.Image, seed: int = 0) -> Image.Image:
    """Simulate a phone photo of a printout: rotation, uneven lighting, blur, noise.

    Used to generate augmented training images and to test that the digitizer
    survives an imperfect capture rather than only clean renders.
    """
    import cv2

    rng = np.random.default_rng(seed)
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]

    angle = rng.uniform(-2.5, 2.5)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    arr = cv2.warpAffine(
        arr, m, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Smooth multiplicative illumination gradient.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = (
        1.0
        - 0.22 * (xx / w) * rng.uniform(0.3, 1.0)
        - 0.16 * (yy / h) * rng.uniform(0.0, 1.0)
        + 0.10 * rng.uniform(0.0, 1.0)
    )
    arr *= shade[..., None]

    k = int(rng.choice([1, 3, 3, 5]))
    if k > 1:
        arr = cv2.GaussianBlur(arr, (k, k), 0)

    arr += rng.normal(0, rng.uniform(1.5, 5.0), arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
