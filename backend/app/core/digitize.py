"""ECG image digitisation: photo or scan of a printout -> calibrated signal.

Pipeline
--------
1. deskew        estimate rotation from the grid lines and correct it
2. separate      isolate the ink trace from the printed grid
3. calibrate     recover mm-per-pixel from the grid period
4. extract       column-wise trace following with continuity tracking
5. scale         convert pixels to millivolts and seconds, then resample

Why the grid separation works
-----------------------------
ECG paper is printed with a red/orange grid. Red ink reflects red light, so in
the **red channel** of an RGB image the grid nearly disappears while the black
stylus trace stays dark. That single channel choice removes most of the grid
before any thresholding happens, which is far more robust than trying to
subtract a detected grid pattern afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MM_PER_SEC = 25.0
MM_PER_MV = 10.0


@dataclass
class DigitizeResult:
    signal: np.ndarray           # millivolts
    fs: float                    # samples per second
    px_per_mm: float
    paper_speed_mm_s: float
    gain_mm_mv: float
    rotation_deg: float
    coverage: float              # fraction of columns where a trace was found
    confidence: float            # 0-1 overall reliability of the digitisation
    warnings: list = field(default_factory=list)
    debug: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Step 1: deskew
# --------------------------------------------------------------------------


def estimate_rotation(gray: np.ndarray, max_deg: float = 8.0) -> float:
    """Estimate skew from the dominant near-horizontal grid lines."""
    edges = cv2.Canny(gray, 40, 120, apertureSize=3)
    min_len = max(40, gray.shape[1] // 6)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 720, threshold=90,
        minLineLength=min_len, maxLineGap=8,
    )
    if lines is None:
        return 0.0

    # OpenCV 4 returns shape (N, 1, 4); OpenCV 5 returns (N, 4). Indexing with
    # [:, 0] silently yields a column of scalars on the latter, which then fails
    # to unpack. Reshaping handles both, and any future layout with 4 values per
    # segment.
    segments = np.asarray(lines).reshape(-1, 4)

    angles = []
    for x1, y1, x2, y2 in segments:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if abs(dx) < 1e-6:
            continue
        a = np.degrees(np.arctan2(dy, dx))
        if abs(a) <= max_deg:
            angles.append(a)
    if len(angles) < 4:
        return 0.0
    return float(np.median(angles))


def deskew(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.15:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


# --------------------------------------------------------------------------
# Step 2: trace / grid separation
# --------------------------------------------------------------------------


def trace_mask(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(trace_mask, grid_mask)`` as boolean arrays."""
    b, g, r = cv2.split(bgr.astype(np.float32))

    # Red channel: the red grid is nearly transparent here, the black trace is not.
    red = r
    red_u8 = _to_uint8(red)

    # Black top-hat: closing removes dark structures thinner than the kernel, so
    # (closing - image) isolates the ink regardless of its orientation.
    #
    # An earlier version subtracted a Gaussian-blurred background instead. That
    # suppresses anything extended in one direction -- the flat plateau of the
    # calibration pulse vanished entirely, and its leftover edge was then read
    # as a heartbeat, inflating RR variability enough to push regular rhythms
    # toward atrial fibrillation. The morphological form has no such blind spot.
    h, w = red_u8.shape
    k = max(7, int(min(h, w) / 25) | 1)  # odd, comfortably wider than the trace
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(red_u8, cv2.MORPH_BLACKHAT, kernel)

    # Slight blur first so sensor noise does not fragment the threshold.
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)
    _, th = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = th > 0

    # The grid is where red is bright but green/blue are suppressed.
    grid = ((r - np.minimum(g, b)) > 18) & (r > 90)

    mask = mask & ~grid

    # Clean up: drop specks, bridge single-pixel breaks in the trace.
    m8 = mask.astype(np.uint8)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    # Keep only components large enough to be a trace segment.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n > 1:
        keep = np.zeros(n, dtype=bool)
        min_area = max(12, (bgr.shape[0] * bgr.shape[1]) // 40000)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[i] = True
        m8 = keep[labels].astype(np.uint8)

    return m8 > 0, grid


# --------------------------------------------------------------------------
# Step 3: grid calibration
# --------------------------------------------------------------------------


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Min-max scale to 0-255 without cv2.normalize.

    cv2.normalize(src, None, ...) is one of several OpenCV calls whose loose
    argument forms were tightened in OpenCV 5. NumPy does the same job here in
    two lines and behaves identically across versions.
    """
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros(a.shape, dtype=np.uint8)
    return (((a - lo) / (hi - lo)) * 255.0).astype(np.uint8)


def _subpixel(ac: np.ndarray, k: int) -> float:
    """Refine an autocorrelation peak to sub-pixel precision.

    The grid period is rarely a whole number of pixels -- an image scaled to fit
    a maximum width almost never lands on an integer. Rounding to the nearest
    pixel puts a systematic error straight into the time axis and therefore into
    every rate measurement. Fitting a parabola through the peak and its two
    neighbours removes most of it.
    """
    if k <= 0 or k >= len(ac) - 1:
        return float(k)
    y0, y1, y2 = float(ac[k - 1]), float(ac[k]), float(ac[k + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(k)
    delta = 0.5 * (y0 - y2) / denom
    if abs(delta) > 0.5:
        return float(k)
    return float(k) + delta


def estimate_px_per_mm(grid: np.ndarray, lo: float = 2.0, hi: float = 40.0) -> tuple[float, float]:
    """Recover the grid period in pixels via autocorrelation of its projection.

    Returns ``(px_per_mm, agreement)`` where agreement is the normalised
    autocorrelation strength at the chosen period.
    """

    def period_of(projection: np.ndarray) -> tuple[float, float]:
        p = projection.astype(float)
        p = p - p.mean()
        if np.allclose(p, 0):
            return 0.0, 0.0
        ac = np.correlate(p, p, mode="full")[len(p) - 1 :]
        if ac[0] <= 0:
            return 0.0, 0.0
        ac = ac / ac[0]
        lo_i, hi_i = int(lo), min(int(hi), len(ac) - 1)
        if hi_i - lo_i < 2:
            return 0.0, 0.0
        seg = ac[lo_i:hi_i]
        # First strong local maximum is the 1 mm period.
        peaks = []
        for i in range(1, len(seg) - 1):
            if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.12:
                peaks.append((i + lo_i, seg[i]))
        if not peaks:
            k = int(np.argmax(seg)) + lo_i
            return _subpixel(ac, k), float(ac[k])
        k = int(peaks[0][0])
        return _subpixel(ac, k), float(peaks[0][1])

    px_x, a_x = period_of(grid.sum(axis=0))
    px_y, a_y = period_of(grid.sum(axis=1))

    candidates = [(p, a) for p, a in ((px_x, a_x), (px_y, a_y)) if p >= lo]
    if not candidates:
        return 0.0, 0.0
    if len(candidates) == 2 and abs(candidates[0][0] - candidates[1][0]) <= 1.0:
        return float(np.mean([c[0] for c in candidates])), float(np.mean([c[1] for c in candidates]))
    return max(candidates, key=lambda c: c[1])


# --------------------------------------------------------------------------
# Step 4: trace extraction
# --------------------------------------------------------------------------


def _runs(col: np.ndarray) -> list:
    """Contiguous True runs in a boolean column as (start, end) inclusive."""
    idx = np.flatnonzero(col)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def extract_trace(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Follow the trace column by column. Returns ``(y_per_column, coverage)``.

    Where a column contains several ink runs -- the trace crossing a gridline
    remnant, or two limbs of a steep deflection -- the run whose centre is
    closest to the previous column is chosen, so the follower stays on one
    continuous waveform instead of jumping between segments.
    """
    h, w = mask.shape
    ys = np.full(w, np.nan)
    prev = None

    for x in range(w):
        runs = _runs(mask[:, x])
        if not runs:
            continue
        if prev is None:
            # Start on the longest run available.
            s, e = max(runs, key=lambda r: r[1] - r[0])
        else:
            s, e = min(runs, key=lambda r: abs((r[0] + r[1]) / 2.0 - prev))
        centre = (s + e) / 2.0
        ys[x] = centre
        prev = centre

    coverage = float(np.mean(~np.isnan(ys)))

    # Bridge short gaps; leave long dropouts as NaN so they can be reported.
    valid = np.flatnonzero(~np.isnan(ys))
    if valid.size >= 2:
        ys = np.interp(np.arange(w), valid, ys[valid])
    return ys, coverage


def detect_calibration_pulse(mask: np.ndarray, px_per_mm: float) -> int:
    """Find where the calibration pulse ends, so it is not read as a heartbeat.

    Detected from the ink mask rather than from the followed trace. The trace
    follower can lose the top of the pulse -- a vertical riser is a single tall
    ink run whose centre sits halfway up, and the follower may stay near that
    centre instead of stepping onto the plateau. Working from the mask sidesteps
    that entirely.

    The pulse has an unmistakable signature: two tall vertical ink runs (the
    risers, each about 10 mm) separated by roughly 5 mm of flat plateau, inside
    the left margin. A QRS complex can be equally tall but is never followed by
    a second riser at a fixed 5 mm spacing.

    Returning 0 means no pulse was found and nothing should be trimmed.
    """
    if px_per_mm <= 0:
        return 0
    h, w = mask.shape
    search = int(min(w - 1, 34 * px_per_mm))
    if search < int(6 * px_per_mm):
        return 0

    min_riser = 6.5 * px_per_mm  # a 10 mm pulse, allowing for blur and clipping
    risers = []
    for x in range(search):
        runs = _runs(mask[:, x])
        if not runs:
            continue
        tallest = max(r[1] - r[0] for r in runs)
        if tallest >= min_riser:
            risers.append(x)
    if len(risers) < 2:
        return 0

    # Group adjacent columns into riser bands.
    bands, start, prev = [], risers[0], risers[0]
    for x in risers[1:]:
        if x - prev > max(2, 0.8 * px_per_mm):
            bands.append((start, prev))
            start = x
        prev = x
    bands.append((start, prev))

    if len(bands) < 2:
        return 0

    # The two risers of a 5 mm pulse sit 3-9 mm apart.
    for i in range(len(bands) - 1):
        gap = bands[i + 1][0] - bands[i][1]
        if 3.0 * px_per_mm <= gap <= 9.0 * px_per_mm:
            return min(int(bands[i + 1][1] + 1.5 * px_per_mm), w - 1)
    return 0


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def digitize(
    image: np.ndarray,
    *,
    target_fs: float = 360.0,
    paper_speed: float = MM_PER_SEC,
    gain: float = MM_PER_MV,
    px_per_mm: float | None = None,
    skip_calibration_pulse: bool = True,
) -> DigitizeResult:
    """Digitise an ECG image. ``image`` is BGR uint8 as returned by cv2.imread."""
    warnings: list = []

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Very large photos are downscaled: the trace is only a couple of pixels
    # wide and extra resolution costs time without adding information.
    max_w = 2400
    if image.shape[1] > max_w:
        # Explicit destination size rather than fx/fy with a None size. OpenCV 5
        # rejects the None form with a TypeError, and every strip wider than
        # max_w reaches this line -- which is most of them.
        scale = max_w / image.shape[1]
        new_w = max(1, int(round(image.shape[1] * scale)))
        new_h = max(1, int(round(image.shape[0] * scale)))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = estimate_rotation(gray)
    if abs(angle) > 0.15:
        image = deskew(image, angle)

    mask, grid = trace_mask(image)

    detected_ppm, agreement = estimate_px_per_mm(grid)
    if px_per_mm is None:
        px_per_mm = detected_ppm
        if px_per_mm < 2.0:
            # No usable grid. Fall back on assuming the strip is a standard
            # 10-second recording spanning the full image width.
            px_per_mm = image.shape[1] / (10.0 * paper_speed)
            warnings.append(
                "No grid detected. Scale assumed from a standard 10-second strip; "
                "amplitude values are approximate."
            )
            agreement = 0.0

    ys, coverage = extract_trace(mask)

    # No ink at all, or so little that nothing can be reconstructed. Caught here
    # rather than downstream: without this guard the NaN column array reaches
    # np.histogram, which fails with "autodetected range of [nan, nan] is not
    # finite" -- accurate, but meaningless to someone who just uploaded a photo.
    if coverage < 0.05 or not np.any(np.isfinite(ys)):
        raise ValueError(
            "No ECG trace was found in this image. Check that it shows a printed "
            "ECG with a dark trace on a light background, is in focus, and is not "
            "cropped to a blank region."
        )

    if coverage < 0.35:
        warnings.append(
            f"Trace recovered in only {coverage * 100:.0f}% of columns. "
            "Check image focus, contrast and cropping."
        )

    start = detect_calibration_pulse(mask, px_per_mm) if skip_calibration_pulse else 0
    if start > 0:
        ys = ys[start:]

    if ys.size < 10:
        raise ValueError("No ECG trace could be recovered from this image.")

    # Baseline: the isoelectric line is the most frequently occupied row.
    finite = ys[np.isfinite(ys)]
    if finite.size < 10:
        raise ValueError(
            "The recovered trace is too fragmented to measure. Try a sharper, "
            "higher-contrast image of the printout."
        )
    hist, edges = np.histogram(finite, bins=min(120, max(10, finite.size // 8)))
    baseline = float((edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1]) / 2.0)

    # Image y grows downward; ECG amplitude grows upward.
    mv = (baseline - ys) / (px_per_mm * gain)

    source_fs = px_per_mm * paper_speed  # columns per second
    if source_fs < 180.0:
        warnings.append(
            f"Image resolution gives only {source_fs:.0f} effective samples/second "
            f"({px_per_mm * 25.4:.0f} DPI). QRS duration will read 15-30 ms wider "
            "than it is. Rescan at 300 DPI or higher for reliable width measurement."
        )
    duration = len(mv) / source_fs
    n_out = max(2, int(round(duration * target_fs)))
    resampled = np.interp(
        np.linspace(0.0, duration, n_out),
        np.linspace(0.0, duration, len(mv)),
        mv,
    )

    # Confidence blends grid agreement, trace coverage and plausible amplitude.
    amp = float(np.percentile(np.abs(resampled), 99))
    amp_ok = 1.0 if 0.15 <= amp <= 6.0 else 0.35
    confidence = float(
        np.clip(0.30 * min(agreement / 0.35, 1.0) + 0.45 * coverage + 0.25 * amp_ok, 0.0, 1.0)
    )
    if not (0.15 <= amp <= 6.0):
        warnings.append(
            f"Recovered amplitude ({amp:.2f} mV peak) is outside the physiological "
            "range; the gain calibration may be wrong."
        )

    return DigitizeResult(
        signal=resampled.astype(np.float32),
        fs=float(target_fs),
        px_per_mm=float(px_per_mm),
        paper_speed_mm_s=float(paper_speed),
        gain_mm_mv=float(gain),
        rotation_deg=float(angle),
        coverage=float(coverage),
        confidence=confidence,
        warnings=warnings,
        debug={
            "grid_agreement": float(agreement),
            "source_fs": float(source_fs),
            "duration_s": float(duration),
            "baseline_px": baseline,
            "cal_pulse_end_px": int(start),
            "image_shape": list(image.shape[:2]),
        },
    )


def digitize_bytes(data: bytes, **kw) -> DigitizeResult:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded file as an image.")
    return digitize(img, **kw)
