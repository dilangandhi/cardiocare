"""CardioCare inference service.

Endpoints
---------
GET  /api/health          service and model status
GET  /api/classes         the 8-class taxonomy
GET  /api/samples         reference strips shipped with the demo
POST /api/analyze/image   upload an ECG image (photo, scan, PNG/JPG)
POST /api/analyze/signal  upload a raw signal (CSV / JSON / one column of mV)
POST /api/analyze/sample  analyse a named sample from the gallery
POST /api/report          render the current finding as a PDF

The service never invents a prediction. If no model is loaded it says so and
reports the rule-engine result alone.
"""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .core import digitize as dig
from .core import fusion, taxonomy
from .core.report import build_report
from .core.signal_engine import analyse

logger = logging.getLogger("cardiocare")

APP_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = APP_ROOT / "frontend"
SAMPLES_DIR = APP_ROOT / "samples"

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
TARGET_FS = 360.0

app = FastAPI(
    title="CardioCare",
    description="ECG arrhythmia analysis. Research use only; not a medical device.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = fusion.ArrhythmiaModel()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _downsample_for_plot(sig: np.ndarray, fs: float, max_points: int = 3000):
    """Thin the trace for transport without losing R-peak amplitude.

    Uniform decimation can land between samples and clip the R peak, which
    would make the rendered strip disagree with the reported measurements. This
    keeps the extreme value within each bucket instead.
    """
    n = len(sig)
    if n <= max_points:
        return sig.astype(float).tolist(), fs
    bucket = int(np.ceil(n / max_points))
    trimmed = sig[: (n // bucket) * bucket].reshape(-1, bucket)
    idx = np.argmax(np.abs(trimmed), axis=1)
    out = trimmed[np.arange(trimmed.shape[0]), idx]
    return out.astype(float).tolist(), fs / bucket


def _analysis_payload(signal: np.ndarray, fs: float, image_rgb=None, source: dict | None = None):
    if len(signal) < int(2 * fs):
        raise HTTPException(
            status_code=422,
            detail="Recording is shorter than 2 seconds. At least 5 seconds is needed; "
                   "10 seconds is recommended.",
        )

    t0 = time.perf_counter()
    measurements = analyse(signal, fs)

    cnn_probs = None
    if image_rgb is not None:
        try:
            cnn_probs = MODEL.predict(image_rgb)
        except Exception:  # noqa: BLE001
            # A failing model must not take down the whole analysis. The rule
            # engine is independent and still has an answer; degrading to it is
            # strictly better than returning nothing.
            logger.exception("Model inference failed; falling back to rules only")
            cnn_probs = None
    finding = fusion.fuse(
        measurements, cnn_probs, MODEL.info, signal_quality=measurements.signal_quality
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    plot, plot_fs = _downsample_for_plot(signal, fs)
    peaks_scaled = [int(p * plot_fs / fs) for p in measurements.r_peaks]

    m = measurements.to_dict()
    m.pop("r_peaks", None)

    return {
        "finding": asdict(finding),
        "measurements": m,
        "waveform": {
            "values": plot,
            "fs": plot_fs,
            "duration_s": len(signal) / fs,
            "r_peaks": peaks_scaled,
        },
        "source": source or {},
        "timing_ms": round(elapsed_ms, 1),
        "disclaimer": "Research use only. Not a medical device. Not for clinical decision-making.",
    }


def _parse_signal_file(raw: bytes, filename: str) -> tuple[np.ndarray, float]:
    """Accept CSV, TSV, JSON or plain newline-delimited millivolt values."""
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    if filename.lower().endswith(".json") or text[0] in "{[":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid JSON: {exc}") from exc
        if isinstance(obj, dict):
            fs = float(obj.get("fs") or obj.get("sampling_rate") or TARGET_FS)
            values = obj.get("signal") or obj.get("values") or obj.get("data")
            if values is None:
                raise HTTPException(
                    status_code=422,
                    detail="JSON must contain a 'signal' array and optionally 'fs'.",
                )
        else:
            fs, values = TARGET_FS, obj
        return np.asarray(values, dtype=float), fs

    rows = [ln for ln in text.splitlines() if ln.strip()]
    # Drop a header row if the first line is not numeric.
    try:
        float(rows[0].replace(",", " ").split()[0])
    except (ValueError, IndexError):
        rows = rows[1:]

    values = []
    for ln in rows:
        parts = ln.replace(",", " ").replace("\t", " ").split()
        if not parts:
            continue
        # Use the last numeric column: "index,value" and bare "value" both work.
        for token in reversed(parts):
            try:
                values.append(float(token))
                break
            except ValueError:
                continue

    if len(values) < 10:
        raise HTTPException(
            status_code=422,
            detail="Could not read numeric samples. Expected one millivolt value per line, "
                   "or a two-column 'time,value' file.",
        )
    return np.asarray(values, dtype=float), TARGET_FS


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": asdict(MODEL.info),
        "classes": taxonomy.NUM_CLASSES,
        "confidence_floor": fusion.CONFIDENCE_FLOOR,
        "target_fs": TARGET_FS,
    }


@app.get("/api/classes")
def classes():
    return {"classes": taxonomy.catalogue()}


@app.get("/api/samples")
def samples():
    index = SAMPLES_DIR / "index.json"
    if not index.exists():
        return {"samples": [], "note": "Run `python ml/make_samples.py` to build the gallery."}
    return json.loads(index.read_text())


@app.get("/api/samples/{name}/image")
def sample_image(name: str):
    path = (SAMPLES_DIR / f"{name}.png").resolve()
    if not path.exists() or SAMPLES_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail="Sample not found.")
    return FileResponse(path, media_type="image/png")


@app.post("/api/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    paper_speed: float = Form(25.0),
    gain: float = Form(10.0),
):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 20 MB limit.")
    if not raw:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    import cv2

    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(
            status_code=422,
            detail="That file could not be decoded as an image. Supported: PNG, JPG, "
                   "BMP, TIFF. For a signal file use the signal upload instead.",
        )

    try:
        result = dig.digitize_bytes(
            raw, target_fs=TARGET_FS, paper_speed=paper_speed, gain=gain
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        # Digitisation touches OpenCV, SciPy and NumPy, any of which can raise
        # something other than ValueError on an image that is not an ECG at all
        # -- a screenshot, a photo of a room, a blank page. Returning a 500 tells
        # the user nothing; this tells them what to try instead.
        logger.exception("Digitisation failed for %s", file.filename)
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not read an ECG trace from this image. It needs a printed "
                "ECG with a visible grid, captured straight-on at 200 DPI or "
                "better; screenshots without a grid and angled photos often fail. "
                f"Internal error: {type(exc).__name__}: {str(exc)[:200]}"
            ),
        ) from exc

    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    payload = _analysis_payload(
        result.signal,
        result.fs,
        image_rgb=rgb,
        source={
            "type": "image",
            "filename": file.filename,
            "digitization": {
                "px_per_mm": round(result.px_per_mm, 2),
                "dpi": round(result.px_per_mm * 25.4, 0),
                "rotation_deg": round(result.rotation_deg, 2),
                "coverage": round(result.coverage, 3),
                "confidence": round(result.confidence, 3),
                "paper_speed_mm_s": result.paper_speed_mm_s,
                "gain_mm_mv": result.gain_mm_mv,
                "warnings": result.warnings,
            },
        },
    )
    return JSONResponse(payload)


@app.post("/api/analyze/signal")
async def analyze_signal(file: UploadFile = File(...), fs: float = Form(0.0)):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 20 MB limit.")
    values, detected_fs = _parse_signal_file(raw, file.filename or "")
    rate = fs if fs and fs > 0 else detected_fs
    return JSONResponse(
        _analysis_payload(
            np.asarray(values, dtype=float),
            float(rate),
            source={"type": "signal", "filename": file.filename, "fs": rate},
        )
    )


@app.post("/api/analyze/sample")
async def analyze_sample(name: str = Form(...)):
    npy = SAMPLES_DIR / f"{name}.npy"
    meta_path = SAMPLES_DIR / f"{name}.json"
    if not npy.exists():
        raise HTTPException(status_code=404, detail="Sample not found.")
    signal = np.load(npy)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    rgb = None
    png = SAMPLES_DIR / f"{name}.png"
    if MODEL.info.available and png.exists():
        import cv2

        arr = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if arr is not None:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    return JSONResponse(
        _analysis_payload(
            signal,
            float(meta.get("fs", TARGET_FS)),
            image_rgb=rgb,
            source={"type": "sample", "name": name, "ground_truth": meta.get("label")},
        )
    )


@app.post("/api/report")
async def report(payload: str = Form(...)):
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid payload: {exc}") from exc
    buf = io.BytesIO()
    build_report(data, buf)
    return Response(
        content=buf.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="cardiocare-report.pdf"'},
    )


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
