"""Tests for the HTTP service.

Kept separate from test_core.py on purpose: ``pytest.importorskip`` at module
scope skips the whole file, so mixing these with the core tests would silently
skip the core suite anywhere fastapi is not installed.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

pytest.importorskip("fastapi", reason="fastapi not installed")

from app.core.render import PaperSpec, render  # noqa: E402
from app.core.taxonomy import KEYS  # noqa: E402

from ml.synth import generate  # noqa: E402

FS = 360.0
SPEC = PaperSpec(px_per_mm=10.0)

@pytest.fixture(scope="module")
def client():
    from app.main import app
    from fastapi.testclient import TestClient

    return TestClient(app)


def test_health_reports_model_state(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["classes"] == 8
    assert "available" in body["model"]


def test_classes_endpoint_matches_taxonomy(client):
    body = client.get("/api/classes").json()
    assert [c["key"] for c in body["classes"]] == list(KEYS)


def test_analyzing_a_sample_returns_a_complete_payload(client):
    res = client.post("/api/analyze/sample", data={"name": "normal"})
    assert res.status_code == 200
    body = res.json()
    assert body["finding"]["key"] == "normal"
    assert body["waveform"]["values"]
    assert body["measurements"]["heart_rate_bpm"] is not None
    assert "not a medical device" in body["disclaimer"].lower()


def test_uploading_a_rendered_image_is_classified(client):
    sig, _ = generate("bradycardia", seed=1, duration=10.0)
    buf = io.BytesIO()
    render(sig, FS, SPEC).save(buf, format="PNG")
    res = client.post(
        "/api/analyze/image",
        files={"file": ("strip.png", buf.getvalue(), "image/png")},
    )
    assert res.status_code == 200
    assert res.json()["finding"]["key"] == "bradycardia"


def test_uploading_a_csv_signal_is_classified(client):
    sig, _ = generate("tachycardia", seed=1, duration=10.0)
    csv = "mv\n" + "\n".join(f"{v:.5f}" for v in sig)
    res = client.post(
        "/api/analyze/signal",
        files={"file": ("strip.csv", csv.encode(), "text/csv")},
        data={"fs": "360"},
    )
    assert res.status_code == 200
    assert res.json()["finding"]["key"] == "tachycardia"


def test_short_recordings_are_rejected_with_a_useful_message(client):
    csv = "\n".join("0.0" for _ in range(100))
    res = client.post(
        "/api/analyze/signal",
        files={"file": ("short.csv", csv.encode(), "text/csv")},
        data={"fs": "360"},
    )
    assert res.status_code == 422
    assert "seconds" in res.json()["detail"].lower()


def test_unreadable_signal_file_is_rejected(client):
    res = client.post(
        "/api/analyze/signal",
        files={"file": ("junk.csv", b"not numbers at all\nnope\n", "text/csv")},
    )
    assert res.status_code == 422


def test_missing_sample_returns_404(client):
    assert client.post("/api/analyze/sample", data={"name": "nope"}).status_code == 404


def test_report_endpoint_returns_a_pdf(client):
    payload = client.post("/api/analyze/sample", data={"name": "afib"}).json()
    res = client.post("/api/report", data={"payload": json.dumps(payload)})
    assert res.status_code == 200
    assert res.content[:4] == b"%PDF"
