"""Combine the neural network and the rule engine into one reported finding.

The two paths are deliberately independent. The CNN reads the *image*; the rule
engine reads *measurements* derived from the digitised signal. They can only
agree by both being right about the underlying rhythm, so agreement is real
evidence and disagreement is a real warning.

If no trained model is present the service degrades to rules only and says so.
It never fabricates a model prediction.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import rules
from .signal_engine import Measurements
from .taxonomy import CLASSES, KEYS, NUM_CLASSES, get

MODEL_DIR = Path(os.environ.get("CARDIOCARE_MODEL_DIR", "models"))

# Below this, the finding is reported as inconclusive rather than as a diagnosis.
CONFIDENCE_FLOOR = float(os.environ.get("CARDIOCARE_CONFIDENCE_FLOOR", "0.45"))

# Weight given to the CNN when both paths are available. The rule engine is not
# a tie-breaker -- it is a full participant, because it is the only path whose
# reasoning can be audited.
CNN_WEIGHT = float(os.environ.get("CARDIOCARE_CNN_WEIGHT", "0.60"))


@dataclass
class ModelInfo:
    available: bool
    name: str = "none"
    path: str = ""
    sha256: str = ""
    input_size: int = 224
    note: str = ""
    # Where the weights came from. A model trained on ml/synth.py has learned
    # the generator's assumptions rather than real cardiac electrophysiology,
    # and must never be presented as though it were clinically trained.
    training_data: str = "unknown"


class ArrhythmiaModel:
    """Thin ONNX Runtime wrapper. Absent weights are a supported state."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.session = None
        self.info = ModelInfo(available=False, note="No trained weights found.")
        self._load()

    def _load(self) -> None:
        if not self.model_dir.exists():
            return
        candidates = sorted(self.model_dir.glob("*.onnx"))
        if not candidates:
            return
        path = candidates[0]
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            shape = self.session.get_inputs()[0].shape
            size = shape[-1] if isinstance(shape[-1], int) else 224

            training_data, note = "unknown", ""
            sidecar = path.with_suffix(".json")
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text())
                    training_data = str(meta.get("training_data", "unknown"))
                    size = int(meta.get("input_size", size))
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
            if training_data == "synthetic":
                note = (
                    "These weights were trained on synthetic waveforms, not real "
                    "recordings. Predictions demonstrate the pipeline and carry no "
                    "evidence about clinical accuracy."
                )
            elif training_data == "unknown":
                note = (
                    "Training provenance unknown: no sidecar metadata alongside the "
                    "weights. Re-export with ml/export_onnx.py to record it."
                )

            self.info = ModelInfo(
                available=True,
                name=path.stem,
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()[:16],
                input_size=int(size),
                note=note,
                training_data=training_data,
            )
        except Exception as exc:  # noqa: BLE001
            self.session = None
            self.info = ModelInfo(
                available=False, note=f"Model present but failed to load: {exc}"
            )

    def predict(self, image_rgb: np.ndarray) -> np.ndarray | None:
        """Return a probability vector over the canonical classes, or None."""
        if self.session is None:
            return None
        import cv2

        size = self.info.input_size
        img = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))[None, ...]

        name = self.session.get_inputs()[0].name
        logits = np.asarray(self.session.run(None, {name: x})[0]).reshape(-1)
        if logits.size != NUM_CLASSES:
            return None
        e = np.exp(logits - logits.max())
        return (e / e.sum()).astype(float)


@dataclass
class Finding:
    key: str
    name: str
    severity: str
    confidence: float
    inconclusive: bool
    agreement: str                     # corroborated | discordant | rules_only
    agreement_note: str
    distribution: dict = field(default_factory=dict)
    differential: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    contradicting: list = field(default_factory=list)
    cnn_distribution: dict | None = None
    rule_distribution: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)


def fuse(
    measurements: Measurements,
    cnn_probs: np.ndarray | None,
    model_info: ModelInfo,
    *,
    signal_quality: float = 1.0,
) -> Finding:
    rule_result = rules.classify(measurements)
    rule_vec = np.array(rules.distribution_vector(rule_result), dtype=float)

    if cnn_probs is None:
        combined = rule_vec
        agreement = "rules_only"
        note = (
            "No trained model is loaded, so this finding comes from the "
            "measurement-based rule engine alone."
        )
        cnn_dist = None
    else:
        combined = CNN_WEIGHT * np.asarray(cnn_probs, float) + (1.0 - CNN_WEIGHT) * rule_vec
        cnn_key = KEYS[int(np.argmax(cnn_probs))]
        rule_key = rule_result["key"]
        cnn_dist = {k: float(v) for k, v in zip(KEYS, cnn_probs)}
        if cnn_key == rule_key:
            agreement = "corroborated"
            note = (
                f"The model and the independent measurement rules both indicate "
                f"{get(cnn_key).name.lower()}."
            )
        else:
            agreement = "discordant"
            note = (
                f"The model reports {get(cnn_key).name.lower()} but the measurement "
                f"rules indicate {get(rule_key).name.lower()}. Manual review required."
            )

    total = combined.sum()
    combined = combined / total if total > 0 else np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)

    order = np.argsort(combined)[::-1]
    top = int(order[0])
    cls = CLASSES[top]
    confidence = float(combined[top])

    # Discordance and poor signal quality both suppress confidence.
    if agreement == "discordant":
        confidence *= 0.75
    confidence *= 0.5 + 0.5 * float(np.clip(signal_quality, 0.0, 1.0))

    inconclusive = confidence < CONFIDENCE_FLOOR or agreement == "discordant"

    differential = [
        {
            "key": CLASSES[i].key,
            "name": CLASSES[i].name,
            "short": CLASSES[i].short,
            "severity": CLASSES[i].severity,
            "probability": float(combined[i]),
        }
        for i in order[:4]
    ]

    ranked_by_key = {r["key"]: r for r in rule_result["ranked"]}
    top_rule = ranked_by_key.get(cls.key, {})

    return Finding(
        key=cls.key,
        name=cls.name,
        severity=cls.severity,
        confidence=confidence,
        inconclusive=inconclusive,
        agreement=agreement,
        agreement_note=note,
        distribution={CLASSES[i].key: float(combined[i]) for i in range(NUM_CLASSES)},
        differential=differential,
        evidence=top_rule.get("evidence", []),
        contradicting=(
            rule_result["against"] if cls.key == rule_result["key"] else []
        ),
        cnn_distribution=cnn_dist,
        rule_distribution=rule_result["distribution"],
        model={
            "available": model_info.available,
            "name": model_info.name,
            "sha256": model_info.sha256,
            "note": model_info.note,
            "training_data": model_info.training_data,
            "cnn_weight": CNN_WEIGHT if model_info.available else 0.0,
        },
    )
