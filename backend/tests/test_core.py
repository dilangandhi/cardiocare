"""Tests for the analysis core.

The digitizer round-trip test is the one that matters most: it renders a signal
whose properties are known, reads it back from the image, and checks the
recovered measurements. That is a genuine end-to-end validation of the image
pipeline, unlike the rule-engine tests below, which only confirm that the rules
respond to the signal features they were written for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.core import rules                                        # noqa: E402
from app.core.digitize import digitize, estimate_px_per_mm        # noqa: E402
from app.core.fusion import ArrhythmiaModel, fuse                 # noqa: E402
from app.core.render import PaperSpec, add_photo_realism, render  # noqa: E402
from app.core.signal_engine import analyse, detect_qrs, preprocess  # noqa: E402
from app.core.taxonomy import CLASSES, KEYS, NUM_CLASSES, get     # noqa: E402
from ml.synth import generate                                     # noqa: E402

FS = 360.0
SPEC = PaperSpec(px_per_mm=10.0)


def _bgr(pil_image):
    import cv2

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------------ taxonomy


def test_taxonomy_is_consistent():
    assert NUM_CLASSES == 8
    assert len({c.key for c in CLASSES}) == 8
    assert len({c.index for c in CLASSES}) == 8
    assert [c.index for c in CLASSES] == list(range(8))
    assert list(KEYS) == [c.key for c in CLASSES]


def test_lookup_by_key_code_and_index():
    for c in CLASSES:
        assert get(c.key) is c
        assert get(c.index) is c
        assert get(c.code) is c
    assert get("not_a_class") is None


# ------------------------------------------------------------ signal engine


@pytest.mark.parametrize("label,lo,hi", [
    ("bradycardia", 35, 60),
    ("normal", 60, 100),
    ("tachycardia", 100, 160),
    ("svt", 160, 245),
])
def test_heart_rate_lands_in_the_expected_band(label, lo, hi):
    for seed in range(5):
        sig, _ = generate(label, seed=seed, duration=10.0)
        hr = analyse(sig, FS).heart_rate_bpm
        assert hr is not None
        assert lo <= hr <= hi, f"{label} seed {seed}: {hr:.1f} bpm outside {lo}-{hi}"


def test_normal_qrs_duration_is_physiological():
    """A normal QRS is 80-100 ms. This calibrates the whole width measurement."""
    widths = []
    for seed in range(8):
        sig, _ = generate("normal", seed=seed, duration=10.0)
        widths.append(analyse(sig, FS).qrs_duration_ms)
    assert 85 <= float(np.mean(widths)) <= 120


def test_afib_is_irregular_and_sinus_is_not():
    afib = [analyse(generate("afib", seed=s)[0], FS).rr_cv for s in range(6)]
    sinus = [analyse(generate("normal", seed=s)[0], FS).rr_cv for s in range(6)]
    assert float(np.mean(afib)) > 0.13
    assert float(np.mean(sinus)) < 0.10


def test_p_waves_not_reported_when_the_rate_is_too_fast_to_resolve():
    """At SVT rates the PR window closes; reporting a P wave there would be a
    T wave misread. The detector must decline rather than guess."""
    detected = sum(
        analyse(generate("svt", seed=s)[0], FS).p_wave_present for s in range(8)
    )
    assert detected <= 2


def test_vfib_has_no_isoelectric_baseline():
    vfib = [analyse(generate("vfib", seed=s)[0], FS).baseline_occupancy for s in range(6)]
    organised = [analyse(generate("normal", seed=s)[0], FS).baseline_occupancy for s in range(6)]
    assert min(vfib) > max(organised)


def test_flutter_is_the_only_class_with_organised_atrial_activity():
    flutter = [analyse(generate("atrial_flutter", seed=s)[0], FS).atrial_organisation
               for s in range(10)]
    others = []
    for label in ("normal", "tachycardia", "svt", "afib", "bradycardia"):
        others += [analyse(generate(label, seed=s)[0], FS).atrial_organisation
                   for s in range(6)]
    assert float(np.mean(flutter)) > 0.10
    assert float(np.mean(others)) < 0.05


def test_detector_survives_a_flat_signal():
    m = analyse(np.zeros(int(10 * FS)), FS)
    assert m.beat_count == 0
    assert m.heart_rate_bpm is None
    assert m.signal_quality < 0.2


def test_detector_survives_pure_noise():
    rng = np.random.default_rng(0)
    m = analyse(rng.normal(0, 1, int(10 * FS)), FS)
    assert m.heart_rate_bpm is None or 0 < m.heart_rate_bpm < 400


def test_qrs_detection_is_invariant_to_amplitude_scaling():
    sig, _ = generate("normal", seed=3, duration=10.0)
    a = detect_qrs(preprocess(sig, FS), FS)
    b = detect_qrs(preprocess(sig * 4.0, FS), FS)
    assert abs(len(a) - len(b)) <= 1


# -------------------------------------------------------------- rule engine


def test_rule_engine_recovers_each_class_from_clean_signals():
    """Circular by construction -- the generator and the rules were written
    against the same physiology -- so this is a regression guard, not evidence
    of clinical accuracy. See docs/METHODOLOGY.md."""
    correct = 0
    total = 0
    for label in KEYS:
        for seed in range(10):
            sig, _ = generate(label, seed=3000 + seed, duration=10.0)
            correct += rules.classify(analyse(sig, FS))["key"] == label
            total += 1
    assert correct / total >= 0.90


def test_distribution_is_a_probability_vector():
    sig, _ = generate("afib", seed=1, duration=10.0)
    result = rules.classify(analyse(sig, FS))
    vec = rules.distribution_vector(result)
    assert len(vec) == NUM_CLASSES
    assert all(v >= 0 for v in vec)
    assert abs(sum(vec) - 1.0) < 1e-6


def test_every_class_has_criteria_and_explanations():
    sig, _ = generate("normal", seed=0, duration=10.0)
    result = rules.classify(analyse(sig, FS))
    assert len(result["ranked"]) == NUM_CLASSES
    assert result["evidence"], "the winning class must explain itself"
    for entry in result["ranked"]:
        assert 0.0 <= entry["score"] <= 1.0


# ---------------------------------------------------------------- digitizer


def test_grid_period_is_recovered_to_sub_pixel_precision():
    sig, _ = generate("normal", seed=0, duration=6.0)
    for ppm in (8.0, 10.0, 12.0):
        img = render(sig, FS, PaperSpec(px_per_mm=ppm))
        result = digitize(_bgr(img))
        assert abs(result.px_per_mm - ppm) < 0.6, f"expected {ppm}, got {result.px_per_mm}"


def test_round_trip_preserves_heart_rate():
    """Render a signal, read it back from the image, compare the rate.

    This is the real validation of the image pipeline: nothing about the
    digitizer knows what rate it is supposed to recover."""
    errors = []
    for label in ("normal", "bradycardia", "tachycardia", "afib", "vpb"):
        for seed in range(3):
            sig, _ = generate(label, seed=4000 + seed, duration=10.0)
            truth = analyse(sig, FS).heart_rate_bpm
            result = digitize(_bgr(render(sig, FS, SPEC)))
            recovered = analyse(result.signal, result.fs).heart_rate_bpm
            assert truth and recovered
            errors.append(abs(recovered - truth))
    assert float(np.median(errors)) < 3.0
    assert max(errors) < 12.0


def test_round_trip_survives_a_simulated_photograph():
    """Rotation, uneven lighting, blur and sensor noise must not break it."""
    errors = []
    for seed in range(6):
        sig, _ = generate("normal", seed=5000 + seed, duration=10.0)
        truth = analyse(sig, FS).heart_rate_bpm
        img = add_photo_realism(render(sig, FS, SPEC), seed=seed)
        result = digitize(_bgr(img))
        recovered = analyse(result.signal, result.fs).heart_rate_bpm
        errors.append(abs(recovered - truth))
        assert result.coverage > 0.7
    assert float(np.median(errors)) < 4.0


def test_low_resolution_input_is_flagged_not_silently_wrong():
    sig, _ = generate("normal", seed=0, duration=10.0)
    result = digitize(_bgr(render(sig, FS, PaperSpec(px_per_mm=5.0))))
    assert any("resolution" in w.lower() or "dpi" in w.lower() for w in result.warnings)


def test_rejects_an_image_with_no_trace():
    import cv2

    blank = np.full((300, 900, 3), 250, np.uint8)
    try:
        result = digitize(blank)
        # If it does not raise, it must at least report very low confidence.
        assert result.confidence < 0.6
    except ValueError:
        pass
    del cv2


# ------------------------------------------------------------------- fusion


def test_service_reports_rules_only_when_no_model_is_present():
    model = ArrhythmiaModel(model_dir=Path("/nonexistent"))
    assert model.info.available is False
    sig, _ = generate("normal", seed=0, duration=10.0)
    finding = fuse(analyse(sig, FS), None, model.info)
    assert finding.agreement == "rules_only"
    assert finding.model["available"] is False
    assert finding.cnn_distribution is None


def test_disagreement_between_paths_is_surfaced_not_hidden():
    sig, _ = generate("normal", seed=0, duration=10.0)
    model = ArrhythmiaModel(model_dir=Path("/nonexistent"))
    # Force a CNN vote for ventricular fibrillation against a sinus strip.
    fake = np.zeros(NUM_CLASSES)
    fake[get("vfib").index] = 1.0
    model.info.available = True
    finding = fuse(analyse(sig, FS), fake, model.info)
    assert finding.agreement == "discordant"
    assert finding.inconclusive is True
    assert "review" in finding.agreement_note.lower()


def test_fused_distribution_sums_to_one():
    sig, _ = generate("afib", seed=2, duration=10.0)
    model = ArrhythmiaModel(model_dir=Path("/nonexistent"))
    finding = fuse(analyse(sig, FS), None, model.info)
    assert abs(sum(finding.distribution.values()) - 1.0) < 1e-6


def test_poor_signal_quality_suppresses_confidence():
    sig, _ = generate("normal", seed=0, duration=10.0)
    m = analyse(sig, FS)
    model = ArrhythmiaModel(model_dir=Path("/nonexistent"))
    good = fuse(m, None, model.info, signal_quality=1.0).confidence
    poor = fuse(m, None, model.info, signal_quality=0.1).confidence
    assert poor < good


def test_calibration_pulse_is_trimmed_from_the_signal():
    """The pulse must not survive into the waveform.

    Its trailing edge is a fast, tall deflection and Pan-Tompkins reads it as a
    QRS. One phantom beat at the start of a strip creates one short RR and one
    long RR, which inflated measured RR variability enough to push regular
    rhythms toward an atrial fibrillation call.
    """
    sig, _ = generate("bradycardia", seed=1, duration=10.0)
    img = render(sig, FS, SPEC)  # PaperSpec draws a calibration pulse by default
    result = digitize(_bgr(img))

    assert result.debug["cal_pulse_end_px"] > 0, "calibration pulse was not detected"
    # Trimming the pulse and left margin should land the duration near 10 s.
    assert abs(result.debug["duration_s"] - 10.0) < 0.4

    truth = analyse(sig, FS)
    recovered = analyse(result.signal, result.fs)
    assert recovered.beat_count == truth.beat_count


def test_digitisation_preserves_rr_variability():
    """Regularity must survive the round trip, since it separates AFib from
    sinus rhythm more than any other single measurement."""
    for label in ("normal", "bradycardia", "afib"):
        sig, _ = generate(label, seed=6000, duration=10.0)
        truth = analyse(sig, FS).rr_cv
        recovered = analyse(*(lambda r: (r.signal, r.fs))(digitize(_bgr(render(sig, FS, SPEC))))).rr_cv
        assert abs(recovered - truth) < 0.05, f"{label}: {truth:.3f} -> {recovered:.3f}"


def test_reference_gallery_round_trips_from_its_own_images():
    """Every shipped sample must be recovered from its rendered PNG.

    These are the strips a visitor clicks first, so a regression here is the
    most visible failure the project can have.
    """
    import cv2

    from app.core.fusion import ArrhythmiaModel

    samples = ROOT / "samples"
    if not (samples / "index.json").exists():
        pytest.skip("run `python ml/make_samples.py` first")

    model = ArrhythmiaModel(model_dir=Path("/nonexistent"))
    misses = []
    for key in KEYS:
        png = samples / f"{key}.png"
        if not png.exists():
            pytest.skip(f"missing sample {png}")
        result = digitize(cv2.imread(str(png), cv2.IMREAD_COLOR))
        m = analyse(result.signal, result.fs)
        finding = fuse(m, None, model.info, signal_quality=m.signal_quality)
        if finding.key != key:
            misses.append(f"{key} -> {finding.key}")
        # A reference example should also be reported confidently, not as
        # inconclusive, or the demo shows a shrug on every strip.
        assert not finding.inconclusive, f"{key} came back inconclusive"
    assert not misses, f"gallery regressions: {misses}"


def test_taxonomy_order_differs_from_alphabetical():
    """Guard against silent relabelling at serving time.

    torchvision's ImageFolder indexes classes by sorted folder name. That order
    disagrees with the taxonomy on five of eight classes here, so a model
    trained without remapping emits index 3 meaning "normal" while the service
    reads index 3 as "bradycardia". Training accuracy looks perfect either way,
    because the labels are self-consistent within training -- the error only
    surfaces in production as a disagreement on every strip.

    This test exists so nobody assumes the two orders coincide.
    """
    assert list(KEYS) != sorted(KEYS), (
        "Taxonomy order now matches alphabetical order. The remapping in "
        "ml/train.py is no longer load-bearing, but leave it in place: the "
        "orders would silently diverge again if a class were renamed."
    )
    # The specific collision that caused the bug.
    assert sorted(KEYS).index("normal") != KEYS.index("normal")


def test_non_ecg_images_fail_with_a_readable_message():
    """Uploading something that is not an ECG must explain itself.

    Before this guard, an all-NaN column array reached np.histogram and surfaced
    as "autodetected range of [nan, nan] is not finite" -- accurate, and useless
    to someone who just uploaded the wrong file.
    """
    cases = {
        "blank page": np.full((400, 900, 3), 252, np.uint8),
        "solid black": np.zeros((400, 900, 3), np.uint8),
        "too small": np.full((10, 10, 3), 200, np.uint8),
        "flat dark screenshot": np.full((800, 1400, 3), 30, np.uint8),
    }
    for name, img in cases.items():
        try:
            result = digitize(img)
            assert result.confidence < 0.7, f"{name} reported high confidence"
        except ValueError as exc:
            message = str(exc).lower()
            assert "nan" not in message, f"{name} leaked a NumPy internal: {exc}"
            assert any(w in message for w in ("ecg", "trace", "image")), (
                f"{name} raised an unhelpful message: {exc}"
            )


def test_no_loose_cv2_none_arguments():
    """OpenCV 5 rejects the None-argument shorthands OpenCV 4 accepted.

    cv2.resize(img, None, fx=..., fy=...) and cv2.normalize(src, None, ...) both
    raise TypeError on OpenCV 5. The failure is environment-dependent, so it
    passes every test on a 4.x machine and breaks every upload on a 5.x one --
    which is exactly how it reached a user.

    Parsed with ast rather than matched with a regex, so prose in comments and
    docstrings describing the problem does not itself trip the check.
    """
    import ast

    offenders = []
    for path in (ROOT / "backend" / "app" / "core").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr in ("resize", "normalize")):
                continue
            if not (isinstance(func.value, ast.Name) and func.value.id == "cv2"):
                continue
            for arg in node.args[1:3]:
                if isinstance(arg, ast.Constant) and arg.value is None:
                    offenders.append(f"{path.name}:{node.lineno} cv2.{func.attr}(..., None, ...)")
    assert not offenders, (
        "Loose cv2 None arguments found; pass explicit values instead:\n  "
        + "\n  ".join(offenders)
    )


def test_digitizer_handles_both_houghlinesp_return_shapes():
    """OpenCV 4 returns line segments as (N, 1, 4); OpenCV 5 returns (N, 4).

    Indexing with [:, 0] works on the first and silently yields a column of
    scalars on the second, which then fails to unpack. The failure is invisible
    on a 4.x machine and breaks every image upload on a 5.x one, so it cannot be
    caught by running the suite in one environment -- this test fakes the other.
    """
    import cv2

    from app.core import digitize as dig

    sig, _ = generate("normal", seed=7000, duration=10.0)
    raw = _bgr(render(sig, FS, SPEC))

    original = cv2.HoughLinesP
    results = {}
    try:
        for label, flatten in (("opencv4", False), ("opencv5", True)):
            def patched(*args, _flat=flatten, **kwargs):
                out = original(*args, **kwargs)
                if out is None:
                    return None
                return out.reshape(-1, 4) if _flat else out

            dig.cv2.HoughLinesP = patched
            results[label] = dig.digitize(raw.copy())
    finally:
        dig.cv2.HoughLinesP = original

    a, b = results["opencv4"], results["opencv5"]
    assert abs(a.rotation_deg - b.rotation_deg) < 0.01
    assert abs(a.px_per_mm - b.px_per_mm) < 0.01
    assert abs(a.coverage - b.coverage) < 0.01
