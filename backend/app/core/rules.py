"""Deterministic rule-based rhythm classifier.

This is a transparent second opinion that runs alongside the neural network.
Every decision is traceable to a named criterion with a numeric threshold, so
when the two disagree a human can see exactly why.

The criteria encode standard bedside ECG interpretation:

    ventricular fibrillation  no organised QRS, no isoelectric baseline
    ventricular premature     a subset of beats is wide and morphologically odd
    atrial flutter            organised atrial activity at 200-380/min
    atrial fibrillation       irregularly irregular RR with no consistent P
    SVT                       regular, narrow, > 150 bpm, P not resolvable
    sinus tachycardia         > 100 bpm with preserved P waves
    sinus bradycardia         < 60 bpm with preserved P waves
    normal sinus rhythm       everything in range

Thresholds live in ``THRESHOLDS`` and are documented with their clinical
provenance. They are calibrated against the synthetic generator, which is
adequate for the plumbing but is *not* a clinical validation -- see docs/METHODOLOGY.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from .signal_engine import Measurements
from .taxonomy import CLASSES, KEYS

# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------

THRESHOLDS = {
    # Rate boundaries: standard clinical definitions.
    "brady_bpm": 60.0,
    "tachy_bpm": 100.0,
    "svt_bpm": 150.0,
    # A QRS wider than 120 ms is "wide" by convention.
    "qrs_wide_ms": 120.0,
    # Fraction of beats that must be wide before ectopy is called.
    "ectopic_fraction": 0.15,
    # Spread between the median and 90th-percentile QRS width. A single beat
    # population has almost no spread; interspersed ectopics create a large one.
    "qrs_spread_ms": 28.0,
    # Coefficient of variation of RR above which a rhythm is "irregular".
    # Sinus rhythm with respiratory variation sits near 0.03-0.06.
    "rr_cv_irregular": 0.13,
    # Beat-template correlation below which morphology is inconsistent.
    "template_inconsistent": 0.92,
    # Raised from 0.45 after measuring digitised input. Following an ink trace
    # column by column smooths a chaotic waveform: ventricular fibrillation
    # measured 0.03 on the source signal but 0.52 after a round trip through an
    # image, because the follower interpolates a continuous path through
    # overlapping ink. Organised rhythms all sit above 0.80, so the gap is wide.
    "template_chaotic": 0.60,
    # Fraction of the strip spent off the isoelectric baseline.
    "occupancy_fibrillatory": 0.45,
    # Autocorrelation strength required to call organised atrial activity.
    "atrial_organised": 0.10,
    # Atrial rate window for flutter.
    "flutter_lo_bpm": 200.0,
    "flutter_hi_bpm": 380.0,
    # P-wave template correlation required to call P waves present.
    "p_consistency": 0.60,
}


@dataclass
class Criterion:
    name: str
    weight: float
    met: bool
    detail: str


@dataclass
class RuleVerdict:
    key: str
    score: float
    criteria: list

    @property
    def evidence(self) -> list:
        return [c.detail for c in self.criteria if c.met]

    @property
    def against(self) -> list:
        return [c.detail for c in self.criteria if not c.met]


def _fmt(v, unit="", nd=0):
    if v is None:
        return "not measurable"
    return f"{v:.{nd}f}{unit}"


def _score(criteria: list) -> float:
    total = sum(c.weight for c in criteria)
    if total <= 0:
        return 0.0
    return sum(c.weight for c in criteria if c.met) / total


# --------------------------------------------------------------------------
# Per-class criteria
# --------------------------------------------------------------------------


def _vfib(m: Measurements) -> list:
    T = THRESHOLDS
    occ = m.baseline_occupancy
    tmpl = m.qrs_template_match
    return [
        Criterion(
            "no_isoelectric_baseline",
            3.0,
            occ is not None and occ > T["occupancy_fibrillatory"],
            f"Baseline occupancy {_fmt(occ, '', 2)} "
            f"(> {T['occupancy_fibrillatory']:.2f} indicates no isoelectric segment)",
        ),
        Criterion(
            "no_organised_qrs",
            3.0,
            tmpl is not None and tmpl < T["template_chaotic"],
            f"Beat-to-beat morphology correlation {_fmt(tmpl, '', 2)} "
            f"(< {T['template_chaotic']:.2f} indicates no reproducible QRS)",
        ),
        Criterion(
            "chaotic_intervals",
            1.0,
            m.rr_cv is not None and m.rr_cv > 0.35,
            f"Interval variation {_fmt(m.rr_cv, '', 2)} is chaotic",
        ),
        Criterion(
            "no_atrial_activity",
            1.0,
            not m.p_wave_present,
            "No organised atrial activity",
        ),
    ]


def _vpb(m: Measurements) -> list:
    T = THRESHOLDS
    wf, tmpl = m.qrs_wide_fraction, m.qrs_template_match
    spread = (
        m.qrs_p90_ms - m.qrs_duration_ms
        if (m.qrs_p90_ms is not None and m.qrs_duration_ms is not None)
        else None
    )
    return [
        # Two beat populations is the defining feature, and it is far more
        # robust than absolute width -- digitising an image inflates every
        # measured QRS a little, but it inflates them all equally.
        Criterion(
            "bimodal_qrs_width",
            3.0,
            spread is not None and spread > T["qrs_spread_ms"],
            f"QRS width spread {_fmt(spread, ' ms', 0)} between median and 90th "
            f"percentile indicates two distinct beat populations",
        ),
        Criterion(
            "subset_of_wide_beats",
            2.5,
            wf is not None and wf > T["ectopic_fraction"],
            f"{_fmt((wf or 0) * 100, '%', 0)} of beats exceed "
            f"{T['qrs_wide_ms']:.0f} ms QRS duration",
        ),
        Criterion(
            "not_all_beats_wide",
            1.5,
            wf is not None and wf < 0.80,
            "Wide complexes are interspersed with normal beats, not sustained",
        ),
        Criterion(
            "mixed_morphology",
            2.0,
            tmpl is not None and T["template_chaotic"] < tmpl < T["template_inconsistent"],
            f"Morphology correlation {_fmt(tmpl, '', 2)} indicates two beat populations",
        ),
        Criterion(
            "organised_underlying_rhythm",
            1.0,
            m.baseline_occupancy is not None
            and m.baseline_occupancy < T["occupancy_fibrillatory"],
            "An organised underlying rhythm is present between ectopics",
        ),
    ]


def _flutter(m: Measurements) -> list:
    T = THRESHOLDS
    org, atr = m.atrial_organisation, m.atrial_rate_bpm
    in_band = atr is not None and T["flutter_lo_bpm"] <= atr <= T["flutter_hi_bpm"]
    return [
        Criterion(
            "organised_atrial_activity",
            3.0,
            org is not None and org > T["atrial_organised"],
            f"Atrial autocorrelation {_fmt(org, '', 2)} "
            f"(> {T['atrial_organised']:.2f} indicates organised flutter waves)",
        ),
        Criterion(
            "atrial_rate_in_flutter_range",
            2.5,
            in_band,
            f"Atrial rate {_fmt(atr, ' /min', 0)} lies in the flutter range "
            f"{T['flutter_lo_bpm']:.0f}-{T['flutter_hi_bpm']:.0f}/min",
        ),
        Criterion(
            "atrial_faster_than_ventricular",
            1.5,
            m.flutter_ratio is not None and m.flutter_ratio >= 1.7,
            f"Atrial:ventricular ratio {_fmt(m.flutter_ratio, ':1', 1)} indicates block",
        ),
        Criterion(
            "narrow_complex",
            1.0,
            m.qrs_duration_ms is not None and m.qrs_duration_ms < T["qrs_wide_ms"],
            f"QRS {_fmt(m.qrs_duration_ms, ' ms', 0)} is narrow, so conduction is supraventricular",
        ),
    ]


def _afib(m: Measurements) -> list:
    T = THRESHOLDS
    return [
        Criterion(
            "irregularly_irregular",
            3.0,
            m.rr_cv is not None and m.rr_cv > T["rr_cv_irregular"],
            f"RR coefficient of variation {_fmt(m.rr_cv, '', 2)} "
            f"(> {T['rr_cv_irregular']:.2f} indicates an irregularly irregular rhythm)",
        ),
        Criterion(
            "no_consistent_p_waves",
            3.0,
            (m.p_wave_consistency or 0.0) < T["p_consistency"],
            f"P-wave template correlation {_fmt(m.p_wave_consistency, '', 2)} "
            f"(< {T['p_consistency']:.2f}: no reproducible atrial depolarisation)",
        ),
        Criterion(
            "narrow_complex",
            1.5,
            m.qrs_wide_fraction is not None and m.qrs_wide_fraction < T["ectopic_fraction"],
            "Complexes remain narrow, so conduction is supraventricular",
        ),
        Criterion(
            "no_organised_flutter",
            1.5,
            (m.atrial_organisation or 0.0) <= T["atrial_organised"],
            "Atrial activity is disorganised rather than a fixed-rate sawtooth",
        ),
        # A strong exclusion, not a nicety: atrial fibrillation requires discrete
        # ventricular complexes. Without them the rhythm is ventricular
        # fibrillation, which is a different clinical emergency entirely.
        Criterion(
            "organised_ventricular_response",
            2.5,
            m.baseline_occupancy is not None
            and m.baseline_occupancy < T["occupancy_fibrillatory"],
            "Discrete QRS complexes are present between beats",
        ),
    ]


def _svt(m: Measurements) -> list:
    T = THRESHOLDS
    hr = m.heart_rate_bpm
    return [
        Criterion(
            "rate_above_svt_threshold",
            3.0,
            hr is not None and hr > T["svt_bpm"],
            f"Ventricular rate {_fmt(hr, ' bpm', 0)} exceeds {T['svt_bpm']:.0f} bpm",
        ),
        Criterion(
            "regular",
            2.5,
            m.rr_cv is not None and m.rr_cv < 0.08,
            f"RR variation {_fmt(m.rr_cv, '', 2)} indicates a regular rhythm",
        ),
        Criterion(
            "narrow_complex",
            2.0,
            m.qrs_duration_ms is not None and m.qrs_duration_ms < T["qrs_wide_ms"],
            f"QRS {_fmt(m.qrs_duration_ms, ' ms', 0)} is narrow",
        ),
        # This is what separates SVT from sinus tachycardia at the same rate.
        Criterion(
            "p_waves_not_resolvable",
            3.0,
            not m.p_wave_present,
            "P waves are buried in the preceding T wave and cannot be resolved",
        ),
        Criterion(
            "no_flutter_waves",
            1.0,
            (m.atrial_organisation or 0.0) <= T["atrial_organised"],
            "No sawtooth flutter waves in the baseline",
        ),
    ]


def _tachycardia(m: Measurements) -> list:
    T = THRESHOLDS
    hr = m.heart_rate_bpm
    return [
        # No upper rate bound. Sinus tachycardia can exceed 150 bpm in young
        # patients; what separates it from SVT is preserved atrial conduction,
        # not the rate itself.
        Criterion(
            "rate_above_100",
            3.0,
            hr is not None and hr > T["tachy_bpm"],
            f"Rate {_fmt(hr, ' bpm', 0)} exceeds {T['tachy_bpm']:.0f} bpm",
        ),
        Criterion(
            "regular",
            2.0,
            m.rr_cv is not None and m.rr_cv < 0.10,
            f"RR variation {_fmt(m.rr_cv, '', 2)} indicates a regular rhythm",
        ),
        Criterion(
            "p_waves_present",
            3.0,
            m.p_wave_present,
            f"P wave precedes each QRS (template correlation "
            f"{_fmt(m.p_wave_consistency, '', 2)}), so the sinus node is pacing",
        ),
        Criterion(
            "narrow_complex",
            1.5,
            m.qrs_duration_ms is not None and m.qrs_duration_ms < T["qrs_wide_ms"],
            f"QRS {_fmt(m.qrs_duration_ms, ' ms', 0)} is narrow",
        ),
        Criterion(
            "no_ectopy",
            1.0,
            (m.qrs_wide_fraction or 0.0) < T["ectopic_fraction"],
            "No ectopic beats",
        ),
    ]


def _bradycardia(m: Measurements) -> list:
    T = THRESHOLDS
    hr = m.heart_rate_bpm
    return [
        Criterion(
            "rate_below_60",
            3.0,
            hr is not None and hr < T["brady_bpm"],
            f"Rate {_fmt(hr, ' bpm', 0)} is below {T['brady_bpm']:.0f} bpm",
        ),
        Criterion(
            "regular",
            2.0,
            m.rr_cv is not None and m.rr_cv < 0.12,
            f"RR variation {_fmt(m.rr_cv, '', 2)} indicates a regular rhythm",
        ),
        Criterion(
            "p_waves_present",
            2.0,
            m.p_wave_present,
            "P wave precedes each QRS, so the sinus node remains the pacemaker",
        ),
        Criterion(
            "narrow_complex",
            1.5,
            m.qrs_duration_ms is not None and m.qrs_duration_ms < T["qrs_wide_ms"],
            f"QRS {_fmt(m.qrs_duration_ms, ' ms', 0)} is narrow",
        ),
        Criterion(
            "prolonged_rr",
            1.0,
            m.rr_mean_ms is not None and m.rr_mean_ms > 1000.0,
            f"Mean RR interval {_fmt(m.rr_mean_ms, ' ms', 0)} is prolonged",
        ),
    ]


def _normal(m: Measurements) -> list:
    T = THRESHOLDS
    hr = m.heart_rate_bpm
    return [
        Criterion(
            "rate_in_range",
            3.0,
            hr is not None and T["brady_bpm"] <= hr <= T["tachy_bpm"],
            f"Rate {_fmt(hr, ' bpm', 0)} is within {T['brady_bpm']:.0f}-{T['tachy_bpm']:.0f} bpm",
        ),
        Criterion(
            "regular",
            2.5,
            m.rr_cv is not None and m.rr_cv < 0.10,
            f"RR variation {_fmt(m.rr_cv, '', 2)} indicates a regular rhythm",
        ),
        Criterion(
            "p_waves_present",
            2.5,
            m.p_wave_present,
            "P wave precedes every QRS",
        ),
        Criterion(
            "narrow_complex",
            2.0,
            m.qrs_duration_ms is not None and m.qrs_duration_ms < T["qrs_wide_ms"],
            f"QRS {_fmt(m.qrs_duration_ms, ' ms', 0)} is within normal limits",
        ),
        Criterion(
            "consistent_morphology",
            1.5,
            m.qrs_template_match is not None
            and m.qrs_template_match >= T["template_inconsistent"],
            f"All beats share one morphology (correlation "
            f"{_fmt(m.qrs_template_match, '', 2)})",
        ),
        Criterion(
            "no_ectopy",
            1.0,
            (m.qrs_wide_fraction or 0.0) < T["ectopic_fraction"],
            "No ectopic or premature beats",
        ),
        # Without this, a 4:1 conducted atrial flutter presents a normal
        # ventricular rate with flutter waves masquerading as P waves, and
        # scores identically to sinus rhythm.
        Criterion(
            "no_organised_atrial_activity",
            2.0,
            (m.atrial_organisation or 0.0) <= T["atrial_organised"],
            "No flutter waves; the baseline between complexes is isoelectric",
        ),
    ]


_CRITERIA_FN = {
    "vfib": _vfib,
    "vpb": _vpb,
    "atrial_flutter": _flutter,
    "afib": _afib,
    "svt": _svt,
    "tachycardia": _tachycardia,
    "bradycardia": _bradycardia,
    "normal": _normal,
}


def classify(m: Measurements) -> dict:
    """Score every class against the measurements.

    Returns a dict with the ranked verdicts, the winning key, a normalised
    confidence and the evidence supporting the top result.
    """
    verdicts = []
    for key in KEYS:
        criteria = _CRITERIA_FN[key](m)
        verdicts.append(RuleVerdict(key=key, score=_score(criteria), criteria=criteria))

    verdicts.sort(key=lambda v: v.score, reverse=True)

    # Normalise into a distribution with a tempered softmax. Temperature is set
    # so that a fully-satisfied rule beating a half-satisfied one yields high
    # but not absolute confidence -- the rule engine should never claim
    # certainty it has not earned.
    temperature = 0.11
    mx = max(v.score for v in verdicts)
    exps = [pow(2.718281828, (v.score - mx) / temperature) for v in verdicts]
    total = sum(exps) or 1.0
    probs = [e / total for e in exps]

    top = verdicts[0]
    runner_up = verdicts[1] if len(verdicts) > 1 else None
    margin = top.score - (runner_up.score if runner_up else 0.0)

    return {
        "key": top.key,
        "confidence": float(probs[0]),
        "raw_score": float(top.score),
        "margin": float(margin),
        "evidence": top.evidence,
        "against": top.against,
        "ranked": [
            {
                "key": v.key,
                "score": float(v.score),
                "probability": float(p),
                "evidence": v.evidence,
            }
            for v, p in zip(verdicts, probs, strict=True)
        ],
        "distribution": {v.key: float(p) for v, p in zip(verdicts, probs, strict=True)},
    }


def distribution_vector(result: dict) -> list:
    """Rule-engine probabilities in canonical class-index order."""
    d = result.get("distribution", {})
    return [float(d.get(c.key, 0.0)) for c in CLASSES]
