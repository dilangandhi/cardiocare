"""Canonical arrhythmia taxonomy.

This module is the single source of truth for class identity across the whole
project: the training pipeline, the inference service and the frontend all read
their labels from here. Do not hard-code class names anywhere else.

The ``source_codes`` field records how each class is assembled from public
PhysioNet databases so the mapping is auditable. MIT-BIH rhythm annotations are
written with a leading ``(`` (e.g. ``(AFIB``) exactly as they appear in the
``.atr`` files; beat annotations are single characters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ArrhythmiaClass:
    index: int
    key: str
    code: str
    name: str
    short: str
    # Clinical description, shown in the UI report.
    description: str
    # The morphological signature a cardiologist reads off the strip.
    signature: str
    # Triage severity. Drives colour and ordering in the UI.
    severity: str  # one of: normal | monitor | urgent | critical
    # How this class is drawn from public databases.
    source_db: str
    source_codes: tuple = field(default_factory=tuple)


CLASSES: tuple[ArrhythmiaClass, ...] = (
    ArrhythmiaClass(
        index=0,
        key="normal",
        code="N",
        name="Normal sinus rhythm",
        short="Normal",
        description=(
            "Regular rhythm originating from the sinoatrial node at a rate of "
            "60-100 bpm, with a P wave preceding every QRS complex."
        ),
        signature="Regular RR, upright P before each QRS, narrow QRS.",
        severity="normal",
        source_db="mitdb",
        source_codes=("(N",),
    ),
    ArrhythmiaClass(
        index=1,
        key="atrial_flutter",
        code="a",
        name="Atrial flutter",
        short="Flutter",
        description=(
            "A macro-reentrant atrial tachycardia producing organised atrial "
            "activity at roughly 250-350 bpm, usually with regular conduction "
            "to the ventricles at a fixed ratio."
        ),
        signature="Sawtooth flutter waves replacing the isoelectric baseline.",
        severity="urgent",
        source_db="mitdb",
        source_codes=("(AFL",),
    ),
    ArrhythmiaClass(
        index=2,
        key="tachycardia",
        code="b",
        name="Sinus tachycardia",
        short="Tachycardia",
        description=(
            "Sinus rhythm at a rate above 100 bpm. Conduction and morphology "
            "remain normal; the rate itself is the abnormality."
        ),
        signature="Rate > 100 bpm, regular RR, narrow QRS, P waves preserved.",
        severity="monitor",
        source_db="mitdb",
        source_codes=("(SBR-inverse", "(N@>100bpm"),
    ),
    ArrhythmiaClass(
        index=3,
        key="bradycardia",
        code="c",
        name="Sinus bradycardia",
        short="Bradycardia",
        description=(
            "Sinus rhythm at a rate below 60 bpm, producing prolonged RR "
            "intervals with otherwise normal complex morphology."
        ),
        signature="Rate < 60 bpm, regular RR, narrow QRS, P waves preserved.",
        severity="monitor",
        source_db="mitdb",
        source_codes=("(SBR",),
    ),
    ArrhythmiaClass(
        index=4,
        key="svt",
        code="e",
        name="Supraventricular tachycardia",
        short="SVT",
        description=(
            "A rapid, regular narrow-complex tachycardia arising above the "
            "ventricles, typically 150-250 bpm, in which P waves are buried "
            "within or immediately after the QRS."
        ),
        signature="Very fast regular rate, short/narrow QRS, P waves absent, distorted T.",
        severity="urgent",
        source_db="mitdb",
        source_codes=("(SVTA",),
    ),
    ArrhythmiaClass(
        index=5,
        key="afib",
        code="h",
        name="Atrial fibrillation",
        short="AFib",
        description=(
            "Disorganised atrial activity with no discrete P waves and "
            "irregularly irregular ventricular response. The dominant "
            "sustained arrhythmia in clinical practice."
        ),
        signature="No P waves, fibrillatory baseline, irregularly irregular RR.",
        severity="urgent",
        source_db="mitdb",
        source_codes=("(AFIB",),
    ),
    ArrhythmiaClass(
        index=6,
        key="vfib",
        code="i",
        name="Ventricular fibrillation",
        short="VFib",
        description=(
            "Chaotic, disorganised ventricular depolarisation with no "
            "identifiable QRS complexes and no effective cardiac output. A "
            "cardiac arrest rhythm requiring immediate defibrillation."
        ),
        signature="No identifiable QRS, chaotic undulating baseline, no organised rate.",
        severity="critical",
        source_db="mitdb + vfdb",
        source_codes=("(VFL", "(VF"),
    ),
    ArrhythmiaClass(
        index=7,
        key="vpb",
        code="j",
        name="Ventricular premature beat",
        short="VPB",
        description=(
            "An ectopic beat arising below the AV node, occurring earlier than "
            "expected, producing a wide bizarre QRS that is not preceded by a "
            "P wave and is followed by a compensatory pause."
        ),
        signature="Wide premature QRS, absent P, discordant T, compensatory pause.",
        severity="monitor",
        source_db="mitdb",
        source_codes=("V", "(B", "(T"),
    ),
)

NUM_CLASSES = len(CLASSES)

BY_KEY = {c.key: c for c in CLASSES}
BY_CODE = {c.code: c for c in CLASSES}
BY_INDEX = {c.index: c for c in CLASSES}

KEYS: tuple[str, ...] = tuple(c.key for c in CLASSES)
NAMES: tuple[str, ...] = tuple(c.name for c in CLASSES)

SEVERITY_ORDER = {"normal": 0, "monitor": 1, "urgent": 2, "critical": 3}


def get(ref) -> Optional[ArrhythmiaClass]:
    """Look a class up by key, legacy letter code, or integer index."""
    if isinstance(ref, ArrhythmiaClass):
        return ref
    if isinstance(ref, int):
        return BY_INDEX.get(ref)
    if isinstance(ref, str):
        return BY_KEY.get(ref) or BY_CODE.get(ref)
    return None


def to_dict(c: ArrhythmiaClass) -> dict:
    return {
        "index": c.index,
        "key": c.key,
        "code": c.code,
        "name": c.name,
        "short": c.short,
        "description": c.description,
        "signature": c.signature,
        "severity": c.severity,
    }


def catalogue() -> list[dict]:
    return [to_dict(c) for c in CLASSES]
