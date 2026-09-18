"""Build the training dataset from PhysioNet recordings.

Downloads the MIT-BIH Arrhythmia Database (and the Malignant Ventricular
Ectopy Database for ventricular fibrillation), slices each recording into
fixed-length windows labelled by rhythm annotation, and renders every window
onto calibrated ECG paper so the CNN sees the same kind of image a user will
upload.

    python ml/prepare_data.py --out data/ --window 10 --stride 5

THE SPLIT IS BY RECORD, NOT BY WINDOW
-------------------------------------
Windows cut from the same recording share a patient, a lead placement and a
beat morphology. If they are shuffled and then split, near-duplicate windows
land in both train and test, and the reported accuracy measures memorisation
rather than generalisation. Published ECG results are frequently inflated this
way. Here, every window from a given record goes entirely to one split, so the
test set contains only patients the model has never seen.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.core.render import PaperSpec, add_photo_realism, render  # noqa: E402
from app.core.taxonomy import KEYS  # noqa: E402

TARGET_FS = 360.0
PAPER = PaperSpec(px_per_mm=10.0, height_mm=40.0)

# MIT-BIH rhythm annotations -> our taxonomy. Rhythm annotations are stored in
# the aux_note field and begin with "(".
RHYTHM_MAP = {
    "(N": "normal",
    "(SBR": "bradycardia",
    "(AFIB": "afib",
    "(AFL": "atrial_flutter",
    "(SVTA": "svt",
    "(VFL": "vfib",     # ventricular flutter
    "(VF": "vfib",      # ventricular fibrillation (vfdb, cudb)
    "(B": "vpb",        # ventricular bigeminy
    "(T": "vpb",        # ventricular trigeminy
}

# Deliberately NOT mapped:
#   (VT   ventricular tachycardia -- organised wide complexes, clinically
#         distinct from fibrillation. Folding it into vfib would teach the model
#         that an organised rhythm is a chaotic one. It has no home in this
#         8-class taxonomy, so those windows are dropped rather than mislabelled.
#   (NOD, (P, (PREX, (IVR, (BII, (ASYS, (HGEA, (NOISE -- outside the taxonomy.
EXCLUDED_RHYTHMS = {"(VT", "(NOD", "(P", "(PREX", "(IVR", "(BII",
                    "(ASYS", "(HGEA", "(NOISE", "(PM"}

# Sinus tachycardia has no dedicated MIT-BIH rhythm annotation, so windows
# labelled "(N" whose measured rate exceeds this are relabelled.
TACHY_BPM = 100.0
BRADY_BPM = 60.0

MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

# MIT-BIH Atrial Fibrillation Database: 23 long recordings of AFib and flutter.
AFDB_RECORDS = [
    "04015", "04043", "04048", "04126", "04746", "04908", "04936", "05091",
    "05121", "05261", "06426", "06453", "06995", "07162", "07859", "07879",
    "07910", "08215", "08219", "08378", "08405", "08434", "08455",
]

# MIT-BIH Malignant Ventricular Ectopy Database: ventricular fibrillation.
VFDB_RECORDS = [
    "418", "419", "420", "421", "422", "423", "424", "425", "426", "427",
    "428", "429", "430", "602", "605", "607", "609", "610", "611", "612",
    "614", "615",
]

# Creighton University Ventricular Tachyarrhythmia Database: more VF.
CUDB_RECORDS = [f"cu{i:02d}" for i in range(1, 36)]

# Why more than one database
# -------------------------
# Rare rhythms are concentrated in very few MIT-BIH recordings: "(VFL" and
# "(SVTA" each appear in exactly ONE record (207). With a record-wise split,
# that record lands wholly in train or wholly in test, so one split ends up with
# zero examples of the class -- the model either never learns it or can never be
# scored on it. Drawing ventricular fibrillation from vfdb and cudb, and atrial
# fibrillation and flutter from afdb, spreads every class across enough distinct
# patients that a record-wise split remains possible.
DATABASES = {
    "mitdb": MITDB_RECORDS,
    "afdb": AFDB_RECORDS,
    "vfdb": VFDB_RECORDS,
    "cudb": CUDB_RECORDS,
}


def _require_wfdb():
    try:
        import wfdb  # noqa: F401
    except ImportError:
        sys.exit(
            "The 'wfdb' package is required.\n"
            "  pip install wfdb\n"
            "It downloads recordings directly from PhysioNet."
        )
    import wfdb

    return wfdb


def load_record(wfdb, record: str, db: str = "mitdb"):
    """Fetch one recording and its annotations from PhysioNet."""
    rec = wfdb.rdrecord(record, pn_dir=db)
    ann = wfdb.rdann(record, "atr", pn_dir=db)
    sig = rec.p_signal[:, 0].astype(np.float32)   # lead II where available
    fs = float(rec.fs)
    # cudb has no reliable amplitude calibration; normalise so the renderer does
    # not draw a trace that runs off the top of the paper.
    if db == "cudb":
        scale = float(np.percentile(np.abs(sig - np.median(sig)), 99))
        if scale > 1e-6:
            sig = (sig - np.median(sig)) / scale
    return sig, fs, ann


def rhythm_timeline(ann, n_samples: int) -> np.ndarray:
    """Expand sparse rhythm annotations into a per-sample label array."""
    labels = np.full(n_samples, "", dtype=object)
    current = ""
    events = []
    for i, note in enumerate(getattr(ann, "aux_note", []) or []):
        note = (note or "").strip().rstrip("\x00")
        if note.startswith("("):
            events.append((int(ann.sample[i]), note))
    events.sort()
    ptr = 0
    for start, note in events:
        if ptr < start:
            labels[ptr:start] = current
        current = note
        ptr = start
    labels[ptr:] = current
    return labels


def window_label(seg_labels: np.ndarray) -> str | None:
    """A window is kept only if one rhythm annotation covers at least 90% of it."""
    vals = [v for v in seg_labels if v]
    if len(vals) < 0.9 * len(seg_labels):
        return None
    counts = Counter(vals)
    note, n = counts.most_common(1)[0]
    if n < 0.9 * len(seg_labels):
        return None
    if note in EXCLUDED_RHYTHMS:
        return None
    return RHYTHM_MAP.get(note)


def refine_sinus(label: str, signal: np.ndarray, fs: float) -> str:
    """Split annotated sinus rhythm into normal / tachycardia / bradycardia."""
    if label != "normal":
        return label
    from app.core.signal_engine import analyse

    hr = analyse(signal, fs).heart_rate_bpm
    if hr is None:
        return label
    if hr > TACHY_BPM:
        return "tachycardia"
    if hr < BRADY_BPM:
        return "bradycardia"
    return "normal"


def resample(sig: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-6:
        return sig
    n_out = int(round(len(sig) * fs_out / fs_in))
    return np.interp(
        np.linspace(0.0, 1.0, n_out), np.linspace(0.0, 1.0, len(sig)), sig
    ).astype(np.float32)


def split_records(db_records: dict, seed: int = 17, ratios=(0.70, 0.15, 0.15)) -> dict:
    """Assign whole records to train / val / test.

    Each database is split separately and the slices merged. Splitting the
    combined pool at random would let an entire database land in one split --
    with ventricular fibrillation drawn only from vfdb and cudb, that would put
    every VFib example on one side of the divide and none on the other.

    Records are never divided. Every window from a recording goes to exactly one
    split, so test performance reflects patients the model has never seen.
    """
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    for db, records in sorted(db_records.items()):
        shuffled = list(records)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_tr = max(1, int(round(n * ratios[0])))
        n_va = max(1, int(round(n * ratios[1]))) if n >= 3 else 0
        out["train"] += [f"{db}/{r}" for r in shuffled[:n_tr]]
        out["val"] += [f"{db}/{r}" for r in shuffled[n_tr : n_tr + n_va]]
        out["test"] += [f"{db}/{r}" for r in shuffled[n_tr + n_va :]]
    return {k: sorted(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--window", type=float, default=10.0, help="window length in seconds")
    ap.add_argument("--stride", type=float, default=5.0, help="stride in seconds")
    ap.add_argument("--databases", nargs="*", default=None,
                    help="which PhysioNet databases to use (default: all four)")
    ap.add_argument("--records", nargs="*", default=None,
                    help="subset of records as db/record, e.g. mitdb/207 vfdb/418")
    ap.add_argument("--max-per-class", type=int, default=4000)
    ap.add_argument("--augment", type=int, default=1,
                    help="extra photo-realistic copies per training window")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    wfdb = _require_wfdb()
    out = Path(args.out)

    if args.records:
        db_records = defaultdict(list)
        for item in args.records:
            db, _, rec = item.partition("/")
            db_records[db or "mitdb"].append(rec or db)
        db_records = dict(db_records)
    else:
        names = args.databases or list(DATABASES)
        db_records = {k: v for k, v in DATABASES.items() if k in names}
    assignment = split_records(db_records, seed=args.seed)
    print("databases: " + ", ".join(f"{k} ({len(v)} records)" for k, v in db_records.items()))

    for split in assignment:
        for key in KEYS:
            (out / split / key).mkdir(parents=True, exist_ok=True)

    manifest, counts = [], defaultdict(Counter)
    win = int(args.window * TARGET_FS)
    stride = int(args.stride * TARGET_FS)

    for split, recs in assignment.items():
        for qualified in recs:
            db, _, record = qualified.partition("/")
            try:
                sig, fs, ann = load_record(wfdb, record, db=db)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {qualified}: {exc}")
                continue

            timeline = rhythm_timeline(ann, len(sig))
            sig = resample(sig, fs, TARGET_FS)
            scale = TARGET_FS / fs
            tl = timeline[np.clip((np.arange(len(sig)) / scale).astype(int), 0, len(timeline) - 1)]

            kept = 0
            for start in range(0, len(sig) - win, stride):
                seg = sig[start : start + win]
                label = window_label(tl[start : start + win])
                if label is None:
                    continue
                label = refine_sinus(label, seg, TARGET_FS)
                if label not in KEYS:
                    continue
                if counts[split][label] >= args.max_per_class:
                    continue

                stem = f"{db}-{record}_{start}"
                dest = out / split / label
                img = render(seg, TARGET_FS, PAPER)
                img.save(dest / f"{stem}.png", optimize=True)
                np.save(dest / f"{stem}.npy", seg.astype(np.float32))
                counts[split][label] += 1
                kept += 1
                manifest.append(
                    {"split": split, "record": qualified, "label": label,
                     "start": start, "file": f"{split}/{label}/{stem}.png"}
                )

                # Augmented copies teach the model to survive phone photos.
                if split == "train":
                    for a in range(args.augment):
                        aug = add_photo_realism(img, seed=hash((qualified, start, a)) % 100_000)
                        aug.save(dest / f"{stem}_aug{a}.png", optimize=True)
                        manifest.append(
                            {"split": split, "record": qualified, "label": label,
                             "start": start, "file": f"{split}/{label}/{stem}_aug{a}.png",
                             "augmented": True}
                        )
            print(f"  {split:5s} {qualified}: {kept} windows")

    meta = {
        "target_fs": TARGET_FS,
        "window_s": args.window,
        "stride_s": args.stride,
        "px_per_mm": PAPER.px_per_mm,
        "classes": list(KEYS),
        "split_by": "record",
        "record_assignment": assignment,
        "counts": {s: dict(c) for s, c in counts.items()},
        "note": (
            "Records are assigned wholly to one split. No recording contributes "
            "windows to more than one split, so test performance reflects unseen "
            "patients."
        ),
    }
    (out / "dataset.json").write_text(json.dumps(meta, indent=2))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nWindows per class:")
    header = "  " + " " * 18 + "".join(f"{s:>9}" for s in ("train", "val", "test"))
    print(header)
    empty = []
    for key in KEYS:
        row = [counts[s][key] for s in ("train", "val", "test")]
        print(f"  {key:18s}" + "".join(f"{v:9d}" for v in row))
        for split, v in zip(("train", "val", "test"), row, strict=True):
            if v == 0:
                empty.append(f"{key}/{split}")
    print("  " + "-" * 45)
    print("  " + f"{'total':18s}" + "".join(
        f"{sum(counts[s].values()):9d}" for s in ("train", "val", "test")))

    if empty:
        print("\n  WARNING: no windows for " + ", ".join(empty))
        print("  Those classes cannot be learned or scored. Add more databases,")
        print("  reduce --stride, or re-run with a different --seed.")

    print(f"\nWrote dataset to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
