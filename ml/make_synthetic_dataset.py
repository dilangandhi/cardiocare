"""Build a training dataset from the synthetic generator instead of PhysioNet.

    python ml/make_synthetic_dataset.py --out data_synth --per-class 500

WHAT THIS IS FOR
----------------
Proving the training pipeline end to end -- dataset, training loop, ONNX export,
and the dual-path interface -- without waiting on a 45-90 minute PhysioNet
download. It produces a genuinely trained model in well under an hour on a
laptop.

WHAT THIS IS NOT FOR
--------------------
Any claim about accuracy. A model trained here learns the quirks of
``ml/synth.py``: Gaussian-shaped waves, one morphology per class, noise drawn
from one distribution. Real ECGs have none of those regularities. Expect this
model to score near 100% on its own test split and to perform badly on real
recordings. Both facts are uninformative about clinical performance.

The exported model is tagged ``training_data: synthetic`` so the service can
display that provenance rather than letting a synthetic model masquerade as a
clinically trained one.

Splitting
---------
Each split draws from a disjoint seed range, so no waveform appears in more than
one split. This is the synthetic analogue of the record-wise split used for real
data -- weaker, because all windows come from the same generator, but it does at
least prevent identical strips landing on both sides.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.core.render import PaperSpec, add_photo_realism, render  # noqa: E402
from app.core.taxonomy import KEYS  # noqa: E402

from ml.synth import generate  # noqa: E402

FS = 360.0
SPEC = PaperSpec(px_per_mm=10.0, height_mm=40.0)

# Disjoint seed ranges keep the splits from sharing waveforms.
SEED_BASE = {"train": 0, "val": 2_000_000, "test": 4_000_000}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--per-class", type=int, default=500,
                    help="training windows per class")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--augment", type=int, default=1,
                    help="extra photo-realistic copies per training window")
    args = ap.parse_args()

    out = Path(args.out)
    ratios = {"train": 1.0, "val": 0.2, "test": 0.2}
    counts = {}

    for split, ratio in ratios.items():
        n = max(1, int(args.per_class * ratio))
        counts[split] = {}
        for cls in KEYS:
            dest = out / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            made = 0
            for i in range(n):
                seed = SEED_BASE[split] + abs(hash(cls)) % 1000 * 100_000 + i
                sig, _ = generate(cls, duration=args.duration, fs=FS, seed=seed)
                img = render(sig, FS, SPEC)
                stem = f"{cls}_{split}_{i:05d}"
                img.save(dest / f"{stem}.png", optimize=True)
                np.save(dest / f"{stem}.npy", sig.astype(np.float32))
                made += 1
                if split == "train":
                    for a in range(args.augment):
                        aug = add_photo_realism(img, seed=seed * 7 + a)
                        aug.save(dest / f"{stem}_aug{a}.png", optimize=True)
                        made += 1
            counts[split][cls] = made
            print(f"  {split:5s} {cls:16s} {made}")

    (out / "dataset.json").write_text(json.dumps({
        "target_fs": FS,
        "window_s": args.duration,
        "px_per_mm": SPEC.px_per_mm,
        "classes": list(KEYS),
        "split_by": "seed_range",
        "source": "synthetic",
        "counts": counts,
        "warning": (
            "SYNTHETIC DATA. A model trained here learns the generator's "
            "assumptions, not real cardiac electrophysiology. Test accuracy on "
            "this dataset says nothing about clinical performance. Use "
            "ml/prepare_data.py with PhysioNet recordings for any real claim."
        ),
    }, indent=2))

    total = sum(sum(c.values()) for c in counts.values())
    print(f"\nWrote {total} images to {out}")
    print("\nNOTE: this dataset is synthetic. A model trained on it will score")
    print("near 100% on its own test split and poorly on real ECGs. It proves")
    print("the pipeline works; it is not evidence of accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
