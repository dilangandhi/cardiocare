"""Evaluate the trained CNN and the rule engine on the same held-out test set.

    python ml/evaluate.py --data data --checkpoint models/resnet50_best.pt

Scoring both paths on identical windows is the point of this script. It answers
three questions the write-up needs:

  1. How well does the CNN generalise to unseen patients?
  2. How well does the deterministic rule engine do on the same data, with no
     training at all? This is the baseline the CNN has to beat to justify itself.
  3. How often do the two agree, and is the fused result better than either?

Outputs a JSON report plus confusion-matrix and ROC figures under docs/figures/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.core import rules                        # noqa: E402
from app.core.signal_engine import analyse        # noqa: E402
from app.core.taxonomy import KEYS, NUM_CLASSES   # noqa: E402


def confusion(true, pred, n=NUM_CLASSES):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred, strict=True):
        cm[t, p] += 1
    return cm


def metrics_from_cm(cm):
    per_class, recalls = {}, []
    for i, key in enumerate(KEYS):
        tp = cm[i, i]
        support = cm[i].sum()
        fp = cm[:, i].sum() - tp
        recall = tp / support if support else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if support:
            recalls.append(recall)
        per_class[key] = {
            "support": int(support), "precision": round(float(precision), 4),
            "recall": round(float(recall), 4), "f1": round(float(f1), 4),
        }
    return {
        "accuracy": round(float(np.trace(cm) / max(cm.sum(), 1)), 4),
        "balanced_accuracy": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_rules(data: Path, fs: float):
    """Run the rule engine over every test window's raw signal."""
    true, pred, probs = [], [], []
    for i, key in enumerate(KEYS):
        for npy in sorted((data / "test" / key).glob("*.npy")):
            sig = np.load(npy)
            result = rules.classify(analyse(sig, fs))
            true.append(i)
            pred.append(KEYS.index(result["key"]))
            probs.append(rules.distribution_vector(result))
    return np.array(true), np.array(pred), np.array(probs)


def evaluate_cnn(data: Path, checkpoint: Path, size: int, batch: int):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from train import build_model

    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model(ckpt.get("arch", "resnet50"), NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(data / "test", transform=tf)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)

    true, probs = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
            probs.append(p)
            true.append(y.numpy())
    probs = np.concatenate(probs)
    true = np.concatenate(true)
    return true, probs.argmax(1), probs


def plot_confusion(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(KEYS)), KEYS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(KEYS)), KEYS, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=11)
    for i in range(len(KEYS)):
        for j in range(len(KEYS)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else "#222")
    fig.colorbar(im, ax=ax, fraction=0.045, label="recall")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_roc(true, probs, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    aucs = {}
    for i, key in enumerate(KEYS):
        y = (true == i).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        fpr, tpr, _ = roc_curve(y, probs[:, i])
        a = auc(fpr, tpr)
        aucs[key] = round(float(a), 4)
        ax.plot(fpr, tpr, lw=1.4, label=f"{key} ({a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return aucs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--fs", type=float, default=360.0)
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()

    data = Path(args.data)
    figs = Path(args.out)
    figs.mkdir(parents=True, exist_ok=True)
    meta_path = data / "dataset.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    report = {"split_by": meta.get("split_by"), "test_records": meta.get("record_assignment", {}).get("test")}

    print("Evaluating the rule engine on the test split...")
    r_true, r_pred, r_probs = evaluate_rules(data, args.fs)
    if len(r_true) == 0:
        print("  no .npy signals found in test/; skipping the rule-engine baseline")
    else:
        cm = confusion(r_true, r_pred)
        report["rules"] = metrics_from_cm(cm)
        report["rules"]["auc"] = plot_roc(r_true, r_probs, "Rule engine — ROC", figs / "roc_rules.png")
        plot_confusion(cm, "Rule engine (no training)", figs / "confusion_rules.png")
        print(f"  accuracy {report['rules']['accuracy']:.4f}   "
              f"balanced {report['rules']['balanced_accuracy']:.4f}")

    if args.checkpoint:
        print("Evaluating the CNN on the test split...")
        c_true, c_pred, c_probs = evaluate_cnn(
            data, Path(args.checkpoint), args.size, args.batch_size)
        cm = confusion(c_true, c_pred)
        report["cnn"] = metrics_from_cm(cm)
        report["cnn"]["auc"] = plot_roc(c_true, c_probs, "CNN — ROC", figs / "roc_cnn.png")
        plot_confusion(cm, "CNN", figs / "confusion_cnn.png")
        print(f"  accuracy {report['cnn']['accuracy']:.4f}   "
              f"balanced {report['cnn']['balanced_accuracy']:.4f}")

        # Agreement between the two independent paths, when window counts line up.
        if len(c_true) == len(r_true) and np.array_equal(c_true, r_true):
            agree = float(np.mean(c_pred == r_pred))
            both_right = float(np.mean((c_pred == c_true) & (r_pred == c_true)))
            fused = (0.6 * c_probs + 0.4 * r_probs).argmax(1)
            fcm = confusion(c_true, fused)
            report["fused"] = metrics_from_cm(fcm)
            report["agreement"] = {
                "paths_agree": round(agree, 4),
                "both_correct": round(both_right, 4),
                "note": ("When the two paths agree the finding is far more likely correct; "
                         "the service reports disagreement rather than hiding it."),
            }
            plot_confusion(fcm, "Fused (60% CNN / 40% rules)", figs / "confusion_fused.png")
            print(f"  paths agree on {agree * 100:.1f}% of windows")
            print(f"  fused balanced accuracy {report['fused']['balanced_accuracy']:.4f}")
        else:
            report["agreement"] = {"note": "Window ordering differed; agreement not computed."}

    Path("docs/evaluation.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport  docs/evaluation.json\nfigures {figs}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
