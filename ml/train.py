"""Train a CNN to classify ECG strip images into the 8 arrhythmia classes.

    python ml/train.py --data data --arch resnet50 --epochs 30

Four architectures are supported so the comparison table in the write-up is
reproducible: resnet50, efficientnet_b0, mobilenet_v2, densenet121.

Design decisions worth knowing
------------------------------
* Transfer learning from ImageNet. Ten thousand ECG windows is far too few to
  train a ResNet-50 from scratch; the early layers transfer as edge and texture
  detectors regardless of domain.
* No horizontal flips. An ECG read backwards is not an ECG -- P before QRS
  before T is the whole diagnosis. Only mild rotation, brightness, contrast and
  noise are applied, matching the ways a real photo varies.
* Class-weighted loss. Ventricular fibrillation windows are two orders of
  magnitude rarer than sinus rhythm, and unweighted training simply predicts
  the majority class.
* Balanced accuracy and per-class recall are the headline metrics. Plain
  accuracy on an imbalanced set is close to meaningless.
* The best checkpoint is chosen on validation balanced accuracy, and the test
  split is touched exactly once, at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, models, transforms
except ImportError:
    sys.exit(
        "PyTorch and torchvision are required for training.\n"
        "  pip install torch torchvision\n"
        "Or open notebooks/train_colab.ipynb to train on a free Colab GPU."
    )

from app.core.taxonomy import KEYS, NUM_CLASSES  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(arch: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
    weights = "DEFAULT" if pretrained else None
    if arch == "resnet50":
        m = models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "densenet121":
        m = models.densenet121(weights=weights)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    else:
        raise ValueError(f"unknown architecture {arch!r}")
    return m


def remap_to_taxonomy(ds, name: str):
    """Force an ImageFolder to use taxonomy class indices.

    ImageFolder assigns indices by sorting folder names alphabetically, which
    for this project gives afib=0, atrial_flutter=1, bradycardia=2, normal=3...
    The taxonomy order is normal=0, atrial_flutter=1, tachycardia=2,
    bradycardia=3... The two disagree on five of eight classes.

    Left unremapped, the model trains perfectly well and its accuracy looks
    fine, because the labels are self-consistent within training. The damage
    only appears at serving time: the service maps output index 3 to
    bradycardia while the model means normal. Every prediction is silently
    relabelled, and the dual-path check reports a disagreement on every strip.
    """
    missing = [c for c in ds.classes if c not in KEYS]
    if missing:
        raise ValueError(
            f"{name}: folder(s) {missing} are not in the taxonomy. "
            f"Expected only: {list(KEYS)}"
        )
    lookup = {c: KEYS.index(c) for c in ds.classes}
    ds.samples = [(path, lookup[ds.classes[idx]]) for path, idx in ds.samples]
    ds.imgs = ds.samples
    ds.targets = [t for _, t in ds.samples]
    ds.class_to_idx = dict(lookup)
    ds.classes = list(KEYS)
    return ds


def build_transforms(size: int):
    train = transforms.Compose([
        transforms.Resize((size, size)),
        # No horizontal flip: ECG morphology is directional.
        transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.97, 1.03)),
        transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.20, scale=(0.01, 0.04)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, eval_tf


def balanced_accuracy(cm: np.ndarray) -> float:
    recalls = []
    for i in range(cm.shape[0]):
        total = cm[i].sum()
        if total > 0:
            recalls.append(cm[i, i] / total)
    return float(np.mean(recalls)) if recalls else 0.0


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int = NUM_CLASSES):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    probs_all, labels_all, loss_sum, n = [], [], 0.0, 0
    lossf = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(lossf(logits, y)) * y.size(0)
        n += y.size(0)
        p = torch.softmax(logits, dim=1)
        pred = p.argmax(1)
        for t, q in zip(y.cpu().numpy(), pred.cpu().numpy(), strict=True):
            cm[t, q] += 1
        probs_all.append(p.cpu().numpy())
        labels_all.append(y.cpu().numpy())
    return {
        "loss": loss_sum / max(n, 1),
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "balanced_accuracy": balanced_accuracy(cm),
        "confusion_matrix": cm.tolist(),
        "probs": np.concatenate(probs_all) if probs_all else np.empty((0, num_classes)),
        "labels": np.concatenate(labels_all) if labels_all else np.empty((0,), dtype=int),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--arch", default="resnet50",
                    choices=["resnet50", "efficientnet_b0", "mobilenet_v2", "densenet121"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = Path(args.data)
    meta_path = data / "dataset.json"
    if not meta_path.exists():
        sys.exit(f"No dataset.json in {data}. Run ml/prepare_data.py first.")
    meta = json.loads(meta_path.read_text())
    if meta.get("split_by") != "record":
        print("WARNING: dataset was not split by record. Test metrics will be inflated.")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}   arch: {args.arch}")

    train_tf, eval_tf = build_transforms(args.size)
    ds_train = remap_to_taxonomy(datasets.ImageFolder(data / "train", transform=train_tf), "train")
    ds_val = remap_to_taxonomy(datasets.ImageFolder(data / "val", transform=eval_tf), "val")
    ds_test = remap_to_taxonomy(datasets.ImageFolder(data / "test", transform=eval_tf), "test")

    assert list(ds_train.classes) == list(KEYS), "class remapping failed"
    print(f"classes (taxonomy order): {list(KEYS)}")

    # Oversample rare classes rather than letting the majority class dominate.
    targets = np.array(ds_train.targets)
    class_counts = np.bincount(targets, minlength=NUM_CLASSES).astype(float)
    class_counts[class_counts == 0] = 1.0
    sample_w = (1.0 / class_counts)[targets]
    sampler = WeightedRandomSampler(sample_w.tolist(), len(sample_w), replacement=True)

    dl = dict(num_workers=args.workers, pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler, **dl)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, **dl)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, **dl)

    print(f"train {len(ds_train)}  val {len(ds_val)}  test {len(ds_test)}")

    model = build_model(args.arch).to(device)
    # Loss weighting on top of sampling handles residual imbalance.
    weights = torch.tensor(
        (class_counts.sum() / (NUM_CLASSES * class_counts)), dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{args.arch}_best.pt"

    best, since_best, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            running += float(loss) * y.size(0)
            seen += y.size(0)
        scheduler.step()

        val = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_balanced_accuracy": val["balanced_accuracy"],
        })
        print(
            f"epoch {epoch:3d}/{args.epochs}  train {running / max(seen, 1):.4f}  "
            f"val {val['loss']:.4f}  acc {val['accuracy']:.4f}  "
            f"bal-acc {val['balanced_accuracy']:.4f}  ({time.time() - t0:.0f}s)"
        )

        if val["balanced_accuracy"] > best:
            best, since_best = val["balanced_accuracy"], 0
            torch.save(
                {"arch": args.arch, "state_dict": model.state_dict(),
                 "classes": list(KEYS), "size": args.size,
                 # Records that outputs are already in taxonomy order, so the
                 # exporter knows no permutation is needed.
                 "class_order": "taxonomy",
                 "val_balanced_accuracy": best},
                ckpt,
            )
        else:
            since_best += 1
            if since_best >= args.patience:
                print(f"early stop: no improvement for {args.patience} epochs")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    test = evaluate(model, test_loader, device)

    cm = np.array(test["confusion_matrix"])
    per_class = {}
    for i, key in enumerate(KEYS):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[key] = {
            "support": int(cm[i].sum()),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        }

    report = {
        "arch": args.arch,
        "epochs_run": len(history),
        "val_balanced_accuracy": round(best, 4),
        "test_accuracy": round(test["accuracy"], 4),
        "test_balanced_accuracy": round(test["balanced_accuracy"], 4),
        "confusion_matrix": test["confusion_matrix"],
        "classes": list(KEYS),
        "per_class": per_class,
        "history": history,
        "split_by": meta.get("split_by"),
        "test_records": meta.get("record_assignment", {}).get("test", []),
        "args": vars(args),
    }
    (out / f"{args.arch}_report.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 62)
    print(f"test accuracy           {test['accuracy']:.4f}")
    print(f"test balanced accuracy  {test['balanced_accuracy']:.4f}   <- headline metric")
    print("=" * 62)
    print(f"{'class':18s}{'support':>9}{'precision':>11}{'recall':>9}{'f1':>8}")
    for key, s in per_class.items():
        print(f"{key:18s}{s['support']:9d}{s['precision']:11.3f}{s['recall']:9.3f}{s['f1']:8.3f}")
    print(f"\ncheckpoint {ckpt}\nreport     {out / f'{args.arch}_report.json'}")
    print("\nExport for serving:  python ml/export_onnx.py --checkpoint", ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
