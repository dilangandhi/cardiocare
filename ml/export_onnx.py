"""Export a trained checkpoint to ONNX for serving.

    python ml/export_onnx.py --checkpoint models/resnet50_best.pt

The service loads whatever .onnx file it finds in models/. ONNX is used rather
than a raw PyTorch checkpoint so the runtime image does not need torch at all:
onnxruntime is roughly 50 MB against torch's 900 MB, which matters a great deal
when deploying to a free tier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

try:
    import numpy as np
    import torch
except ImportError:
    sys.exit("PyTorch is required to export. pip install torch torchvision")

from app.core.taxonomy import KEYS, NUM_CLASSES  # noqa: E402
from train import build_model  # noqa: E402


def final_linear(model, arch: str):
    """The classification head, whose output rows are the class scores."""
    if arch == "resnet50":
        return model.fc
    if arch == "densenet121":
        return model.classifier
    return model.classifier[1]  # efficientnet_b0, mobilenet_v2


def permute_head(model, arch: str, source_order: list) -> None:
    """Reorder the head so output index i means ``KEYS[i]``.

    Checkpoints trained before the ImageFolder ordering fix emit classes in
    alphabetical folder order. Permuting the rows of the final linear layer
    converts them to taxonomy order without retraining -- the learned features
    are untouched, only the order in which the scores come out.
    """
    import torch

    if list(source_order) == list(KEYS):
        return
    perm = [source_order.index(k) for k in KEYS]
    head = final_linear(model, arch)
    with torch.no_grad():
        head.weight.copy_(head.weight[perm].clone())
        if head.bias is not None:
            head.bias.copy_(head.bias[perm].clone())
    print(f"Permuted the classifier head from {list(source_order)}")
    print(f"                              to {list(KEYS)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None, help="output .onnx path")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--source-order", default=None, nargs="*",
                    help="class order the checkpoint emits, if it predates the "
                         "ImageFolder ordering fix; defaults to alphabetical")
    ap.add_argument("--training-data", default=None,
                    choices=["physionet", "synthetic", "unknown"],
                    help="provenance of the training set; inferred from the "
                         "checkpoint report when omitted")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"No checkpoint at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch = ckpt.get("arch", "resnet50")
    size = int(ckpt.get("size", 224))
    classes = ckpt.get("classes", list(KEYS))

    model = build_model(arch, NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Checkpoints written by the fixed train.py are already in taxonomy order.
    # Anything older emits ImageFolder's alphabetical order and needs permuting.
    if args.source_order:
        source_order = list(args.source_order)
    elif ckpt.get("class_order") == "taxonomy":
        source_order = list(KEYS)
    else:
        source_order = sorted(KEYS)
        print("Checkpoint predates the class-ordering fix; assuming alphabetical")
        print("folder order. Pass --source-order to override.")
    permute_head(model, arch, source_order)
    del classes

    out = Path(args.out) if args.out else ckpt_path.with_suffix(".onnx")
    dummy = torch.randn(1, 3, size, size)

    torch.onnx.export(
        model, dummy, str(out),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )

    # Verify the export actually reproduces the PyTorch output.
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {"image": dummy.numpy()})[0]
        with torch.no_grad():
            torch_out = model(dummy).numpy()
        drift = float(np.abs(onnx_out - torch_out).max())
        print(f"max |onnx - torch| = {drift:.2e}")
        if drift > 1e-3:
            print("WARNING: exported model diverges from the checkpoint.")
    except ImportError:
        print("onnxruntime not installed; skipping verification.")

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    training_data = args.training_data
    if training_data is None:
        # Infer from the dataset the run used, so provenance is not left to memory.
        training_data = "unknown"
        report = ckpt_path.parent / f"{arch}_report.json"
        if report.exists():
            try:
                data_dir = Path(json.loads(report.read_text())["args"]["data"])
                ds = data_dir / "dataset.json"
                if ds.exists():
                    src = json.loads(ds.read_text()).get("source")
                    training_data = "synthetic" if src == "synthetic" else "physionet"
            except (KeyError, json.JSONDecodeError, OSError):
                pass

    if training_data == "synthetic":
        print("\nNOTE: tagged as synthetic. The service will display that these")
        print("weights carry no evidence about clinical accuracy.")
    elif training_data == "unknown":
        print("\nWARNING: could not determine training provenance. Pass")
        print("--training-data explicitly so the service can report it.")

    meta = {
        "arch": arch, "input_size": size, "classes": list(KEYS),
        "opset": args.opset, "sha256": sha, "training_data": training_data,
        "val_balanced_accuracy": ckpt.get("val_balanced_accuracy"),
        "normalisation": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"sha256 {sha[:16]}")
    print("\nPlace it in models/ and restart the service to enable the CNN path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
