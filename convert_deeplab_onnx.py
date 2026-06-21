#!/usr/bin/env python3
"""Export DeepLabV3+ PyTorch checkpoint → ONNX FP32."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nn.deeplab_inference import _preprocess_rgb, load_deeplab_model


def export_deeplab_onnx(
    weights: Path,
    out: Path,
    input_size: tuple[int, int] = (1024, 1024),
    opset: int = 17,
) -> None:
    import torch

    dl = load_deeplab_model(weights)
    model = dl.model.cpu().eval()

    h, w = input_size
    dummy_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    x = torch.from_numpy(_preprocess_rgb(dummy_rgb)).unsqueeze(0)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        x,
        str(out),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        opset_version=opset,
        do_constant_folding=True,
    )
    mb = out.stat().st_size / 1e6
    print(f"ONNX saved: {out} ({mb:.1f} MB)  input={list(input_size)} opset={opset}")


def main():
    root = Path(__file__).parent
    p = argparse.ArgumentParser(description="Export DeepLabV3+ .pth → ONNX FP32")
    p.add_argument("--weights", type=Path, default=root / "models" / "best_model.pth")
    p.add_argument("--out", type=Path, default=root / "models" / "deeplab_fp32.onnx")
    p.add_argument("--input-size", type=int, nargs=2, default=[1024, 1024])
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")

    export_deeplab_onnx(args.weights, args.out, tuple(args.input_size), args.opset)


if __name__ == "__main__":
    main()
