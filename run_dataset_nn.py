#!/usr/bin/env python3
"""ONNX D-LinkNet mask → Tier C vectorize pipeline with debug viz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mask_from_satellite import list_satellite_images, load_rgb
from nn.inference import load_onnx_model, predict_prob_map
from postprocess_nn import NNVectorizeConfig, vectorize_nn
from tier_a import skeletonize_mask
from tier_b import graph_stats
from tier_c import TierCConfig, vectorize_c


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 220, 0), alpha=0.45) -> np.ndarray:
    img = rgb.copy()
    if img.max() <= 1.0:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    m = mask.astype(bool)
    blend = img.copy()
    blend[m] = (
        (1 - alpha) * blend[m].astype(np.float32) + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return blend


def visualize(
    name: str,
    rgb: np.ndarray,
    prob: np.ndarray,
    mask: np.ndarray,
    skel: np.ndarray,
    out_dir: Path,
    stats: dict,
    threshold: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"{name} — satellite")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(prob, cmap="magma", vmin=0, vmax=1)
    axes[0, 1].set_title(f"ONNX FP32 prob (thr={threshold})")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(overlay_mask(rgb, mask))
    axes[1, 0].set_title(f"road mask ({mask.mean()*100:.1f}%)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(rgb)
    skel_rgba = np.zeros((*skel.shape, 4), dtype=float)
    skel_rgba[skel] = [0.2, 1.0, 0.35, 0.9]
    axes[1, 1].imshow(skel_rgba)
    axes[1, 1].set_title(
        f"Tier C graph — {stats['nodes']} nodes, {stats['edges']} edges, "
        f"{stats['dead_ends']} dead-ends"
    )
    axes[1, 1].axis("off")

    fig.suptitle("D-LinkNet ONNX → NN post-process", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path(__file__).parent
    p = argparse.ArgumentParser(description="sat2graph: ONNX neural mask + Tier C vectorize")
    p.add_argument("--input", type=Path, default=root.parent / "dataset")
    p.add_argument("--out", type=Path, default=root / "output" / "dataset_nn")
    p.add_argument(
        "--onnx",
        type=Path,
        default=root / "models" / "roads_extraction_fp32.onnx",
        help="ONNX FP32 weights",
    )
    p.add_argument("--input-size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--stride", type=int, default=128, help="Sliding stride when --sliding")
    p.add_argument(
        "--sliding",
        action="store_true",
        help="Sliding window (experimental; current model trained on whole-tile resize)",
    )
    p.add_argument(
        "--legacy-post",
        action="store_true",
        help="Use old Tier C post-process instead of NN-tuned pipeline",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    if not args.onnx.is_file():
        raise SystemExit(f"ONNX model not found: {args.onnx}")

    cfg = TierCConfig() if args.legacy_post else NNVectorizeConfig()
    input_size = tuple(args.input_size)
    infer_mode = "sliding" if args.sliding else "resize"
    stride = args.stride if args.sliding else None
    images = list_satellite_images(args.input)
    if args.limit > 0:
        images = images[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    summary = []
    if args.skip_existing and summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        done = {s["stem"] for s in summary}
        images = [path for path in images if path.stem.replace("_sat", "") not in done]

    print(f"Loading ONNX FP32: {args.onnx}")
    model = load_onnx_model(args.onnx)
    print(f"  providers: {model.session.get_providers()}")
    print(
        f"NN pipeline: {len(images)} images @ {input_size}, "
        f"post={'legacy' if args.legacy_post else 'nn'}, thr={args.threshold}\n"
        + "=" * 50
    )

    for path in images:
        stem = path.stem.replace("_sat", "")
        rgb = load_rgb(path)
        prob = predict_prob_map(
            model,
            rgb,
            input_size=input_size,
            output_size=rgb.shape[:2],
            mode=infer_mode,
            stride=stride,
        )
        if args.legacy_post:
            mask = prob >= args.threshold
            graph, simp, geojson = vectorize_c(mask, cfg)
            skel = skeletonize_mask(mask)
        else:
            graph, simp, geojson, mask, skel = vectorize_nn(
                prob, cfg, threshold=args.threshold
            )

        for feat in geojson["features"]:
            feat["properties"]["source"] = path.name
            feat["properties"]["tier"] = "NN+TierC" if args.legacy_post else "NN+postprocess"
            feat["properties"]["mask_backend"] = "onnx_fp32"

        with open(args.out / f"{stem}.geojson", "w") as f:
            json.dump(geojson, f, indent=2)

        stats = graph_stats(graph, simp)
        stats["mask_fraction"] = float(mask.mean())
        stats["prob_mean"] = float(prob.mean())
        stats["file"] = path.name
        stats["stem"] = stem
        stats["onnx"] = str(args.onnx.name)
        stats["threshold"] = args.threshold
        stats["infer_mode"] = infer_mode
        stats["stride"] = stride
        summary.append(stats)
        visualize(stem, rgb, prob, mask, skel, args.out, stats, args.threshold)
        print(
            f"{path.name:22} mask={stats['mask_fraction']:.2f}  "
            f"nodes={stats['nodes']:4} edges={stats['edges']:4} dead={stats['dead_ends']:4}"
        )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n→ {args.out}/")


if __name__ == "__main__":
    main()
