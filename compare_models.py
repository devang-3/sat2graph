#!/usr/bin/env python3
"""Compare ONNX / Keras road models: inference time, agreement, mask viz."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mask_from_satellite import list_satellite_images, load_rgb
from nn.deeplab_inference import (
    load_deeplab_model,
    load_deeplab_onnx,
    predict_prob_map as deeplab_pth_prob,
    predict_prob_map_onnx as deeplab_onnx_prob,
)


def preprocess_batch(rgb: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    arr = rgb.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((input_size[1], input_size[0]), Image.BILINEAR)
    batch = np.asarray(pil, dtype=np.float32) / 255.0
    return batch[np.newaxis, ...]


@dataclass
class OnnxBackend:
    name: str
    path: Path
    session: object
    input_name: str
    output_name: str
    load_s: float

    @classmethod
    def load(cls, name: str, path: Path) -> "OnnxBackend":
        import onnxruntime as ort

        t0 = time.perf_counter()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        sess = ort.InferenceSession(str(path), opts, providers=providers)
        load_s = time.perf_counter() - t0
        return cls(
            name=name,
            path=path,
            session=sess,
            input_name=sess.get_inputs()[0].name,
            output_name=sess.get_outputs()[0].name,
            load_s=load_s,
        )

    def prob(self, batch: np.ndarray) -> np.ndarray:
        out = self.session.run([self.output_name], {self.input_name: batch})[0]
        return out[0, ..., 0]


def mask_metrics(prob_ref: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    a = prob_ref >= threshold
    b = prob >= threshold
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return {
        "iou_vs_ref": float(inter / union) if union else 1.0,
        "prob_mae_vs_ref": float(np.mean(np.abs(prob_ref - prob))),
        "mask_frac": float(b.mean()),
    }


def timed_ms(fn, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    deltas = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        deltas.append(time.perf_counter() - t0)
    arr = np.asarray(deltas, dtype=np.float64)
    return float(arr.mean() * 1000), float(arr.std() * 1000)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 220, 0), alpha=0.45) -> np.ndarray:
    img = rgb.copy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    m = mask.astype(bool)
    blend = img.copy()
    blend[m] = (
        (1 - alpha) * blend[m].astype(np.float32) + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return blend


def save_viz_grid(
    out_path: Path,
    rgb_small: np.ndarray,
    panels: list[tuple[str, np.ndarray, float]],
    threshold: float,
    stem: str,
    ref_label: str,
):
    n = len(panels)
    cols = 2
    rows = (n + 1 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_1d(axes).flatten()

    axes[0].imshow(rgb_small)
    axes[0].set_title(f"{stem} — RGB (256²)")
    axes[0].axis("off")

    ref_mask = panels[0][1] >= threshold
    ious = []
    for ax, (label, prob, ms) in zip(axes[1:], panels):
        mask = prob >= threshold
        ax.imshow(overlay_mask(rgb_small, mask))
        ax.set_title(f"{label}  ({ms:.0f} ms)")
        ax.axis("off")
        if label != ref_label:
            inter = (mask & ref_mask).sum()
            union = (mask | ref_mask).sum()
            ious.append(f"{label.split()[-1]}: {inter / max(union, 1):.3f}")

    if ious:
        fig.suptitle(f"IoU vs {ref_label} — " + "  ".join(ious), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_summary_chart(out_path: Path, summary: dict, keys: list[str], labels: list[str]):
    means = [summary[f"{k}_infer_ms_mean"] for k in keys]
    stds = [summary[f"{k}_infer_ms_std"] for k in keys]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"][: len(keys)]

    fig, ax = plt.subplots(figsize=(max(4, 2 * len(keys)), 4))
    ax.bar(labels, means, yerr=stds, capsize=4, color=colors)
    ax.set_ylabel("Inference time (ms)")
    ax.set_title(f"Mean latency @ {summary['input_size']} (n={summary['n_tiles']} tiles)")
    for i, m in enumerate(means):
        ax.text(i, m + stds[i] + 5, f"{m:.0f}ms", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def stat(arr):
    a = np.asarray(arr, dtype=np.float64)
    return float(a.mean()), float(a.std())


def run_fp32_fp16(args, images, input_size):
    for path, label in [
        (args.onnx_fp32, "ONNX FP32"),
        (args.onnx_fp16, "ONNX FP16"),
    ]:
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}\nRun: python3 convert_onnx.py --only fp16")

    viz_dir = args.out / "viz_fp32_fp16"
    if args.viz_limit > 0:
        viz_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ONNX FP32...")
    fp32 = OnnxBackend.load("fp32", args.onnx_fp32)
    print("Loading ONNX FP16...")
    fp16 = OnnxBackend.load("fp16", args.onnx_fp16)
    print(f"  providers: {fp32.session.get_providers()}")

    per_tile = []
    times = {k: [] for k in ("fp32", "fp16")}
    iou_fp16, mae_fp16 = [], []

    print(f"\nBenchmark FP32 vs FP16: {len(images)} tiles @ {input_size[0]}×{input_size[1]}")
    print("=" * 72)

    for vi, path in enumerate(images):
        stem = path.stem.replace("_sat", "")
        rgb = load_rgb(path)
        batch = preprocess_batch(rgb, input_size)
        rgb_small = batch[0]

        t_f32, _ = timed_ms(lambda: fp32.prob(batch), args.warmup, args.repeats)
        t_f16, _ = timed_ms(lambda: fp16.prob(batch), args.warmup, args.repeats)

        prob_f32 = fp32.prob(batch)
        prob_f16 = fp16.prob(batch)
        m16 = mask_metrics(prob_f32, prob_f16, args.threshold)

        row = {
            "stem": stem,
            "fp32_ms": t_f32,
            "fp16_ms": t_f16,
            "iou_fp16_vs_fp32": m16["iou_vs_ref"],
            "mae_fp16_vs_fp32": m16["prob_mae_vs_ref"],
            "mask_frac_fp32": float((prob_f32 >= args.threshold).mean()),
            "mask_frac_fp16": m16["mask_frac"],
        }
        per_tile.append(row)
        times["fp32"].append(t_f32)
        times["fp16"].append(t_f16)
        iou_fp16.append(m16["iou_vs_ref"])
        mae_fp16.append(m16["prob_mae_vs_ref"])

        print(f"{stem:10s}  FP32 {t_f32:6.0f}ms  FP16 {t_f16:6.0f}ms  IoU {m16['iou_vs_ref']:.3f}")

        if args.viz_limit > 0 and vi < args.viz_limit:
            save_viz_grid(
                viz_dir / f"{stem}_compare.png",
                rgb_small,
                [
                    ("ONNX FP32", prob_f32, t_f32),
                    ("ONNX FP16", prob_f16, t_f16),
                ],
                args.threshold,
                stem,
                ref_label="ONNX FP32",
            )

    f32_m, f32_s = stat(times["fp32"])
    f16_m, f16_s = stat(times["fp16"])

    summary = {
        "mode": "fp32-fp16",
        "n_tiles": len(images),
        "input_size": list(input_size),
        "threshold": args.threshold,
        "note": "IoU/MAE vs FP32 reference. No GT masks in dataset/.",
        "fp32_load_s": fp32.load_s,
        "fp16_load_s": fp16.load_s,
        "fp32_infer_ms_mean": f32_m,
        "fp32_infer_ms_std": f32_s,
        "fp16_infer_ms_mean": f16_m,
        "fp16_infer_ms_std": f16_s,
        "iou_fp16_vs_fp32_mean": float(np.mean(iou_fp16)),
        "mae_fp16_vs_fp32_mean": float(np.mean(mae_fp16)),
        "per_tile": per_tile,
    }

    out_json = args.out / "compare_fp32_fp16.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    save_summary_chart(
        args.out / "latency_bar_fp32_fp16.png",
        summary,
        keys=["fp32", "fp16"],
        labels=["FP32", "FP16"],
    )

    print("\n" + "=" * 72)
    print(f"Load (s): FP32 {fp32.load_s:.1f} | FP16 {fp16.load_s:.1f}")
    print(f"Latency:  FP32 {f32_m:.0f}±{f32_s:.0f}ms | FP16 {f16_m:.0f}±{f16_s:.0f}ms")
    print(f"IoU vs FP32: FP16 {summary['iou_fp16_vs_fp32_mean']:.4f}")
    print(f"Saved: {out_json}")
    print(f"Chart: {args.out / 'latency_bar_fp32_fp16.png'}")
    if args.viz_limit > 0:
        print(f"Viz:   {viz_dir}/")


def run_deeplab_pth_onnx(args, images, input_size):
    if not args.deeplab_pth.is_file():
        raise SystemExit(f"DeepLab .pth not found: {args.deeplab_pth}")
    if not args.deeplab_onnx.is_file():
        raise SystemExit(
            f"DeepLab ONNX not found: {args.deeplab_onnx}\n"
            "Run: python3 convert_deeplab_onnx.py"
        )

    viz_dir = args.out / "viz_deeplab_pth_onnx"
    if args.viz_limit > 0:
        viz_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DeepLab PyTorch...")
    t0 = time.perf_counter()
    pth = load_deeplab_model(args.deeplab_pth)
    pth_load_s = time.perf_counter() - t0

    print("Loading DeepLab ONNX FP32...")
    t0 = time.perf_counter()
    onnx = load_deeplab_onnx(args.deeplab_onnx)
    onnx_load_s = time.perf_counter() - t0
    print(f"  PyTorch device: {pth.device}")
    print(f"  ONNX providers: {onnx.session.get_providers()}")

    per_tile = []
    times = {"pth": [], "onnx": []}
    iou_onnx, mae_onnx = [], []

    print(f"\nBenchmark DeepLab PyTorch vs ONNX: {len(images)} tiles @ {input_size[0]}×{input_size[1]}")
    print("=" * 72)

    for vi, path in enumerate(images):
        stem = path.stem.replace("_sat", "")
        rgb = load_rgb(path)
        if rgb.shape[0] != input_size[0] or rgb.shape[1] != input_size[1]:
            rgb = np.array(
                Image.fromarray(rgb).resize((input_size[1], input_size[0]), Image.BILINEAR)
            )

        def run_pth():
            return deeplab_pth_prob(pth, rgb, input_size=input_size, output_size=input_size)

        def run_onnx():
            return deeplab_onnx_prob(onnx, rgb, input_size=input_size, output_size=input_size)

        t_pt, _ = timed_ms(run_pth, args.warmup, args.repeats)
        t_ox, _ = timed_ms(run_onnx, args.warmup, args.repeats)

        prob_pth = run_pth()
        prob_ox = run_onnx()
        m = mask_metrics(prob_pth, prob_ox, args.threshold)

        row = {
            "stem": stem,
            "pth_ms": t_pt,
            "onnx_ms": t_ox,
            "speedup_onnx_vs_pth": t_pt / max(t_ox, 1e-6),
            "iou_onnx_vs_pth": m["iou_vs_ref"],
            "mae_onnx_vs_pth": m["prob_mae_vs_ref"],
            "mask_frac_pth": float((prob_pth >= args.threshold).mean()),
            "mask_frac_onnx": m["mask_frac"],
        }
        per_tile.append(row)
        times["pth"].append(t_pt)
        times["onnx"].append(t_ox)
        iou_onnx.append(m["iou_vs_ref"])
        mae_onnx.append(m["prob_mae_vs_ref"])

        print(
            f"{stem:10s}  PyTorch {t_pt:6.0f}ms  ONNX {t_ox:6.0f}ms  "
            f"{t_pt / max(t_ox, 1e-6):4.1f}x  IoU {m['iou_vs_ref']:.4f}  MAE {m['prob_mae_vs_ref']:.4f}"
        )

        if args.viz_limit > 0 and vi < args.viz_limit:
            save_viz_grid(
                viz_dir / f"{stem}_compare.png",
                rgb,
                [
                    ("DeepLab PyTorch", prob_pth, t_pt),
                    ("DeepLab ONNX FP32", prob_ox, t_ox),
                ],
                args.threshold,
                stem,
                ref_label="DeepLab PyTorch",
            )

    pt_m, pt_s = stat(times["pth"])
    ox_m, ox_s = stat(times["onnx"])

    summary = {
        "mode": "deeplab-pth-onnx",
        "n_tiles": len(images),
        "input_size": list(input_size),
        "threshold": args.threshold,
        "note": "IoU/MAE vs PyTorch reference. Same ImageNet preprocessing.",
        "deeplab_pth": str(args.deeplab_pth),
        "deeplab_onnx": str(args.deeplab_onnx),
        "pth_load_s": pth_load_s,
        "onnx_load_s": onnx_load_s,
        "pth_infer_ms_mean": pt_m,
        "pth_infer_ms_std": pt_s,
        "onnx_infer_ms_mean": ox_m,
        "onnx_infer_ms_std": ox_s,
        "speedup_onnx_vs_pth_mean": pt_m / max(ox_m, 1e-6),
        "iou_onnx_vs_pth_mean": float(np.mean(iou_onnx)),
        "mae_onnx_vs_pth_mean": float(np.mean(mae_onnx)),
        "per_tile": per_tile,
    }

    out_json = args.out / "compare_deeplab_pth_onnx.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    save_summary_chart(
        args.out / "latency_bar_deeplab_pth_onnx.png",
        summary,
        keys=["pth", "onnx"],
        labels=["PyTorch", "ONNX FP32"],
    )

    print("\n" + "=" * 72)
    print(f"Load (s): PyTorch {pth_load_s:.1f} | ONNX {onnx_load_s:.1f}")
    print(f"Latency:  PyTorch {pt_m:.0f}±{pt_s:.0f}ms | ONNX {ox_m:.0f}±{ox_s:.0f}ms ({summary['speedup_onnx_vs_pth_mean']:.2f}x)")
    print(f"IoU vs PyTorch: ONNX {summary['iou_onnx_vs_pth_mean']:.4f}  MAE {summary['mae_onnx_vs_pth_mean']:.4f}")
    print(f"Saved: {out_json}")
    print(f"Chart: {args.out / 'latency_bar_deeplab_pth_onnx.png'}")
    if args.viz_limit > 0:
        print(f"Viz:   {viz_dir}/")


def main():
    root = Path(__file__).parent
    p = argparse.ArgumentParser(description="Compare road mask models")
    p.add_argument(
        "--mode",
        choices=["deeplab-pth-onnx", "fp32-fp16"],
        default="deeplab-pth-onnx",
        help="comparison set (default: deeplab-pth-onnx)",
    )
    p.add_argument("--input", type=Path, default=root.parent / "dataset")
    p.add_argument("--onnx-fp32", type=Path, default=root / "models" / "roads_extraction_fp32.onnx")
    p.add_argument("--onnx-fp16", type=Path, default=root / "models" / "roads_extraction_fp16.onnx")
    p.add_argument("--deeplab-pth", type=Path, default=root / "models" / "best_model.pth")
    p.add_argument("--deeplab-onnx", type=Path, default=root / "models" / "deeplab_fp32.onnx")
    p.add_argument("--out", type=Path, default=root / "output" / "model_compare")
    p.add_argument("--input-size", type=int, nargs=2, default=None)
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--viz-limit", type=int, default=6, help="save mask PNGs for first N tiles (0=off)")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    if args.input_size is None:
        input_size = (1024, 1024) if args.mode == "deeplab-pth-onnx" else (256, 256)
    else:
        input_size = tuple(args.input_size)

    images = list_satellite_images(args.input)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No tiles in {args.input}")

    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode == "fp32-fp16":
        run_fp32_fp16(args, images, input_size)
    else:
        run_deeplab_pth_onnx(args, images, input_size)


if __name__ == "__main__":
    main()
