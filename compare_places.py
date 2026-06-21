#!/usr/bin/env python3
"""Compare D-LinkNet vs DeepLab ONNX backends on fetched places — same post-process."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from geo_imagery import fetch_mosaic, geocode_place, slugify
from nn.mask_backend import load_mask_backend, predict_road_prob
from postprocess_nn import NNVectorizeConfig, vectorize_nn
from tier_b import graph_stats


DEFAULT_PLACES = [
    "Harvard Square, Cambridge MA",
    "Boston, MA",
    "Seattle, WA",
    "Austin, Texas",
    "Chicago, IL",
    "Denver, CO",
    "Phoenix, AZ",
    "Miami, FL",
    "Los Angeles, CA",
    "New York City, NY",
]


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


def save_compare_viz(
    out_path: Path,
    rgb: np.ndarray,
    prob_dlink: np.ndarray,
    prob_deeplab: np.ndarray,
    mask_dlink: np.ndarray,
    mask_deeplab: np.ndarray,
    skel_dlink: np.ndarray,
    skel_deeplab: np.ndarray,
    stats_dlink: dict,
    stats_deeplab: dict,
    place: str,
    threshold: float,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("satellite")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(prob_dlink, cmap="magma", vmin=0, vmax=1)
    axes[0, 1].set_title("D-LinkNet prob")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(prob_deeplab, cmap="magma", vmin=0, vmax=1)
    axes[0, 2].set_title("DeepLab prob")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(overlay_mask(rgb, mask_dlink, color=(255, 200, 0)))
    axes[1, 0].set_title(f"D-Link {mask_dlink.mean()*100:.1f}%")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay_mask(rgb, mask_deeplab, color=(0, 200, 255)))
    axes[1, 1].set_title(f"DeepLab {mask_deeplab.mean()*100:.1f}%")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(rgb)
    skel_rgba = np.zeros((*skel_dlink.shape, 4))
    skel_rgba[skel_dlink] = [1.0, 0.75, 0.0, 0.85]
    skel_rgba[skel_deeplab] = [0.0, 0.85, 1.0, 0.85]
    axes[1, 2].imshow(skel_rgba)
    axes[1, 2].set_title(
        f"D-Link {stats_dlink['nodes']}n/{stats_dlink['edges']}e  "
        f"DeepLab {stats_deeplab['nodes']}n/{stats_deeplab['edges']}e"
    )
    axes[1, 2].axis("off")

    fig.suptitle(
        f"{place}\n"
        f"D-Link {stats_dlink['infer_ms']:.0f}ms dead={stats_dlink['dead_ends']} | "
        f"DeepLab {stats_deeplab['infer_ms']:.0f}ms dead={stats_deeplab['dead_ends']}  "
        f"(thr={threshold})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_model_pipeline(
    prob: np.ndarray,
    cfg: NNVectorizeConfig,
    threshold: float,
    infer_ms: float,
    name: str,
) -> dict:
    graph, simp, _, mask, skel = vectorize_nn(prob, cfg, threshold=threshold)
    stats = graph_stats(graph, simp)
    stats["mask_fraction"] = float(mask.mean())
    stats["prob_mean"] = float(prob.mean())
    stats["infer_ms"] = infer_ms
    stats["model"] = name
    stats["threshold"] = threshold
    return {"stats": stats, "mask": mask, "skel": skel, "prob": prob}


def main():
    root = Path(__file__).parent
    p = argparse.ArgumentParser(description="Compare pipeline mask backends on real places")
    p.add_argument("--places", nargs="+", default=DEFAULT_PLACES)
    p.add_argument("--out", type=Path, default=root / "output" / "compare_models_places")
    p.add_argument("--dlink-onnx", type=Path, default=None)
    p.add_argument("--deeplab-onnx", type=Path, default=None)
    p.add_argument("--zoom", type=int, default=18)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--threshold", type=float, default=0.35)
    args = p.parse_args()

    if args.size % 256 != 0:
        raise SystemExit("--size must be multiple of 256")

    dlink = load_mask_backend("dlink", args.dlink_onnx, root=root)
    deeplab = load_mask_backend("deeplab", args.deeplab_onnx, root=root)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = NNVectorizeConfig()

    print(f"D-Link:  {dlink.default_onnx}  input={dlink.input_size}")
    print(f"DeepLab: {deeplab.default_onnx}  input={deeplab.input_size}")
    print(f"\nCompare {len(args.places)} places @ {args.size}px, thr={args.threshold}")
    print("=" * 72)

    summary = []
    for place in args.places:
        slug = slugify(place)
        place_dir = args.out / slug
        place_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{place}")
        geo = geocode_place(place)
        rgb, meta = fetch_mosaic(geo.lat, geo.lon, zoom=args.zoom, size_px=args.size, place_name=geo.display_name)
        Image.fromarray(rgb).save(place_dir / "sat.jpg")

        t0 = time.perf_counter()
        prob_dl = predict_road_prob(dlink, rgb, output_size=rgb.shape[:2], mode="resize")
        ms_dlink = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        prob_db = predict_road_prob(deeplab, rgb, output_size=rgb.shape[:2])
        ms_deeplab = (time.perf_counter() - t0) * 1000

        res_dlink = run_model_pipeline(prob_dl, cfg, args.threshold, ms_dlink, "dlinknet")
        res_deeplab = run_model_pipeline(prob_db, cfg, args.threshold, ms_deeplab, "deeplab")

        save_compare_viz(
            place_dir / "compare.png",
            rgb,
            prob_dl,
            prob_db,
            res_dlink["mask"],
            res_deeplab["mask"],
            res_dlink["skel"],
            res_deeplab["skel"],
            res_dlink["stats"],
            res_deeplab["stats"],
            geo.display_name,
            args.threshold,
        )

        row = {
            "place": geo.display_name,
            "slug": slug,
            "lat": geo.lat,
            "lon": geo.lon,
            "dlinknet": res_dlink["stats"],
            "deeplab": res_deeplab["stats"],
            "bounds": [meta.west, meta.south, meta.east, meta.north],
        }
        summary.append(row)
        with open(place_dir / "summary.json", "w") as f:
            json.dump(row, f, indent=2)

        d, db = res_dlink["stats"], res_deeplab["stats"]
        print(
            f"  D-LinkNet  {ms_dlink:6.0f}ms  mask={d['mask_fraction']*100:5.1f}%  "
            f"{d['nodes']:3}n {d['edges']:3}e {d['dead_ends']:3} dead"
        )
        print(
            f"  DeepLab    {ms_deeplab:6.0f}ms  mask={db['mask_fraction']*100:5.1f}%  "
            f"{db['nodes']:3}n {db['edges']:3}e {db['dead_ends']:3} dead"
        )

    out_json = args.out / "summary.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "dlinknet": {"path": str(dlink.default_onnx), "input_size": list(dlink.input_size)},
                "deeplab": {"path": str(deeplab.default_onnx), "input_size": list(deeplab.input_size)},
                "postprocess": "NNVectorizeConfig",
                "places": summary,
            },
            f,
            indent=2,
        )
    print(f"\n→ {out_json}")


if __name__ == "__main__":
    main()
