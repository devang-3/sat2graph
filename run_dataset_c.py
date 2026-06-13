#!/usr/bin/env python3
"""Run Tier C (heuristic) on dataset/ folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mask_from_satellite import extract_road_mask_c, list_satellite_images, load_rgb
from tier_a import skeletonize_mask
from tier_b import graph_stats
from tier_c import TierCConfig, vectorize_c


def visualize(name, rgb, mask, skel, out_dir, stats):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("satellite")
    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title(f"Tier C mask ({mask.mean()*100:.1f}%)")
    axes[1, 0].imshow(skel, cmap="hot")
    axes[1, 0].set_title("skeleton")
    axes[1, 1].imshow(rgb)
    rgba = np.zeros((*skel.shape, 4))
    rgba[skel] = [0.2, 1.0, 0.3, 0.85]
    axes[1, 1].imshow(rgba)
    axes[1, 1].set_title(
        f"{stats['nodes']} nodes, {stats['edges']} edges, {stats['dead_ends']} dead-ends"
    )
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path(__file__).parent
    p = argparse.ArgumentParser(description="sat2graph: Tier C heuristic vectorization")
    p.add_argument("--input", type=Path, default=root.parent / "dataset")
    p.add_argument("--out", type=Path, default=root / "output" / "dataset_c")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    cfg = TierCConfig()
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
        images = [p for p in images if p.stem.replace("_sat", "") not in done]

    print(f"Tier C: {len(images)} images\n" + "=" * 50)
    for path in images:
        stem = path.stem.replace("_sat", "")
        rgb = load_rgb(path)
        mask = extract_road_mask_c(rgb)
        graph, simp, geojson = vectorize_c(mask, cfg)
        skel = skeletonize_mask(mask)

        for feat in geojson["features"]:
            feat["properties"]["source"] = path.name
            feat["properties"]["tier"] = "C"

        with open(args.out / f"{stem}.geojson", "w") as f:
            json.dump(geojson, f, indent=2)

        stats = graph_stats(graph, simp)
        stats["mask_fraction"] = float(mask.mean())
        stats["file"] = path.name
        stats["stem"] = stem
        summary.append(stats)
        visualize(stem, rgb, mask, skel, args.out, stats)
        print(
            f"{path.name:22} mask={stats['mask_fraction']:.2f}  "
            f"nodes={stats['nodes']:4} edges={stats['edges']:4} dead={stats['dead_ends']:4}"
        )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n→ {args.out}/")


if __name__ == "__main__":
    main()
