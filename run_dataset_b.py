#!/usr/bin/env python3
"""Run Tier B on satellite images (same interface as run_dataset.py)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mask_from_satellite import extract_road_mask_b, list_satellite_images, load_rgb
from tier_a import skeletonize_mask
from tier_b import TierBConfig, graph_stats, vectorize_b


def visualize(
    name: str,
    rgb: np.ndarray,
    mask: np.ndarray,
    skel: np.ndarray,
    out_dir: Path,
    stats: dict,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"{name}: satellite")
    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title(f"Tier B mask ({mask.mean()*100:.1f}%)")
    axes[1, 0].imshow(skel, cmap="hot")
    axes[1, 0].set_title("skeleton")
    axes[1, 1].imshow(rgb)
    skel_rgba = np.zeros((*skel.shape, 4), dtype=float)
    skel_rgba[skel] = [1.0, 0.4, 0.1, 0.85]
    axes[1, 1].imshow(skel_rgba)
    axes[1, 1].set_title(
        f"Tier B — {stats['nodes']} nodes, {stats['edges']} edges, "
        f"{stats['dead_ends']} dead-ends"
    )
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def process_image(path: Path, out_dir: Path, cfg: TierBConfig) -> dict:
    stem = path.stem.replace("_sat", "")
    rgb = load_rgb(path)
    mask = extract_road_mask_b(rgb)
    graph, simplified, geojson = vectorize_b(mask, cfg)
    skel = skeletonize_mask(mask)

    for feat in geojson["features"]:
        feat["properties"]["source"] = path.name
        feat["properties"]["tier"] = "B"

    geo_path = out_dir / f"{stem}.geojson"
    with open(geo_path, "w") as f:
        json.dump(geojson, f, indent=2)

    stats = graph_stats(graph, simplified)
    stats["mask_fraction"] = float(mask.mean())
    visualize(stem, rgb, mask, skel, out_dir, stats)

    return {
        "file": path.name,
        "stem": stem,
        "tier": "B",
        **stats,
        "geojson": str(geo_path),
    }


def main() -> None:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description="sat2graph: Tier B vectorization on satellite folder")
    parser.add_argument("--input", type=Path, default=root.parent / "dataset")
    parser.add_argument("--out", type=Path, default=root / "output" / "dataset_b")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gap-dist", type=float, default=35.0)
    parser.add_argument("--gap-angle", type=float, default=45.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = TierBConfig(
        gap_max_distance=args.gap_dist,
        gap_max_angle_deg=args.gap_angle,
    )

    images = list_satellite_images(args.input)
    if args.limit > 0:
        images = images[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    summary: list[dict] = []
    if args.skip_existing and summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        done = {s["stem"] for s in summary}
        images = [p for p in images if p.stem.replace("_sat", "") not in done]

    print(f"Tier B: {len(images)} images\n" + "=" * 50)
    for path in images:
        row = process_image(path, args.out, cfg)
        summary.append(row)
        print(
            f"{row['file']:22}  mask={row['mask_fraction']:.2f}  "
            f"nodes={row['nodes']:4}  edges={row['edges']:4}  "
            f"dead={row['dead_ends']:4}"
        )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n→ {args.out}/")


if __name__ == "__main__":
    main()
