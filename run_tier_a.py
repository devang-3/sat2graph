#!/usr/bin/env python3
"""sat2graph — run Tier A on synthetic test masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synthetic_masks import ALL
from tier_a import skeletonize_mask, vectorize


def visualize(
    name: str,
    mask: np.ndarray,
    skel: np.ndarray,
    out_dir: Path,
    n_nodes: int,
    n_edges: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(mask, cmap="gray")
    axes[0].set_title(f"{name}: mask")
    axes[1].imshow(skel, cmap="hot")
    axes[1].set_title("skeleton")
    axes[2].imshow(mask, cmap="gray", alpha=0.4)
    axes[2].imshow(skel, cmap="hot", alpha=0.8)
    axes[2].set_title(f"overlay ({n_nodes} nodes, {n_edges} edges)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_one(name: str, mask: np.ndarray, out_dir: Path, spur: float, eps: float) -> dict:
    graph, simplified, geojson = vectorize(mask, spur_min_length=spur, simplify_epsilon=eps)
    skel = skeletonize_mask(mask)

    geo_path = out_dir / f"{name}.geojson"
    with open(geo_path, "w") as f:
        import json

        json.dump(geojson, f, indent=2)

    visualize(name, mask, skel, out_dir, len(graph.nodes), len(simplified))

    coords_counts = [len(list(line.coords)) for _, _, line in simplified]
    return {
        "name": name,
        "nodes": len(graph.nodes),
        "edges": len(simplified),
        "points_per_edge": coords_counts,
        "geojson": str(geo_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Tier A vector road pipeline")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Output directory",
    )
    parser.add_argument("--spur", type=float, default=5.0, help="Min edge length (px)")
    parser.add_argument("--epsilon", type=float, default=2.0, help="DP simplify tolerance (px)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = []

    print("Tier A pipeline test\n" + "=" * 40)
    for name, fn in ALL.items():
        mask = fn()
        stats = run_one(name, mask, args.out, args.spur, args.epsilon)
        results.append(stats)
        pts = stats["points_per_edge"]
        print(
            f"{name:12}  nodes={stats['nodes']:2}  edges={stats['edges']:2}  "
            f"pts/edge={pts}  → {stats['geojson']}"
        )

    print(f"\nDebug PNGs → {args.out}/")
    print("Done.")


if __name__ == "__main__":
    main()
