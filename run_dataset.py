#!/usr/bin/env python3
"""Run Tier A on a folder of satellite/aerial images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mask_from_satellite import extract_road_mask, list_images, load_rgb
from tier_a import preprocess_mask, skeletonize_mask, vectorize


def visualize_satellite(
    name: str,
    rgb: np.ndarray,
    mask: np.ndarray,
    skel: np.ndarray,
    out_dir: Path,
    n_nodes: int,
    n_edges: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"{name}: satellite")
    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title(f"road mask ({mask.mean() * 100:.1f}% road)")

    axes[1, 0].imshow(skel, cmap="hot")
    axes[1, 0].set_title("skeleton")

    axes[1, 1].imshow(rgb)
    skel_rgba = np.zeros((*skel.shape, 4), dtype=float)
    skel_rgba[skel] = [1.0, 0.1, 0.1, 0.85]
    axes[1, 1].imshow(skel_rgba)
    axes[1, 1].set_title(f"overlay — {n_nodes} nodes, {n_edges} edges")

    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def process_image(
    path: Path,
    out_dir: Path,
    *,
    spur: float,
    epsilon: float,
    morph_radius: int,
    min_component: int,
) -> dict:
    stem = path.stem.replace("_sat", "")
    rgb = load_rgb(path)
    mask = extract_road_mask(
        rgb, morph_radius=morph_radius, min_component=min_component
    )
    cleaned = preprocess_mask(mask, radius=1)
    skel = skeletonize_mask(cleaned)

    graph, simplified, geojson = vectorize(
        mask,
        morph_radius=1,
        spur_min_length=spur,
        simplify_epsilon=epsilon,
    )

    geojson["properties"] = {"source": path.name}
    for feat in geojson["features"]:
        feat["properties"]["source"] = path.name

    geo_path = out_dir / f"{stem}.geojson"
    with open(geo_path, "w") as f:
        json.dump(geojson, f, indent=2)

    visualize_satellite(
        stem, rgb, mask, skel, out_dir, len(graph.nodes), len(simplified)
    )

    return {
        "file": path.name,
        "stem": stem,
        "shape": list(rgb.shape),
        "mask_fraction": float(mask.mean()),
        "nodes": len(graph.nodes),
        "edges": len(simplified),
        "geojson": str(geo_path),
    }


def main() -> None:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description="sat2graph: vectorize roads from satellite images (Tier A)")
    parser.add_argument(
        "--input",
        type=Path,
        default=root.parent / "dataset",
        help="Folder with satellite images",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "output" / "dataset",
        help="Output folder for GeoJSON + PNG",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max images (0 = all)")
    parser.add_argument("--spur", type=float, default=20.0, help="Min edge length (px)")
    parser.add_argument(
        "--epsilon", type=float, default=4.0, help="Douglas-Peucker tolerance (px)"
    )
    parser.add_argument("--morph", type=int, default=3, help="Mask morphology radius")
    parser.add_argument(
        "--min-component", type=int, default=250, help="Drop blobs smaller than this"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already have GeoJSON in --out",
    )
    args = parser.parse_args()

    images = list_images(args.input)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found in {args.input}")

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    summary: list[dict] = []
    if args.skip_existing and summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    done = {s["stem"] for s in summary}
    if args.skip_existing:
        for p in list_images(args.input):
            stem = p.stem.replace("_sat", "")
            if (args.out / f"{stem}.geojson").exists():
                done.add(stem)
        images = [p for p in images if p.stem.replace("_sat", "") not in done]
        summary = [s for s in summary if s["stem"] in done and (args.out / f"{s['stem']}.geojson").exists()]

    if not images:
        print(f"All images already processed in {args.out}")
        return

    print(f"Processing {len(images)} images from {args.input}\n" + "=" * 50)
    for i, path in enumerate(images, 1):
        stats = process_image(
            path,
            args.out,
            spur=args.spur,
            epsilon=args.epsilon,
            morph_radius=args.morph,
            min_component=args.min_component,
        )
        summary.append(stats)
        print(
            f"[{len(summary):3}] {stats['file']:22}  "
            f"mask={stats['mask_fraction']:.2f}  "
            f"nodes={stats['nodes']:4}  edges={stats['edges']:4}"
        )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs → {args.out}/")
    print(f"Summary  → {summary_path} ({len(summary)} total)")


if __name__ == "__main__":
    main()
