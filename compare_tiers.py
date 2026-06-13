#!/usr/bin/env python3
"""Compare Tier A / B / C on dataset/ only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

from mask_from_satellite import (
    extract_road_mask,
    extract_road_mask_b,
    extract_road_mask_c,
    list_satellite_images,
    load_rgb,
)
from tier_a import skeletonize_mask, vectorize
from tier_b import TierBConfig, graph_stats, vectorize_b
from tier_c import TierCConfig, vectorize_c


def _draw_graph(ax, rgb, lines, color: str, lw: float = 1.0) -> None:
    ax.imshow(rgb)
    for _, _, line in lines:
        xs, ys = line.xy
        ax.plot(
            xs, ys, color=color, linewidth=lw, solid_capstyle="round",
            path_effects=[pe.Stroke(linewidth=lw + 1.2, foreground="black"), pe.Normal()],
        )


def compare_one(path: Path, out_dir: Path, cfg_b: TierBConfig, cfg_c: TierCConfig) -> dict:
    stem = path.stem.replace("_sat", "")
    rgb = load_rgb(path)

    mask_a = extract_road_mask(rgb)
    mask_b = extract_road_mask_b(rgb)
    mask_c = extract_road_mask_c(rgb)

    g_a, s_a, _ = vectorize(mask_a, spur_min_length=20, simplify_epsilon=4)
    g_b, s_b, _ = vectorize_b(mask_b, cfg_b)
    g_c, s_c, _ = vectorize_c(mask_c, cfg_c)

    st_a = graph_stats(g_a, s_a)
    st_b = graph_stats(g_b, s_b)
    st_c = graph_stats(g_c, s_c)
    st_a["mask_fraction"] = float(mask_a.mean())
    st_b["mask_fraction"] = float(mask_b.mean())
    st_c["mask_fraction"] = float(mask_c.mean())

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    tiers = [
        ("A", mask_a, s_a, st_a, "#00d4ff"),
        ("B", mask_b, s_b, st_b, "#ff6b35"),
        ("C", mask_c, s_c, st_c, "#7cfc00"),
    ]
    for row, (name, mask, simp, st, color) in enumerate(tiers):
        skel = skeletonize_mask(mask)
        axes[row, 0].imshow(mask, cmap="gray")
        axes[row, 0].set_title(f"Tier {name} mask ({st['mask_fraction']*100:.1f}%)")
        axes[row, 1].imshow(skel, cmap="hot")
        axes[row, 1].set_title(f"Tier {name} skeleton")
        axes[row, 2].set_title(f"Tier {name}: {st['nodes']}n {st['edges']}e {st['dead_ends']} dead")
        _draw_graph(axes[row, 2], rgb, simp, color)

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{stem} — A / B / C", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_abc.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_graph(ax, rgb, s_a, "#00d4ff", 0.8)
    _draw_graph(ax, rgb, s_b, "#ff6b35", 0.8)
    _draw_graph(ax, rgb, s_c, "#7cfc00", 1.2)
    ax.set_title(f"{stem}: cyan=A orange=B green=C")
    ax.axis("off")
    fig.savefig(out_dir / f"{stem}_overlay_abc.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    return {"file": path.name, "stem": stem, "tier_a": st_a, "tier_b": st_b, "tier_c": st_c}


def main() -> None:
    root = Path(__file__).parent
    p = argparse.ArgumentParser(description="sat2graph: compare Tier A / B / C on dataset/")
    p.add_argument("--input", type=Path, default=root.parent / "dataset")
    p.add_argument("--out", type=Path, default=root / "output" / "compare_abc")
    p.add_argument("--limit", type=int, default=8)
    args = p.parse_args()

    images = list_satellite_images(args.input)[: args.limit or None]
    if not images:
        raise SystemExit(f"No images in {args.input}")

    cfg_b = TierBConfig()
    cfg_c = TierCConfig()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"A/B/C compare on {len(images)} images from {args.input}\n" + "=" * 72)
    hdr = f"{'image':22} {'An':>5} {'Bn':>5} {'Cn':>5}  {'Ae':>5} {'Be':>5} {'Ce':>5}  {'Ad':>5} {'Bd':>5} {'Cd':>5}"
    print(hdr)
    print("-" * 72)

    for path in images:
        row = compare_one(path, args.out, cfg_b, cfg_c)
        results.append(row)
        a, b, c = row["tier_a"], row["tier_b"], row["tier_c"]
        print(
            f"{row['file']:22} {a['nodes']:5} {b['nodes']:5} {c['nodes']:5}  "
            f"{a['edges']:5} {b['edges']:5} {c['edges']:5}  "
            f"{a['dead_ends']:5} {b['dead_ends']:5} {c['dead_ends']:5}"
        )

    with open(args.out / "comparison_abc.json", "w") as f:
        json.dump(results, f, indent=2)

    for tier in ("a", "b", "c"):
        key = f"tier_{tier}"
        print(f"Avg {tier.upper()} nodes: {np.mean([r[key]['nodes'] for r in results]):.0f}  "
              f"edges: {np.mean([r[key]['edges'] for r in results]):.0f}  "
              f"dead: {np.mean([r[key]['dead_ends'] for r in results]):.0f}")
    print(f"\n→ {args.out}/")


if __name__ == "__main__":
    main()
