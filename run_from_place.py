#!/usr/bin/env python3
"""Place name or lat/lon → fetch satellite → ONNX mask → georeferenced road graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from geo_imagery import (
    MosaicMeta,
    fetch_mosaic,
    geocode_place,
    georef_geojson,
    slugify,
    write_map_preview,
)
from nn.mask_backend import MaskBackendKind, load_mask_backend, predict_road_prob
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


def save_pipeline_viz(
    out_path: Path,
    rgb: np.ndarray,
    prob: np.ndarray,
    mask: np.ndarray,
    skel: np.ndarray,
    title: str,
    stats: dict,
    threshold: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("satellite (fetched)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(prob, cmap="magma", vmin=0, vmax=1)
    axes[0, 1].set_title(f"road prob (thr={threshold})")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(overlay_mask(rgb, mask))
    axes[1, 0].set_title(f"road mask ({mask.mean()*100:.1f}%)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(rgb)
    skel_rgba = np.zeros((*skel.shape, 4), dtype=float)
    skel_rgba[skel] = [0.2, 1.0, 0.35, 0.9]
    axes[1, 1].imshow(skel_rgba)
    axes[1, 1].set_title(
        f"graph — {stats['nodes']} nodes, {stats['edges']} edges, {stats['dead_ends']} dead-ends"
    )
    axes[1, 1].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_pipeline(
    rgb: np.ndarray,
    meta: MosaicMeta,
    backend,
    cfg: NNVectorizeConfig | TierCConfig,
    *,
    input_size: tuple[int, int],
    threshold: float,
    mode: str = "resize",
    stride: int | None = None,
    legacy_post: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict, dict]:
    prob = predict_road_prob(
        backend,
        rgb,
        output_size=rgb.shape[:2],
        mode=mode,  # type: ignore[arg-type]
        stride=stride,
    )

    if legacy_post:
        mask = prob >= threshold
        graph, simp, pixel_geojson = vectorize_c(mask, cfg)  # type: ignore[arg-type]
        skel = skeletonize_mask(mask)
        tier_label = "NN+TierC"
    else:
        graph, simp, pixel_geojson, mask, skel = vectorize_nn(
            prob, cfg, threshold=threshold  # type: ignore[arg-type]
        )
        tier_label = "NN+postprocess"

    for feat in pixel_geojson["features"]:
        feat["properties"]["tier"] = tier_label
        feat["properties"]["mask_backend"] = backend.mask_backend_tag
        feat["properties"]["model"] = backend.kind

    wgs84_geojson = georef_geojson(pixel_geojson, meta)
    stats = graph_stats(graph, simp)
    stats["mask_fraction"] = float(mask.mean())
    stats["prob_mean"] = float(prob.mean())
    stats["threshold"] = threshold
    stats["postprocess"] = "legacy_tier_c" if legacy_post else "nn_vectorize"
    stats["model"] = backend.kind
    stats["mask_backend"] = backend.mask_backend_tag
    return prob, mask, skel, pixel_geojson, wgs84_geojson, stats


def main():
    root = Path(__file__).parent
    p = argparse.ArgumentParser(
        description="Fetch satellite imagery for a place and extract a georeferenced road graph"
    )
    loc = p.add_mutually_exclusive_group(required=True)
    loc.add_argument("--place", type=str, help='Place name or address, e.g. "Cambridge, MA"')
    loc.add_argument("--lat", type=float, help="Center latitude (use with --lon)")
    p.add_argument("--lon", type=float, help="Center longitude (required with --lat)")
    p.add_argument("--zoom", type=int, default=18, help="Web map zoom (18≈0.6m, 17≈1.2m)")
    p.add_argument("--size", type=int, default=1024, help="Mosaic size in pixels (multiple of 256)")
    p.add_argument("--out", type=Path, default=None, help="Output dir (default: output/places/<slug>)")
    p.add_argument(
        "--model",
        choices=["dlink", "deeplab"],
        default="deeplab",
        help="Mask model backend (default: deeplab)",
    )
    p.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Override ONNX path (default: model-specific)",
    )
    p.add_argument("--input-size", type=int, nargs=2, default=None)
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument(
        "--sliding",
        action="store_true",
        help="Sliding-window infer (experimental — worse with current model; trained on whole-tile resize)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Sliding-window stride when --sliding (default 128)",
    )
    p.add_argument(
        "--legacy-post",
        action="store_true",
        help="Use old Tier C post-process instead of NN-tuned pipeline",
    )
    p.add_argument("--no-html", action="store_true", help="Skip Leaflet preview.html")
    args = p.parse_args()

    if args.lat is not None and args.lon is None:
        raise SystemExit("--lon required when using --lat")
    if args.lon is not None and args.lat is None:
        raise SystemExit("--lat required when using --lon")
    if args.size % 256 != 0:
        raise SystemExit("--size must be a multiple of 256")

    model_kind: MaskBackendKind = args.model  # type: ignore[assignment]
    backend = load_mask_backend(model_kind, args.onnx, root=root)
    input_size = tuple(args.input_size) if args.input_size else backend.input_size

    if model_kind == "deeplab" and args.sliding:
        raise SystemExit("--sliding is D-LinkNet only; DeepLab uses native 1024 inference")
    if model_kind == "deeplab" and args.input_size and tuple(args.input_size) != (1024, 1024):
        print("WARNING: DeepLab trained at 1024×1024; non-default input-size may hurt quality")

    if args.place:
        print(f"Geocoding: {args.place!r}")
        geo = geocode_place(args.place)
        lat, lon, place_label = geo.lat, geo.lon, geo.display_name
        print(f"  → {place_label}")
        print(f"  → lat={lat:.5f}, lon={lon:.5f}")
        slug = slugify(args.place)
    else:
        lat, lon = args.lat, args.lon
        place_label = f"{lat:.5f}, {lon:.5f}"
        slug = slugify(f"{lat}_{lon}")

    out_dir = args.out or (root / "output" / "places" / slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching Esri World Imagery @ z{args.zoom}, {args.size}×{args.size}px …")
    rgb, meta = fetch_mosaic(
        lat, lon, zoom=args.zoom, size_px=args.size, place_name=place_label
    )
    meta_dict = meta.to_dict()
    Image.fromarray(rgb).save(out_dir / "sat.jpg", quality=92)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta_dict, f, indent=2)
    print(f"  bounds: W={meta.west:.5f} S={meta.south:.5f} E={meta.east:.5f} N={meta.north:.5f}")

    print(f"Loading {backend.label}: {backend.default_onnx}")
    cfg = TierCConfig() if args.legacy_post else NNVectorizeConfig()
    infer_mode = "sliding" if args.sliding else (backend.infer_mode or "native")
    stride = args.stride if args.sliding else None
    if args.sliding:
        print(
            "WARNING: --sliding uses native 256px crops; D-LinkNet was trained on "
            "whole-tile 1024→256 resize. Expect weaker masks unless retrained on crops."
        )
    print(
        f"Model: {backend.kind}  input={input_size}  infer={infer_mode}"
        + (f" stride={stride}" if stride else "")
    )
    prob, mask, skel, pixel_gj, wgs84_gj, stats = run_pipeline(
        rgb,
        meta,
        backend,
        cfg,
        input_size=input_size,
        threshold=args.threshold,
        mode=infer_mode,
        stride=stride,
        legacy_post=args.legacy_post,
    )

    stats.update(
        {
            "place": place_label,
            "lat": lat,
            "lon": lon,
            "zoom": args.zoom,
            "size_px": args.size,
            "bounds": [meta.west, meta.south, meta.east, meta.north],
            "infer_mode": infer_mode,
            "stride": stride,
            "onnx": str(backend.default_onnx),
            "input_size": list(input_size),
        }
    )

    with open(out_dir / "roads_pixels.geojson", "w") as f:
        json.dump(pixel_gj, f, indent=2)
    with open(out_dir / "roads.geojson", "w") as f:
        json.dump(wgs84_gj, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    save_pipeline_viz(
        out_dir / "pipeline.png",
        rgb,
        prob,
        mask,
        skel,
        place_label,
        stats,
        args.threshold,
    )

    if not args.no_html:
        write_map_preview(out_dir / "preview.html", meta, wgs84_gj)

    print("\n" + "=" * 60)
    print(f"Place:   {place_label}")
    print(
        f"Graph:   {stats['nodes']} nodes, {stats['edges']} edges, "
        f"{stats['dead_ends']} dead-ends  (mask {stats['mask_fraction']*100:.1f}%)"
    )
    print(f"Output:  {out_dir}/")
    print(f"  sat.jpg, roads.geojson (WGS84), roads_pixels.geojson, pipeline.png")
    if not args.no_html:
        print(f"  preview.html  ← open in browser")


if __name__ == "__main__":
    main()
