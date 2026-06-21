"""Dataset discovery and loading for road segmentation (Mass Roads / DeepGlobe layouts)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img

Layout = Literal["auto", "mass_roads", "deepglobe", "flat_mass"]

_TILE_RE = re.compile(r"^(.+?)_sat\.(jpg|jpeg|png)$", re.IGNORECASE)
_MASK_SUFFIXES = (".png", ".jpg", ".jpeg")


def _mask_candidates(tile_id: str, mask_dir: Path) -> list[Path]:
    out: list[Path] = []
    for ext in _MASK_SUFFIXES:
        out.append(mask_dir / f"{tile_id}_mask{ext}")
    return out


def discover_pairs(
    data_root: str | Path,
    layout: Layout = "auto",
) -> list[tuple[Path, Path]]:
    """
    Return (sat_image, mask) path pairs.

    Layouts:
      - mass_roads:  {root}/sat/*_sat.jpg + {root}/gt/*_mask.png
      - deepglobe:   {root}/*_sat.jpg + {root}/*_mask.jpg
      - flat_mass:   {root}/*_sat.jpg + {root}/*_mask.png  (same folder)
      - auto:        try mass_roads, then flat_mass, then deepglobe
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")

    def mass_roads_pairs() -> list[tuple[Path, Path]]:
        sat_dir, gt_dir = root / "sat", root / "gt"
        if not sat_dir.is_dir() or not gt_dir.is_dir():
            return []
        pairs: list[tuple[Path, Path]] = []
        for sat in sorted(sat_dir.glob("*_sat.*")):
            m = _TILE_RE.match(sat.name)
            if not m:
                continue
            tile_id = m.group(1)
            masks = [p for p in _mask_candidates(tile_id, gt_dir) if p.is_file()]
            if masks:
                pairs.append((sat, masks[0]))
        return pairs

    def flat_pairs(mask_ext: str) -> list[tuple[Path, Path]]:
        pairs: list[tuple[Path, Path]] = []
        for sat in sorted(root.glob("*_sat.*")):
            m = _TILE_RE.match(sat.name)
            if not m:
                continue
            tile_id = m.group(1)
            mask = root / f"{tile_id}_mask{mask_ext}"
            if mask.is_file():
                pairs.append((sat, mask))
        return pairs

    def deepglobe_pairs() -> list[tuple[Path, Path]]:
        return flat_pairs(".jpg") or flat_pairs(".png")

    if layout == "mass_roads":
        pairs = mass_roads_pairs()
    elif layout == "flat_mass":
        pairs = flat_pairs(".png") or flat_pairs(".jpg")
    elif layout == "deepglobe":
        pairs = deepglobe_pairs()
    else:
        pairs = mass_roads_pairs() or flat_pairs(".png") or deepglobe_pairs()

    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs under {root}. "
            "Expected Mass Roads (sat/ + gt/), flat *_sat.jpg + *_mask.png, or DeepGlobe."
        )
    return pairs


def split_by_tile_id(
    pairs: list[tuple[Path, Path]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 110,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Split by tile (no leakage). Ratios apply to train+val+test."""
    ids = [p[0].stem.replace("_sat", "") for p in pairs]
    train_val, test = train_test_split(pairs, test_size=test_ratio, random_state=seed, stratify=None)
    val_size = val_ratio / (1.0 - test_ratio) if test_ratio < 1 else val_ratio
    train, val = train_test_split(train_val, test_size=val_size, random_state=seed)
    return list(train), list(val), list(test)


def _load_pair(
    sat_path: Path,
    mask_path: Path,
    input_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    img = load_img(str(sat_path), target_size=input_size, color_mode="rgb")
    mask = load_img(str(mask_path), target_size=input_size, color_mode="grayscale")

    img_arr = img_to_array(img).astype(np.float32) / 255.0
    mask_arr = img_to_array(mask).astype(np.float32)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    mask_arr = (mask_arr > 127.0).astype(np.float32)
    mask_arr = mask_arr[..., np.newaxis]
    return img_arr, mask_arr


def pairs_to_arrays(
    pairs: list[tuple[Path, Path]],
    input_size: tuple[int, int] = (256, 256),
    augment: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for sat_path, mask_path in pairs:
        img, mask = _load_pair(sat_path, mask_path, input_size)
        images.append(img)
        masks.append(mask)
        if augment:
            images.append(np.flip(img, axis=1).copy())
            masks.append(np.flip(mask, axis=1).copy())
            images.append(np.flip(img, axis=0).copy())
            masks.append(np.flip(mask, axis=0).copy())

    return np.asarray(images, dtype=np.float32), np.asarray(masks, dtype=np.float32)


def load_split_arrays(
    data_root: str | Path,
    input_size: tuple[int, int] = (256, 256),
    layout: Layout = "auto",
    augment_train: bool = False,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 110,
):
    pairs = discover_pairs(data_root, layout=layout)
    train_p, val_p, test_p = split_by_tile_id(pairs, val_ratio, test_ratio, seed)
    train_x, train_y = pairs_to_arrays(train_p, input_size, augment=augment_train)
    val_x, val_y = pairs_to_arrays(val_p, input_size, augment=False)
    test_x, test_y = pairs_to_arrays(test_p, input_size, augment=False)
    meta = {
        "n_total": len(pairs),
        "n_train": len(train_p),
        "n_val": len(val_p),
        "n_test": len(test_p),
        "train_tiles": [p[0].name for p in train_p],
        "test_tiles": [p[0].name for p in test_p],
    }
    return train_x, train_y, val_x, val_y, test_x, test_y, meta
