"""sat2graph — road mask extraction (Tier A / B / C heuristics)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import gaussian, threshold_local, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import (
    closing,
    disk,
    opening,
    reconstruction,
    remove_small_objects,
    white_tophat,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_rgb(path: Path | str) -> np.ndarray:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def _bright_road_mask(gray, saturation, value, t: float) -> np.ndarray:
    return (
        (gray > t * 0.88)
        & (saturation < 0.42)
        & (value > 0.10)
        & (value < 0.94)
    )


def _dark_road_mask(gray, saturation, value) -> np.ndarray:
    p40, p55 = np.percentile(gray, 40), np.percentile(gray, 55)
    return (
        (gray > p40)
        & (gray < p55)
        & (saturation < 0.38)
        & (value > 0.08)
        & (value < 0.65)
    )


def _clean_mask(
    mask: np.ndarray, *, open_r: int = 3, close_r: int = 5, min_comp: int = 300
) -> np.ndarray:
    mask = opening(mask, disk(open_r))
    mask = closing(mask, disk(close_r))
    mask = remove_small_objects(mask, max_size=max(min_comp - 1, 1))
    return closing(mask, disk(2))


def extract_road_mask(
    img: np.ndarray,
    *,
    morph_radius: int = 3,
    min_component: int = 250,
) -> np.ndarray:
    """Tier A — global Otsu + HSV."""
    gray = rgb2gray(img)
    hsv = rgb2hsv(img)
    s, v = hsv[..., 1], hsv[..., 2]
    t = threshold_otsu(gray)
    return _clean_mask(
        _bright_road_mask(gray, s, v, t),
        open_r=morph_radius,
        close_r=morph_radius,
        min_comp=min_component,
    )


def extract_road_mask_b(
    img: np.ndarray,
    *,
    morph_radius: int = 4,
    min_component: int = 400,
    close_radius: int = 5,
) -> np.ndarray:
    """Tier B — adaptive + top-hat + Otsu fallback."""
    gray = rgb2gray(img)
    hsv = rgb2hsv(img)
    s, v = hsv[..., 1], hsv[..., 2]

    local_t = threshold_local(gray, block_size=51, offset=-0.02)
    bright = (gray > local_t) & (s < 0.45) & (v > 0.10) & (v < 0.93)
    tophat = white_tophat(gray, disk(8))
    tophat_m = tophat > np.percentile(tophat, 85)
    t = threshold_otsu(gray)
    otsu_m = (gray > t * 0.85) & (s < 0.42)
    mask = bright | (tophat_m & otsu_m)

    base = _bright_road_mask(gray, s, v, t)
    if mask.mean() < 0.05:
        mask = base
    else:
        mask = mask | (base & closing(mask, disk(12)))

    mask = gaussian(mask.astype(float), sigma=1.0) > 0.4
    return _clean_mask(mask, open_r=morph_radius, close_r=close_radius, min_comp=min_component)


def extract_road_mask_c(img: np.ndarray) -> np.ndarray:
    """
    Tier C mask — ensemble voting + vegetation reject + morphological reconstruction.

    Heuristic only; tuned for dataset/ Mass-Roads-style tiles.
    """
    gray = rgb2gray(img)
    hsv = rgb2hsv(img)
    s, v = hsv[..., 1], hsv[..., 2]
    t = threshold_otsu(gray)

    m_bright = _bright_road_mask(gray, s, v, t)
    m_dark = _dark_road_mask(gray, s, v)
    local_t = threshold_local(gray, block_size=35, offset=-0.01)
    m_local = (gray > local_t) & (s < 0.40) & (v > 0.08) & (v < 0.90)

    votes = m_bright.astype(np.uint8) + m_dark.astype(np.uint8) + m_local.astype(np.uint8)
    mask = votes >= 2  # 2-of-3 agreement

    # Per-tile road appearance
    if m_bright.sum() > m_dark.sum() * 1.3:
        mask = mask | (m_bright & closing(m_bright, disk(5)))
    elif m_dark.sum() > m_bright.sum() * 1.3:
        mask = mask | (m_dark & closing(m_dark, disk(5)))
    else:
        mask = mask | ((m_bright | m_dark) & (votes >= 1))

    # Reject strong vegetation (high green relative to red)
    r, g, b = img[..., 0].astype(float), img[..., 1].astype(float), img[..., 2].astype(float)
    vegetation = (g > r * 1.15) & (g > b * 1.12) & (s > 0.28)
    mask = mask & ~vegetation

    # Morphological reconstruction: fill holes inside roads
    seed = opening(mask, disk(2))
    mask = reconstruction(seed, mask.astype(bool), method="dilation")
    mask = closing(mask, disk(6))

    # Drop only small compact blobs (roofs), keep elongated road segments
    labeled = label(mask)
    filtered = np.zeros_like(mask, dtype=bool)
    for reg in regionprops(labeled):
        if reg.area < 120:
            continue
        if reg.eccentricity >= 0.45 or reg.area > 5000:
            filtered[labeled == reg.label] = True
    if filtered.sum() > mask.sum() * 0.15:
        mask = filtered

    return _clean_mask(mask, open_r=2, close_r=6, min_comp=300)


def list_satellite_images(folder: Path) -> list[Path]:
    folder = Path(folder)
    sat = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and "_sat" in p.stem
    )
    return sat if sat else list_images(folder)


def list_images(folder: Path) -> list[Path]:
    folder = Path(folder)
    return [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
