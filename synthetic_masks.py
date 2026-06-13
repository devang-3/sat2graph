"""Synthetic road masks for testing Tier A."""

import numpy as np


def blank(h: int = 128, w: int = 128) -> np.ndarray:
    return np.zeros((h, w), dtype=bool)


def draw_line(mask: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], width: int = 7) -> None:
    """Draw thick line between (row, col) endpoints."""
    r0, c0 = p0
    r1, c1 = p1
    length = int(np.hypot(r1 - r0, c1 - c0)) + 1
    for t in np.linspace(0, 1, length):
        r = int(round(r0 + t * (r1 - r0)))
        c = int(round(c0 + t * (c1 - c0)))
        r_lo, r_hi = max(0, r - width // 2), min(mask.shape[0], r + width // 2 + 1)
        c_lo, c_hi = max(0, c - width // 2), min(mask.shape[1], c + width // 2 + 1)
        mask[r_lo:r_hi, c_lo:c_hi] = True


def cross(h: int = 128, w: int = 128, width: int = 7) -> np.ndarray:
    m = blank(h, w)
    mid_r, mid_c = h // 2, w // 2
    draw_line(m, (mid_r, 10), (mid_r, w - 10), width)
    draw_line(m, (10, mid_c), (h - 10, mid_c), width)
    return m


def t_junction(h: int = 128, w: int = 128, width: int = 7) -> np.ndarray:
    m = blank(h, w)
    mid_r, mid_c = h // 2, w // 2
    draw_line(m, (mid_r, 10), (mid_r, w - 10), width)
    draw_line(m, (10, mid_c), (mid_r, mid_c), width)
    return m


def loop(h: int = 128, w: int = 128, width: int = 7) -> np.ndarray:
    m = blank(h, w)
    cx, cy = h // 2, w // 2
    for angle in np.linspace(0, 2 * np.pi, 200):
        r = int(cx + 35 * np.sin(angle))
        c = int(cy + 45 * np.cos(angle))
        r_lo, r_hi = max(0, r - width // 2), min(h, r + width // 2 + 1)
        c_lo, c_hi = max(0, c - width // 2), min(w, c + width // 2 + 1)
        m[r_lo:r_hi, c_lo:c_hi] = True
    return m


def grid_2x2(h: int = 128, w: int = 128, width: int = 7) -> np.ndarray:
    m = blank(h, w)
    for r in (h // 3, 2 * h // 3):
        draw_line(m, (r, 10), (r, w - 10), width)
    for c in (w // 3, 2 * w // 3):
        draw_line(m, (10, c), (h - 10, c), width)
    return m


ALL = {
    "cross": cross,
    "t_junction": t_junction,
    "loop": loop,
    "grid_2x2": grid_2x2,
}
