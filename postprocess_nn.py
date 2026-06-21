"""Post-processing tuned for D-LinkNet probability masks → road graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import LineString
from skimage.measure import label, regionprops
from skimage.morphology import (
    closing,
    disk,
    opening,
    reconstruction,
    remove_small_objects,
    skeletonize,
)

from tier_a import (
    RoadGraph,
    extract_graph,
    merge_nearby_nodes,
    simplify_edges,
    to_geojson,
)
from tier_c import (
    TierCConfig,
    close_junction_gaps,
    filter_road_components,
    merge_collinear_edges,
    prune_spurs_gentle,
    remove_small_loops,
)
from tier_b import close_gaps


@dataclass
class NNVectorizeConfig(TierCConfig):
    """Tier C params tuned for neural road probability maps."""

    # prob → binary
    threshold_high: float = 0.42
    threshold_low: float = 0.22
    prob_smooth_sigma: float = 0.8

    # mask morphology (lighter than heuristic Tier C)
    open_radius: int = 1
    close_radius: int = 4
    min_component_px: int = 40

    # skeleton
    min_road_radius: float = 1.0
    max_road_radius: float = 20.0

    # graph (more bridging, less aggressive component drop)
    merge_radius: float = 8.0
    spur_min_length: float = 14.0
    spur_iterations: int = 3
    gap_max_distance: float = 55.0
    gap_max_angle_deg: float = 55.0
    junction_gap_distance: float = 38.0
    collinear_angle_deg: float = 18.0
    min_component_length: float = 18.0
    keep_top_components: int = 16
    min_component_edges: int = 1
    simplify_epsilon: float = 2.5
    collinear_passes: int = 3


def refine_prob_mask(prob: np.ndarray, cfg: NNVectorizeConfig, threshold: float) -> np.ndarray:
    """Hysteresis + smooth + morph close — reconnect broken NN road segments."""
    p = prob.astype(np.float32)
    if cfg.prob_smooth_sigma > 0:
        p = gaussian_filter(p, sigma=cfg.prob_smooth_sigma)

    t_hi = max(threshold, cfg.threshold_high)
    t_lo = min(threshold, cfg.threshold_low, t_hi - 0.05)

    strong = p >= t_hi
    weak = p >= t_lo
    mask = reconstruction(strong.astype(np.uint8), weak.astype(np.uint8)).astype(bool)

    if cfg.open_radius > 0:
        mask = opening(mask, disk(cfg.open_radius))
    if cfg.close_radius > 0:
        mask = closing(mask, disk(cfg.close_radius))

    mask = remove_small_objects(mask, min_size=cfg.min_component_px)
    return mask


def skeleton_from_mask(mask: np.ndarray, cfg: NNVectorizeConfig) -> np.ndarray:
    """Centerline on cleaned NN mask with width band from distance transform."""
    m = mask.astype(bool)
    if m.sum() == 0:
        return m

    dist = distance_transform_edt(m)
    skel = skeletonize(m)
    skel = skel & (dist >= cfg.min_road_radius) & (dist <= cfg.max_road_radius)

    if skel.sum() < 50:
        skel = skeletonize(m)
        skel = skel & (dist >= 1.0)

    # Drop tiny spurs on skeleton before graph extract
    labeled = label(skel)
    out = np.zeros_like(skel)
    for reg in regionprops(labeled):
        if reg.area >= 8:
            out[labeled == reg.label] = True
    return out if out.sum() > 0 else skel


def _merge_collinear_repeated(graph: RoadGraph, cfg: NNVectorizeConfig) -> RoadGraph:
    g = graph
    for _ in range(cfg.collinear_passes):
        g2 = merge_collinear_edges(g, cfg)
        if len(g2.edges) == len(g.edges):
            break
        g = g2
    return g


def vectorize_nn(
    prob: np.ndarray,
    cfg: NNVectorizeConfig | None = None,
    *,
    threshold: float = 0.35,
) -> tuple[RoadGraph, list[tuple[int, int, LineString]], dict, np.ndarray, np.ndarray]:
    """
    NN mask post-process → graph.

    Returns graph, simplified edges, pixel GeoJSON, refined bool mask, skeleton.
    """
    cfg = cfg or NNVectorizeConfig()
    mask = refine_prob_mask(prob, cfg, threshold)
    skel = skeleton_from_mask(mask, cfg)

    graph = extract_graph(skel)
    graph = merge_nearby_nodes(graph, radius=cfg.merge_radius)
    graph = prune_spurs_gentle(graph, cfg)
    graph = filter_road_components(graph, cfg)
    graph = close_gaps(graph, cfg)
    graph = close_junction_gaps(graph, cfg)
    graph = merge_nearby_nodes(graph, radius=cfg.merge_radius)
    graph = _merge_collinear_repeated(graph, cfg)
    graph = remove_small_loops(graph, max_perimeter=80.0)
    graph = prune_spurs_gentle(graph, cfg)
    graph = close_gaps(graph, cfg)
    graph = close_junction_gaps(graph, cfg)
    graph = merge_nearby_nodes(graph, radius=cfg.merge_radius)
    graph = _merge_collinear_repeated(graph, cfg)

    simplified = simplify_edges(graph, epsilon=cfg.simplify_epsilon)
    geojson = to_geojson(simplified)
    return graph, simplified, geojson, mask, skel
