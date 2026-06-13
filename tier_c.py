"""Tier C: advanced heuristic mask → vector pipeline (no ML)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, medial_axis, opening, reconstruction, skeletonize

from tier_a import (
    RoadGraph,
    extract_graph,
    merge_nearby_nodes,
    remove_spurs,
    simplify_edges,
    to_geojson,
    _polyline_length,
)
from tier_b import (
    TierBConfig,
    _angle_between,
    _bridge_points,
    _dead_end_nodes,
    _edge_at_node,
    _node_degree,
    _tangent_vector,
    close_gaps,
    graph_stats,
)


@dataclass
class TierCConfig(TierBConfig):
    # mask / skeleton
    min_road_radius: float = 2.0
    max_road_radius: float = 16.0
    min_blob_eccentricity: float = 0.45
    min_blob_area: int = 120

    # graph filtering
    merge_radius: float = 6.0
    spur_min_length: float = 12.0
    spur_iterations: int = 2
    gap_max_distance: float = 45.0
    gap_max_angle_deg: float = 50.0
    junction_gap_distance: float = 25.0
    collinear_angle_deg: float = 15.0
    min_component_length: float = 30.0
    keep_top_components: int = 8
    min_component_edges: int = 2
    simplify_epsilon: float = 3.5


def preprocess_mask_c(mask: np.ndarray, cfg: TierCConfig) -> np.ndarray:
    """Morphological reconstruction + width-aware filtering."""
    m = mask.astype(bool)
    se = disk(2)
    m = opening(m, se)
    m = closing(m, disk(5))
    m = closing(m, disk(8))  # bridge shadow gaps

    # Drop compact blobs (roofs, parking patches) — roads are elongated
    labeled = label(m)
    out = np.zeros_like(m)
    for reg in regionprops(labeled):
        if reg.area < cfg.min_blob_area:
            continue
        if reg.eccentricity >= cfg.min_blob_eccentricity or reg.area > 2500:
            out[labeled == reg.label] = True
    if out.sum() < m.sum() * 0.25:
        out = m  # safety: don't wipe mask
    return out


def width_filtered_skeleton(mask: np.ndarray, cfg: TierCConfig) -> np.ndarray:
    """Medial axis with road-width band; fallback to Zhang-Suen."""
    m = mask.astype(bool)
    dist = distance_transform_edt(m)
    skel, skel_dist = medial_axis(m, return_distance=True)
    skel = skel & (skel_dist >= cfg.min_road_radius) & (skel_dist <= cfg.max_road_radius)

    if skel.sum() < 80:
        skel = skeletonize(m)
        skel = skel & (dist >= 1.5)
    return skel.astype(bool)


def _edge_components(graph: RoadGraph) -> list[list[int]]:
    node_to_edges: dict[int, list[int]] = {}
    for i, (u, v, _) in enumerate(graph.edges):
        node_to_edges.setdefault(u, []).append(i)
        node_to_edges.setdefault(v, []).append(i)

    visited: set[int] = set()
    comps: list[list[int]] = []
    for start in range(len(graph.edges)):
        if start in visited:
            continue
        stack = [start]
        comp: list[int] = []
        while stack:
            e = stack.pop()
            if e in visited:
                continue
            visited.add(e)
            comp.append(e)
            u, v, _ = graph.edges[e]
            for n in (u, v):
                for nb in node_to_edges.get(n, []):
                    if nb not in visited:
                        stack.append(nb)
        comps.append(comp)
    return comps


def prune_spurs_gentle(graph: RoadGraph, cfg: TierCConfig) -> RoadGraph:
    """Length-only spur removal — no aggressive dead-end cascade."""
    g = graph
    for _ in range(cfg.spur_iterations):
        g = remove_spurs(g, min_length=cfg.spur_min_length)
    return g


def filter_road_components(graph: RoadGraph, cfg: TierCConfig) -> RoadGraph:
    """Keep largest road networks by total length."""
    if not graph.edges:
        return graph

    scored = []
    for comp in _edge_components(graph):
        edges = [graph.edges[i] for i in comp]
        length = sum(_polyline_length(pts) for _, _, pts in edges)
        scored.append((length, len(edges), comp))

    scored.sort(reverse=True)
    max_len = scored[0][0] if scored else 0
    min_len = max(cfg.min_component_length, max_len * 0.08)
    keep = [
        c
        for length, n, c in scored[: cfg.keep_top_components]
        if length >= min_len and n >= cfg.min_component_edges
    ]
    if not keep and scored:
        keep = [scored[0][2]]

    keep_idx = {i for c in keep for i in c}
    kept_edges = [graph.edges[i] for i in sorted(keep_idx)]
    used = set()
    for u, v, _ in kept_edges:
        used.add(u)
        used.add(v)
    return RoadGraph(nodes={n: graph.nodes[n] for n in used}, edges=kept_edges)


def close_junction_gaps(graph: RoadGraph, cfg: TierCConfig) -> RoadGraph:
    """Bridge dead-ends toward nearby junction nodes (not only other dead-ends)."""
    nodes = dict(graph.nodes)
    edges = list(graph.edges)

    junctions = [n for n in nodes if _node_degree(RoadGraph(nodes, edges), n) >= 3]
    dead = _dead_end_nodes(RoadGraph(nodes, edges))

    for n1 in dead:
        e1 = _edge_at_node(RoadGraph(nodes, edges), n1)
        if not e1:
            continue
        u1, v1, pts1 = e1
        t1 = _tangent_vector(pts1, from_start=(n1 == u1), k=cfg.tangent_points)
        if t1 == (0.0, 0.0):
            continue
        p1 = nodes[n1]

        best: tuple[float, int, list[tuple[int, int]]] | None = None
        for n2 in junctions:
            if n1 == n2:
                continue
            p2 = nodes[n2]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if dist > cfg.junction_gap_distance or dist < 2:
                continue
            toward = ((p2[0] - p1[0]) / dist, (p2[1] - p1[1]) / dist)
            if _angle_between(t1, toward) > cfg.gap_max_angle_deg:
                continue
            score = dist + _angle_between(t1, toward)
            if best is None or score < best[0]:
                best = (score, n2, _bridge_points(p1, p2))

        if best:
            _, n2, bridge = best
            edges.append((n1, n2, bridge))

    return RoadGraph(nodes=nodes, edges=edges)


def merge_collinear_edges(graph: RoadGraph, cfg: TierCConfig) -> RoadGraph:
    """Merge two edges at degree-2 node if nearly straight."""
    nodes = dict(graph.nodes)
    edges = list(graph.edges)
    changed = True

    while changed:
        changed = False
        g = RoadGraph(nodes=nodes, edges=edges)
        for n in list(nodes):
            if _node_degree(g, n) != 2:
                continue
            incident = [(i, e) for i, e in enumerate(edges) if n in (e[0], e[1])]
            if len(incident) != 2:
                continue
            (i0, (u0, v0, pts0)), (i1, (u1, v1, pts1)) = incident

            def orient(u, v, pts, node):
                if u == node:
                    return pts
                return pts[::-1]

            p0 = orient(u0, v0, pts0, n)
            p1 = orient(u1, v1, pts1, n)
            if len(p0) < 2 or len(p1) < 2:
                continue

            t0 = _tangent_vector(p0, from_start=True, k=cfg.tangent_points)
            t1 = _tangent_vector(p1, from_start=True, k=cfg.tangent_points)
            # tangents point away from shared node; straight road ≈ opposite directions
            angle = _angle_between(t0, (-t1[0], -t1[1]))
            if angle > cfg.collinear_angle_deg:
                continue

            far0 = u0 if u0 != n else v0
            far1 = u1 if u1 != n else v1
            merged = p0 + p1[1:]
            new_edges = [e for j, e in enumerate(edges) if j not in (i0, i1)]
            new_edges.append((far0, far1, merged))
            edges = new_edges
            changed = True
            break

    return RoadGraph(nodes=nodes, edges=edges)


def remove_small_loops(graph: RoadGraph, max_perimeter: float = 60.0) -> RoadGraph:
    """Drop self-loop edges (cycle components with one node)."""
    kept = []
    for u, v, pts in graph.edges:
        if u == v and _polyline_length(pts) < max_perimeter:
            continue
        kept.append((u, v, pts))
    used = set()
    for u, v, _ in kept:
        used.add(u)
        used.add(v)
    return RoadGraph(nodes={n: graph.nodes[n] for n in used}, edges=kept)


def vectorize_c(
    mask: np.ndarray,
    cfg: TierCConfig | None = None,
) -> tuple[RoadGraph, list[tuple[int, int, LineString]], dict]:
    """
  Full Tier C pipeline:
    mask cleanup → width skeleton → graph extract
    → merge → spur prune → component filter
    → gap close (dead-dead + dead-junction) → collinear merge
    → loop removal → second gap pass → simplify
    """
    cfg = cfg or TierCConfig()
    cleaned = preprocess_mask_c(mask, cfg)
    skel = width_filtered_skeleton(cleaned, cfg)

    graph = extract_graph(skel)
    graph = merge_nearby_nodes(graph, radius=cfg.merge_radius)
    graph = prune_spurs_gentle(graph, cfg)
    graph = filter_road_components(graph, cfg)
    graph = close_gaps(graph, cfg)
    graph = close_junction_gaps(graph, cfg)
    graph = merge_nearby_nodes(graph, radius=cfg.merge_radius)
    graph = merge_collinear_edges(graph, cfg)
    graph = remove_small_loops(graph)
    graph = prune_spurs_gentle(graph, cfg)
    graph = close_gaps(graph, cfg)
    graph = merge_nearby_nodes(graph, radius=cfg.merge_radius)

    simplified = simplify_edges(graph, epsilon=cfg.simplify_epsilon)
    return graph, simplified, to_geojson(simplified)
