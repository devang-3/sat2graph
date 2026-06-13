"""Tier B: practical mask → vector pipeline with gap closing and graph cleanup."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev, splprep
from shapely.geometry import LineString, mapping
from skimage.morphology import closing, disk, opening, skeletonize

from tier_a import (
    RoadGraph,
    extract_graph,
    merge_nearby_nodes,
    remove_spurs,
    simplify_edges,
    to_geojson,
    _polyline_length,
)


@dataclass
class TierBConfig:
    morph_radius: int = 2
    merge_radius: float = 5.0
    spur_min_length: float = 18.0
    spur_iterations: int = 3
    gap_max_distance: float = 35.0
    gap_max_angle_deg: float = 45.0
    tangent_points: int = 8
    simplify_epsilon: float = 4.0
    use_splines: bool = False
    spline_points: int = 20


def _node_degree(graph: RoadGraph, node_id: int) -> int:
    deg = 0
    for u, v, _ in graph.edges:
        if u == node_id or v == node_id:
            deg += 1
    return deg


def _dead_end_nodes(graph: RoadGraph) -> list[int]:
    return [n for n in graph.nodes if _node_degree(graph, n) == 1]


def _edge_at_node(
    graph: RoadGraph, node_id: int
) -> tuple[int, int, list[tuple[int, int]]] | None:
    for edge in graph.edges:
        u, v, pts = edge
        if u == node_id or v == node_id:
            return edge
    return None


def _tangent_vector(
    pts: list[tuple[int, int]], *, from_start: bool, k: int
) -> tuple[float, float]:
    """Unit tangent pointing away from dead-end into the edge."""
    if len(pts) < 2:
        return (0.0, 0.0)
    if from_start:
        seg = pts[: min(k, len(pts))]
        r0, c0 = seg[0]
        r1, c1 = seg[-1]
    else:
        seg = pts[-min(k, len(pts)) :]
        r0, c0 = seg[-1]
        r1, c1 = seg[0]
    dr, dc = r1 - r0, c1 - c0
    norm = math.hypot(dr, dc)
    if norm < 1e-6:
        return (0.0, 0.0)
    return (dr / norm, dc / norm)


def _angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def _bridge_points(
    p1: tuple[int, int], p2: tuple[int, int]
) -> list[tuple[int, int]]:
    """Integer line raster between two points."""
    r0, c0 = p1
    r1, c1 = p2
    n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
    if n <= 1:
        return [p1, p2]
    rows = np.linspace(r0, r1, n).round().astype(int)
    cols = np.linspace(c0, c1, n).round().astype(int)
    return list(dict.fromkeys(zip(rows.tolist(), cols.tolist())))


def close_gaps(graph: RoadGraph, cfg: TierBConfig) -> RoadGraph:
    """Connect nearby dead-ends whose tangents face each other."""
    nodes = dict(graph.nodes)
    edges = list(graph.edges)
    next_id = max(nodes.keys(), default=-1) + 1

    dead = _dead_end_nodes(RoadGraph(nodes=nodes, edges=edges))
    used: set[int] = set()

    for i, n1 in enumerate(dead):
        if n1 in used or n1 not in nodes:
            continue
        e1 = _edge_at_node(RoadGraph(nodes=nodes, edges=edges), n1)
        if e1 is None:
            continue
        u1, v1, pts1 = e1
        from_start1 = n1 == u1
        t1 = _tangent_vector(pts1, from_start=from_start1, k=cfg.tangent_points)
        if t1 == (0.0, 0.0):
            continue
        p1 = nodes[n1]

        best: tuple[float, int, list[tuple[int, int]]] | None = None
        for n2 in dead[i + 1 :]:
            if n2 in used or n2 not in nodes or n1 == n2:
                continue
            p2 = nodes[n2]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if dist > cfg.gap_max_distance or dist < 1:
                continue

            e2 = _edge_at_node(RoadGraph(nodes=nodes, edges=edges), n2)
            if e2 is None:
                continue
            u2, v2, pts2 = e2
            from_start2 = n2 == u2
            t2 = _tangent_vector(pts2, from_start=from_start2, k=cfg.tangent_points)
            if t2 == (0.0, 0.0):
                continue

            # tangents should point toward each other
            toward = (
                (p2[0] - p1[0]) / dist,
                (p2[1] - p1[1]) / dist,
            )
            a1 = _angle_between(t1, toward)
            a2 = _angle_between(t2, (-toward[0], -toward[1]))
            if a1 > cfg.gap_max_angle_deg or a2 > cfg.gap_max_angle_deg:
                continue

            bridge = _bridge_points(p1, p2)
            score = dist + a1 + a2
            if best is None or score < best[0]:
                best = (score, n2, bridge)

        if best is not None:
            _, n2, bridge = best
            edges.append((n1, n2, bridge))
            used.add(n1)
            used.add(n2)

    return RoadGraph(nodes=nodes, edges=edges)


def prune_spurs_iterative(graph: RoadGraph, cfg: TierBConfig) -> RoadGraph:
    """Repeatedly remove short dead-end branches."""
    g = graph
    for _ in range(cfg.spur_iterations):
        g = remove_spurs(g, min_length=cfg.spur_min_length)
        dead = _dead_end_nodes(g)
        short_dead = []
        for n in dead:
            e = _edge_at_node(g, n)
            if e and _polyline_length(e[2]) < cfg.spur_min_length * 1.5:
                short_dead.append(n)
        if not short_dead:
            break
        keep = [
            (u, v, pts)
            for u, v, pts in g.edges
            if not (
                (u in short_dead and _node_degree(g, u) == 1)
                or (v in short_dead and _node_degree(g, v) == 1)
            )
        ]
        used = set()
        for u, v, _ in keep:
            used.add(u)
            used.add(v)
        g = RoadGraph(
            nodes={n: g.nodes[n] for n in used},
            edges=keep,
        )
    return g


def fit_spline(coords: list[tuple[float, float]], n: int) -> LineString:
    """Cubic B-spline through polyline coords (col, row)."""
    if len(coords) < 4:
        return LineString(coords)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    tck, _ = splprep([xs, ys], s=0, k=3)
    u = np.linspace(0, 1, n)
    x, y = splev(u, tck)
    return LineString(list(zip(x.tolist(), y.tolist())))


def simplify_with_splines(
    graph: RoadGraph, cfg: TierBConfig
) -> list[tuple[int, int, LineString]]:
    lines = simplify_edges(graph, epsilon=cfg.simplify_epsilon)
    if not cfg.use_splines:
        return lines
    out = []
    for u, v, line in lines:
        coords = list(line.coords)
        if len(coords) >= 4:
            line = fit_spline(coords, cfg.spline_points)
        out.append((u, v, line))
    return out


def preprocess_mask_b(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Stronger morphology before skeleton — bridge small mask gaps."""
    m = mask.astype(bool)
    se = disk(radius)
    m = opening(m, se)
    m = closing(m, se)
    # extra closing pass for road gaps (shadows / trees)
    m = closing(m, disk(radius + 1))
    return m


def vectorize_b(
    mask: np.ndarray,
    cfg: TierBConfig | None = None,
) -> tuple[RoadGraph, list[tuple[int, int, LineString]], dict]:
    """Full Tier B geometry pipeline."""
    cfg = cfg or TierBConfig()
    cleaned = preprocess_mask_b(mask, radius=cfg.morph_radius)
    skel = skeletonize(cleaned)

    graph = extract_graph(skel)
    graph = merge_nearby_nodes(graph, radius=cfg.merge_radius)
    graph = prune_spurs_iterative(graph, cfg)
    graph = close_gaps(graph, cfg)
    graph = merge_nearby_nodes(graph, radius=cfg.merge_radius)
    graph = prune_spurs_iterative(graph, cfg)

    simplified = simplify_with_splines(graph, cfg)
    geojson = to_geojson(simplified)
    return graph, simplified, geojson


def graph_stats(graph: RoadGraph, simplified: list) -> dict:
    """Summary metrics for comparison."""
    dead = len(_dead_end_nodes(graph))
    lengths = [_polyline_length(pts) for _, _, pts in graph.edges]
    simp_lengths = [line.length for _, _, line in simplified]
    return {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "dead_ends": dead,
        "total_length_px": float(sum(lengths)),
        "simplified_length_px": float(sum(simp_lengths)),
        "mean_edge_length_px": float(np.mean(lengths)) if lengths else 0.0,
        "simplified_edges": len(simplified),
    }
