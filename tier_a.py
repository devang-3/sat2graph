"""sat2graph — Tier A: minimal mask → vector road pipeline (CPU)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from shapely.geometry import LineString, mapping
from skimage.morphology import closing, disk, opening, skeletonize

# 8-connected neighbor offsets (row, col)
_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class RoadGraph:
    """Simple road network: nodes are (row, col), edges are polylines."""

    nodes: dict[int, tuple[int, int]] = field(default_factory=dict)
    edges: list[tuple[int, int, list[tuple[int, int]]]] = field(default_factory=list)


def preprocess_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Opening/closing to drop noise and fill tiny holes."""
    m = mask.astype(bool)
    se = disk(radius)
    m = opening(m, se)
    m = closing(m, se)
    return m


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask.astype(bool))


def _neighbor_count(skel: np.ndarray, r: int, c: int) -> int:
    h, w = skel.shape
    count = 0
    for dr, dc in _NEIGHBORS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and skel[nr, nc]:
            count += 1
    return count


def _is_node(skel: np.ndarray, r: int, c: int) -> bool:
    n = _neighbor_count(skel, r, c)
    return n != 2  # terminals (1) and junctions (>=3)


def _neighbors(skel: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = skel.shape
    out = []
    for dr, dc in _NEIGHBORS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and skel[nr, nc]:
            out.append((nr, nc))
    return out


def extract_graph(skel: np.ndarray) -> RoadGraph:
    """Classify nodes by 3×3 degree; trace edges along degree-2 pixels."""
    skel = skel.astype(bool)
    h, w = skel.shape
    node_at: dict[tuple[int, int], int] = {}
    nodes: dict[int, tuple[int, int]] = {}
    nid = 0

    for r in range(h):
        for c in range(w):
            if skel[r, c] and _is_node(skel, r, c):
                node_at[(r, c)] = nid
                nodes[nid] = (r, c)
                nid += 1

    visited_edge: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    edges: list[tuple[int, int, list[tuple[int, int]]]] = []

    for start, u in node_at.items():
        for nxt in _neighbors(skel, *start):
            key = (start, nxt) if start <= nxt else (nxt, start)
            if key in visited_edge:
                continue
            path = [start, nxt]
            prev, cur = start, nxt
            while cur not in node_at:
                nbrs = [p for p in _neighbors(skel, *cur) if p != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                path.append(nxt)
                prev, cur = cur, nxt
            if cur in node_at and cur != start:
                u_id, v_id = node_at[start], node_at[cur]
                edges.append((u_id, v_id, path))
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    ek = (a, b) if a <= b else (b, a)
                    visited_edge.add(ek)

    used_pixels = set()
    for _, _, path in edges:
        used_pixels.update(path)

    # Pure cycles: all skeleton pixels have degree 2 → no junctions detected above
    visited_comp: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if not skel[r, c] or (r, c) in used_pixels or (r, c) in visited_comp:
                continue
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                p = stack.pop()
                if p in visited_comp:
                    continue
                visited_comp.add(p)
                comp.append(p)
                for nbr in _neighbors(skel, *p):
                    if nbr not in used_pixels and nbr not in visited_comp:
                        stack.append(nbr)
            if not comp or any(_neighbor_count(skel, *p) != 2 for p in comp):
                continue
            start = min(comp)  # deterministic break point
            nbrs = _neighbors(skel, *start)
            if not nbrs:
                continue
            path = [start]
            prev, cur = start, nbrs[0]
            path.append(cur)
            while cur != start:
                nxt_opts = [p for p in _neighbors(skel, *cur) if p != prev]
                if not nxt_opts:
                    break
                nxt = nxt_opts[0]
                path.append(nxt)
                prev, cur = cur, nxt
                if len(path) > len(comp) + 2:
                    break
            if cur == start and len(path) > 3:
                node_at[start] = nid
                nodes[nid] = start
                edges.append((nid, nid, path))
                used_pixels.update(path)
                nid += 1

    return RoadGraph(nodes=nodes, edges=edges)


def merge_nearby_nodes(graph: RoadGraph, radius: float = 3.0) -> RoadGraph:
    """Merge junction clusters into single nodes."""
    ids = list(graph.nodes)
    if not ids:
        return graph

    parent = {i: i for i in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in ids:
        r0, c0 = graph.nodes[i]
        for j in ids:
            if j <= i:
                continue
            r1, c1 = graph.nodes[j]
            if np.hypot(r1 - r0, c1 - c0) <= radius:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in ids:
        clusters.setdefault(find(i), []).append(i)

    new_nodes: dict[int, tuple[int, int]] = {}
    old_to_new: dict[int, int] = {}
    for new_id, members in enumerate(clusters.values()):
        rs = [graph.nodes[m][0] for m in members]
        cs = [graph.nodes[m][1] for m in members]
        new_nodes[new_id] = (int(round(np.mean(rs))), int(round(np.mean(cs))))
        for m in members:
            old_to_new[m] = new_id

    seen: set[tuple[int, int]] = set()
    new_edges: list[tuple[int, int, list[tuple[int, int]]]] = []
    for u, v, pts in graph.edges:
        nu, nv = old_to_new[u], old_to_new[v]
        key = (min(nu, nv), max(nu, nv), nu == nv)
        if key in seen:
            continue
        seen.add(key)
        new_edges.append((nu, nv, pts))

    return RoadGraph(nodes=new_nodes, edges=new_edges)


def _polyline_length(pts: list[tuple[int, int]]) -> float:
    total = 0.0
    for i in range(len(pts) - 1):
        r0, c0 = pts[i]
        r1, c1 = pts[i + 1]
        total += np.hypot(r1 - r0, c1 - c0)
    return total


def remove_spurs(graph: RoadGraph, min_length: float = 5.0) -> RoadGraph:
    """Drop short edges; remove orphaned nodes."""
    kept = [
        (u, v, pts)
        for u, v, pts in graph.edges
        if _polyline_length(pts) >= min_length
    ]
    used_nodes = set()
    for u, v, _ in kept:
        used_nodes.add(u)
        used_nodes.add(v)
    nodes = {nid: graph.nodes[nid] for nid in used_nodes}
    return RoadGraph(nodes=nodes, edges=kept)


def simplify_edges(
    graph: RoadGraph, epsilon: float = 2.0
) -> list[tuple[int, int, LineString]]:
    """Douglas-Peucker via Shapely (x=col, y=row for GeoJSON convention)."""
    out = []
    for u, v, pts in graph.edges:
        # GeoJSON order: (x, y) = (col, row)
        coords = [(c, r) for r, c in pts]
        line = LineString(coords).simplify(epsilon, preserve_topology=True)
        out.append((u, v, line))
    return out


def to_geojson(
    simplified: Iterable[tuple[int, int, LineString]],
    *,
    properties: dict | None = None,
) -> dict:
    features = []
    for u, v, line in simplified:
        feat = {
            "type": "Feature",
            "geometry": mapping(line),
            "properties": {
                "u": u,
                "v": v,
                **(properties or {}),
            },
        }
        features.append(feat)
    return {"type": "FeatureCollection", "features": features}


def vectorize(
    mask: np.ndarray,
    *,
    morph_radius: int = 1,
    spur_min_length: float = 5.0,
    simplify_epsilon: float = 2.0,
) -> tuple[RoadGraph, list[tuple[int, int, LineString]], dict]:
    """Full Tier A pipeline."""
    cleaned = preprocess_mask(mask, radius=morph_radius)
    skel = skeletonize_mask(cleaned)
    graph = extract_graph(skel)
    graph = merge_nearby_nodes(graph, radius=3.0)
    graph = remove_spurs(graph, min_length=spur_min_length)
    simplified = simplify_edges(graph, epsilon=simplify_epsilon)
    geojson = to_geojson(simplified)
    return graph, simplified, geojson


def save_geojson(geojson: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)
