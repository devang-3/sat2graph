# sat2graph

**Satellite imagery → vector road graphs.** Tiny segmentation + classical geometry. Edge-ready.

Heuristic tiers (A/B/C) turn aerial RGB tiles into road network graphs (GeoJSON). Planned: smallest trainable mask model for fast edge inference, then same graph vectorization stack.

Tested on `dataset/` (`*_sat.jpg`, 1024×1024, Mass-Roads-style tiles).

## Quick start

```bash
cd sat2graph

# Compare all three tiers side-by-side
python3 compare_tiers.py --input ../dataset --limit 8 --out output/compare_abc

# Run one tier on a folder
python3 run_dataset.py --input ../dataset --limit 10      # Tier A
python3 run_dataset_b.py --input ../dataset --limit 10    # Tier B
python3 run_dataset_c.py --input ../dataset --limit 10    # Tier C

# Synthetic sanity check (no images needed)
python3 run_tier_a.py
```

Outputs: `*.geojson`, debug `*.png`, `comparison_abc.json`.

---

## Pipeline

```
RGB satellite tile → road mask → skeleton / centerline → graph (nodes + edges) → simplify → GeoJSON
```

| Stage | Methods |
|-------|---------|
| **Mask** | Heuristic color/threshold (A/B/C) — tiny NN planned |
| **Skeleton** | Zhang-Suen (`skeletonize`) or medial axis (Tier C) |
| **Graph** | 3×3 degree: dead-end / junction / edge trace |
| **Cleanup** | Gap closing, spur prune, merge (Tier B/C) |
| **Export** | Douglas–Peucker → GeoJSON LineStrings |

Core graph code: `tier_a.py`. Tier B/C add mask + graph repair.

---

## Tier A — Minimal baseline

**Files:** `tier_a.py`, `extract_road_mask`, `run_dataset.py`

| Mask | Otsu + HSV (low saturation, mid brightness), morphology |
| Geometry | Zhang-Suen skeleton → graph → merge nodes (3 px) → spur prune → DP simplify |

Defaults on `dataset/`: `--spur 20 --epsilon 4`

---

## Tier B — Practical heuristics

**Files:** `tier_b.py`, `extract_road_mask_b`, `run_dataset_b.py`

| Mask | Local adaptive threshold + white top-hat + Otsu fusion + Gaussian smooth |
| Geometry | + stronger close, iterative spur prune, **dead-end gap closing** (distance + tangent) |

---

## Tier C — Advanced heuristics

**Files:** `tier_c.py`, `extract_road_mask_c`, `run_dataset_c.py`

| Mask | 2-of-3 ensemble vote, vegetation reject, morphological reconstruction, shape filter |
| Geometry | + medial-axis skeleton, component filter, dead↔junction gaps, collinear merge, loop removal |

---

## Results on `dataset/` (6-tile sample)

| Tier | Avg nodes | Avg edges | Avg dead-ends |
|------|-----------|-----------|---------------|
| A | ~809 | ~646 | ~431 |
| B | ~444 | ~460 | ~134 |
| C | ~80 | ~76 | ~33 |

Mask quality limits all tiers today — see `next.md` for classical improvements and tiny-model path.

---

## Project layout

```
sat2graph/
├── tier_a.py              # Graph extract + Tier A vectorize
├── tier_b.py              # Gap closing + iterative spur prune
├── tier_c.py              # Medial axis + component filter + collinear merge
├── mask_from_satellite.py # Tier A/B/C mask heuristics
├── compare_tiers.py         # A vs B vs C comparison
├── run_dataset.py         # Batch Tier A
├── run_dataset_b.py       # Batch Tier B
├── run_dataset_c.py       # Batch Tier C
├── run_tier_a.py          # Synthetic mask tests
├── synthetic_masks.py     # Cross / T-junction / loop test masks
├── knowledge.md           # Algorithm reference
├── plan.md                # Architecture notes
└── next.md                # What to try next
```

Place satellite tiles in `../dataset/` (or pass `--input`).

---

## Dependencies

```bash
pip install numpy scikit-image scipy shapely matplotlib pillow
```

---

## Docs

- [knowledge.md](./knowledge.md) — algorithms, curves, graphs
- [plan.md](./plan.md) — original pipeline design
- [next.md](./next.md) — roadmap (line detectors, tiny seg, Dijkstra gaps)
