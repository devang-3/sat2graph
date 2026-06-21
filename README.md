# sat2graph

**Place name → satellite image → road probability mask → vector graph (GeoJSON).**

Production path: fetch Esri World Imagery, run a neural road-segmentation model (ONNX), post-process mask into a graph, export WGS84 GeoJSON + Leaflet preview.

---

## Quick start

```bash
cd sat2graph
pip install numpy scikit-image scipy shapely matplotlib pillow requests onnxruntime

# default: DeepLabV3+ ONNX @ 1024²
python3 run_from_place.py --place "Harvard Square, Cambridge MA"

# alternate backend: D-LinkNet ONNX @ 256² resize
python3 run_from_place.py --place "Seattle, WA" --model dlink
```

Outputs → `output/places/<slug>/`:
- `sat.jpg` — fetched mosaic
- `roads.geojson` — graph in lat/lon (EPSG:4326)
- `roads_pixels.geojson` — pixel coords (debug)
- `pipeline.png` — mask + graph viz
- `preview.html` — Leaflet overlay
- `summary.json` — stats (nodes, edges, dead-ends, mask %)

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | `deeplab` | `deeplab` or `dlink` |
| `--zoom` | 18 | ~0.6 m/px mid-lat |
| `--size` | 1024 | Multiple of 256 |
| `--threshold` | 0.35 | Mask binarization (before hysteresis) |
| `--legacy-post` | off | Old Tier C post-process (debug only) |

Imagery: **Esri World Imagery**. Geocoding: **Nominatim** (OSM). Rate-limit bulk requests.

---

## Pipeline

```
place / lat,lon
  → Nominatim geocode
  → Esri tile mosaic (1024×1024)
  → neural prob map (ONNX)
  → postprocess_nn.vectorize_nn  (hysteresis, morph, skeleton, graph)
  → simplify + WGS84 GeoJSON
```

| Stage | File | What |
|-------|------|------|
| Fetch + georef | `geo_imagery.py` | Tiles, bounds, HTML preview |
| Mask backends | `nn/mask_backend.py` | D-Link / DeepLab ONNX load + predict |
| Post-process | `postprocess_nn.py` | `NNVectorizeConfig` → graph |
| Graph primitives | `tier_a.py`, `tier_b.py`, `tier_c.py` | Skeleton, gap close, merge (used by post-process) |
| CLI | `run_from_place.py` | End-to-end orchestrator |

---

## Models (what we tried → what we kept)

### Kept — in production

| Model | File | Why kept |
|-------|------|----------|
| **DeepLabV3+ ONNX** (default) | `models/deeplab_fp32.onnx` | Best on Esri tiles in 20-place eval. Native 1024 + ImageNet norm. ONNX = **2.4× faster** than PyTorch on CPU, identical accuracy. |
| **DeepLab PyTorch** (export only) | `models/best_model.pth` | Source checkpoint for ONNX export (`convert_deeplab_onnx.py`). |
| **D-LinkNet ONNX** (alt backend) | `models/roads_extraction_fp32.onnx` | Faster (~300 ms), strong on Mass-Roads-like NE tiles. `--model dlink`. |

### Removed — and why

| Thing | Why removed |
|-------|-------------|
| **Heuristic tiers A/B/C runners** (`run_dataset*.py`, `compare_tiers.py`) | Explored classical CV masks; too noisy vs neural. Graph code in `tier_*.py` **kept** — post-process still uses it. |
| **Sliding-window infer (default)** | Worse than whole-tile resize for D-LinkNet (trained on 1024→256 squash). Still available as `--sliding` for D-Link only, with warning. |
| **DeepLab 10-epoch retrain** (`best_model_new.pth`) | Underperformed original checkpoint on 20-place compare. Deleted. |
| **Keras H5 + FP4/FP16 quant artifacts** | Benchmark-only; regeneratable via `convert_onnx.py`. Removed from disk to slim repo. |
| **mapunet notebook** | Unused experiment. |
| **knowledge.md / plan.md / next.md** | Consolidated into this README + `../improvements.md`. |
| **All `output/` artifacts** | Regeneratable; gitignored. |

---

## What we learned (20-place compare, Jun 2025)

Same post-process (`NNVectorizeConfig`, thr=0.35) for fair comparison:

| | D-LinkNet | DeepLab (old) |
|--|-----------|---------------|
| Avg mask | 6.0% | **8.1%** |
| Avg infer (CPU) | **318 ms** | ~1100 ms (ONNX) |
| Best on | NE grid cities (Harvard, Cambridge) | West/south Esri tiles (Seattle, Austin, Chicago) |

**Decision:** DeepLab ONNX as default deploy model. D-Link kept as fast fallback for Mass-Roads-domain tiles.

**Open problem:** dead-ends still high on grid cities (~50–70% of nodes in Austin/Chicago). Post-process tuning is next — see `../improvements.md` §2–3.

---

## Eval / benchmark scripts

Not part of deploy pipeline — used to pick models and tune post-process.

```bash
# D-Link vs DeepLab on real places (Esri tiles)
python3 compare_places.py
python3 compare_places.py --places "Boston, MA" "Seattle, WA"

# DeepLab PyTorch vs ONNX speed + accuracy on dataset/
pip install -r requirements-deeplab.txt
python3 convert_deeplab_onnx.py          # pth → ONNX (once)
python3 compare_models.py --mode deeplab-pth-onnx --limit 20

# Batch NN pipeline on Mass Roads dataset/
python3 run_dataset_nn.py --limit 8

# D-Link FP32 → FP16 quant (optional)
python3 convert_onnx.py --only fp16
python3 compare_models.py --mode fp32-fp16 --viz-limit 6
```

---

## Training (retrain / fine-tune)

| Model | Notebook | Export |
|-------|----------|--------|
| D-LinkNet | `notebook/dl-linknet.ipynb` | Kaggle → `.keras` → ONNX (manual) |
| DeepLabV3+ | `notebook/road-extraction-from-satellite-images-deeplabv3.ipynb` | `torch.save(model)` → `convert_deeplab_onnx.py` |

See [`nn/README.md`](./nn/README.md) for D-LinkNet data layout and Kaggle setup.

---

## Project layout

```
sat2graph/
├── run_from_place.py       # ★ main pipeline CLI
├── geo_imagery.py          # geocode + Esri tiles + georef
├── postprocess_nn.py       # NN-tuned mask → graph
├── nn/
│   ├── mask_backend.py     # dlink / deeplab ONNX backends
│   ├── deeplab_inference.py
│   ├── inference.py        # D-LinkNet infer
│   ├── data.py             # training data loader
│   └── dlinknet_model.py
├── tier_a.py / tier_b.py / tier_c.py   # graph geometry (used by postprocess)
├── mask_from_satellite.py  # load_rgb helper for eval scripts
├── compare_places.py       # model compare on fetched places
├── compare_models.py       # ONNX benchmark on dataset/
├── convert_deeplab_onnx.py # DeepLab pth → ONNX
├── convert_onnx.py         # D-Link FP32 → FP16
├── run_dataset_nn.py       # batch eval on ../dataset/
├── notebook/               # training notebooks
├── models/                 # weights (gitignored, see above)
└── output/                 # generated (gitignored)
```

---

## Dependencies

```bash
# pipeline (CPU)
pip install numpy scikit-image scipy shapely matplotlib pillow requests onnxruntime

# DeepLab export / PyTorch compare
pip install -r requirements-deeplab.txt

# D-LinkNet training
pip install -r requirements-nn.txt
```

---

## Roadmap

Detailed improvement notes: [`../improvements.md`](../improvements.md)

Top post-process next steps (from place examples):
1. Prob-path gap bridge before skeleton (Austin, Chicago dead-ends)
2. Adaptive hysteresis for sparse tiles (Houston, Minneapolis)
3. Backend-specific `NNVectorizeConfig` presets (D-Link vs DeepLab)
4. OSM overlay on `preview.html` for ground-truth sanity check
