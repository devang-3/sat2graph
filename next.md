# next.md — What to try next (no neural networks)

Tier A/B/C all fail on `dataset/` because **mask step is wrong** more often than graph step. Geometry heuristics help noise, but can't invent roads that were never segmented.

Below: classical / algorithmic ideas worth trying next, ordered roughly by impact vs effort.

---

## 1. Fix the mask (highest leverage)

Current masks use Otsu + HSV + morphology. Roads on your tiles share appearance with roofs, concrete, bare earth, shadows.

### A. Line / ridge detectors (road ≈ long thin structure)

| Method | Idea | Library |
|--------|------|---------|
| **Canny + probabilistic Hough** | Edge map → line segments → rasterize centerlines | OpenCV `HoughLinesP` |
| **LSD** (Line Segment Detector) | Direct segment find in gray | OpenCV `createLineSegmentDetector` |
| **Frangi vesselness** | Multi-scale ridge filter — finds curvilinear structures | `skimage.filters.frangi` or custom |
| **Steerable filters / Gabor bank** | Respond to bar-shaped textures at several orientations | `scipy.ndimage` convolve |
| **Directional morphology** | Open/close with **line** structuring elements at 0°, 45°, 90°, 135° | `skimage.morphology` |

**Pipeline sketch:**
```
gray → Frangi or multi-angle line enhance → threshold → thin → graph
```
Bypasses “is this pixel asphalt color?” — uses **shape**.

### B. Graph-cut / energy segmentation

- Build grid graph per pixel; data term from color distance to **road vs non-road** samples; smoothness term penalizes label changes.
- Road/non-road color models from **auto-clustering** (k-means in Lab) or manual picks per tile.
- `skimage.segmentation` + custom energy, or `PyMaxflow`.

### C. Watershed on distance / gradient

- Compute **morphological gradient** or **Sobel** magnitude.
- Mark sure-road seeds (long thin components from line detector) + sure-background (vegetation mask).
- `skimage.segmentation.watershed` with markers.

### D. GrabCut-style refinement

- Initial mask from any tier → OpenCV `grabCut` with probable foreground/background.
- Refines boundaries around already-rough road blobs.

### E. Superpixel + hand-crafted classify

- **SLIC** superpixels (`skimage.segmentation.slic`).
- Per superpixel features: mean Lab, variance, eccentricity, length/width, Frangi max response.
- Rule tree or logistic regression (still not a CNN — linear model on features).
- Merge adjacent “road” superpixels.

---

## 2. Better centerlines (after mask improves)

| Method | Why |
|--------|-----|
| **Distance transform ridges** | `distance_transform_edt` → find ridges where dist is locally max — smoother than Zhang-Suen on noisy masks |
| **Steger / ridge tracing** | Sub-pixel centerline on Hessian eigenvalues — used in fingerprint / vessel literature |
| **Potrace-style contour tracing** | Trace boundaries then dual graph — good for clean masks |
| **momepy** | `momepy.centerline` / street network tools from Shapely polygons — if mask → road polygons first |

Tier C medial axis is a start; try **ridge of distance transform** instead of binary medial_axis on messy masks.

---

## 3. Graph-level optimization (mask still rough)

Once you have a noisy skeleton, treat road finding as **network inference**:

| Method | Idea |
|--------|------|
| **Shortest-path tree from saliency** | Cost map = 1 − road_probability; connect components via min-cost paths on image grid |
| **Minimum spanning tree** on junction candidates | Seed high-confidence junctions, MST with edge costs from image |
| **Prune by road width consistency** | Walk skeleton; split edge where distance-transform width jumps |
| **Angle + length MRF** | Junction types (L, T, X) have priors; reject spur configs with low likelihood |
| **Global collinearity** | Hough votes on edge directions — snap nearly-parallel fragments |
| **momepy graph cleaning** | `momepy` prune stubs, consolidate intersections on `GeoDataFrame` |

### Gap closing upgrades

Current: dead-end ↔ dead-end and dead-end ↔ junction with tangent check.

Try:
- Dead-end ↔ **any point on nearby edge** (not just nodes) — closest point on segment + angle test.
- **Dijkstra** bridge: path must stay in high-confidence mask band.
- **Template matching** for T/X junctions in skeleton patches.

---

## 4. Multi-scale & per-tile adaptation

Your tiles vary (bright suburbs vs dark forest roads). Fixed thresholds hurt.

| Approach | How |
|----------|-----|
| **Tile stats** | Estimate `road_brightness_mode` from histogram peaks; switch bright vs dark pipeline automatically (Tier C started this — go further) |
| **Multi-scale masks** | Run mask at 512 / 1024 / 2048; OR only where 2 scales agree |
| **Quadtree split** | Process 256×256 sub-tiles with local Otsu; stitch with overlap |
| **Auto-tune** | Grid-search `spur`, `gap_dist`, `epsilon` per tile to maximize skeleton connectivity / minimize dead-end ratio |

---

## 5. Use structure in the data you have

`dataset/` has **203 tiles** but **no mask labels**. Still usable without NN:

| Idea | How |
|------|-----|
| **Self-consistency** | Roads persist across slight blur/threshold jitter — keep stable pixels |
| **Cross-tile stitching** | If tiles are georeferenced and adjacent, graph nodes on borders should align — use as constraint |
| **Voting across tiers** | Pixel = road only if **2 of A/B/C** agree AND passes line detector — precision ↑ |
| **Manual sparse labels** | Label 5–10 tiles in Paint → k-means color model or GrabCut foreground model for rest |

---

## 6. Post-vector geometry (polish, not discovery)

After graph is roughly right:

| Method | Use |
|--------|-----|
| **Douglas–Peucker** | Already used — tune ε per edge length |
| **Visvalingam–Whyatt** | Better shape hierarchy than DP for bends |
| **B-spline / clothoid fit** | Smooth for export; doesn't fix topology |
| **Snap to straight** | If segment R² > 0.98 on window → exact line (highway blocks) |
| **OSM-free topology rules** | Degree-4+ rare → merge close junctions; dead-end cap at tile border |

---

## 7. Suggested implementation order

```
Step 1  Line/Frangi mask branch     → compare to Tier B mask on 10 tiles
Step 2  OR-vote: (Tier B mask) AND (line mask)   → precision filter
Step 3  Dijkstra gap bridge on distance map      → fix broken roads
Step 4  Dead-end ↔ edge gap close                → better than node-only
Step 5  momepy or custom MST pruning             → global network shape
Step 6  Per-tile auto-threshold from histogram   → reduce tile-to-tile variance
```

---

## 8. What probably won't help much (without labels)

- More morphology tuning alone
- Larger gap-close distance (connects wrong things)
- Stronger spur prune (Tier C already over-prunes)
- B-splines / prettier curves
- Running on external road datasets (you want `dataset/` only)

---

## 9. Minimal experiment for next session

Pick **one** image where Tier B finds roads but with noise (e.g. `100393_sat.jpg`):

1. Add `mask_line_frangi.py` — Frangi + Hough lines → binary centerline mask.
2. `mask = tier_b_mask & dilate(line_mask)` — keep B only where line structure exists.
3. Run existing `vectorize_b` on result.
4. Compare in `compare_tiers.py` as **Tier D**.

If precision improves on 5–10 tiles, invest in Step 3 (Dijkstra bridging). If not, try watershed + SLIC (Section 1B/1E).

---

## References (concepts)

- Frangi et al. — multiscale vesselness filter
- Douglas–Peucker / Visvalingam — polyline simplification
- momepy docs — urban street network from geometry
- Steger — subpixel line detection
- Massachusetts Roads dataset paper — typical road width / appearance stats

See also `knowledge.md` for full algorithm map.
