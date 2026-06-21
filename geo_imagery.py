"""Fetch georeferenced satellite mosaics (Nominatim + Esri World Imagery)."""

from __future__ import annotations

import io
import math
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import requests
from PIL import Image

USER_AGENT = "sat2graph/1.0 (road-graph demo; https://github.com/sat2graph)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
ESRI_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
TILE_SIZE = 256


@dataclass
class GeocodeResult:
    lat: float
    lon: float
    display_name: str
    place_id: int | None = None


@dataclass
class MosaicMeta:
    width: int
    height: int
    west: float
    south: float
    east: float
    north: float
    zoom: int
    center_lat: float
    center_lon: float
    source: str = "Esri World Imagery"
    place_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def slugify(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return s[:max_len] or "place"


def geocode_place(query: str, timeout: float = 30.0) -> GeocodeResult:
    """Place name / address → lat, lon via OpenStreetMap Nominatim (free, rate-limited)."""
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": query, "format": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise ValueError(f"No results for: {query!r}")
    row = rows[0]
    return GeocodeResult(
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        display_name=row.get("display_name", query),
        place_id=int(row["place_id"]) if row.get("place_id") else None,
    )


def latlon_to_tile_xy(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_xy_to_latlon(x: float, y: float, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    return math.degrees(lat_rad), lon


def pixel_to_latlon(col: float, row: float, meta: MosaicMeta) -> tuple[float, float]:
    """Image pixel (col, row) → (lat, lon). Row 0 = north."""
    lon = meta.west + (col / meta.width) * (meta.east - meta.west)
    lat = meta.north - (row / meta.height) * (meta.north - meta.south)
    return lat, lon


def latlon_to_pixel(lat: float, lon: float, meta: MosaicMeta) -> tuple[float, float]:
    col = (lon - meta.west) / (meta.east - meta.west) * meta.width
    row = (meta.north - lat) / (meta.north - meta.south) * meta.height
    return col, row


def _fetch_tile(z: int, x: int, y: int, session: requests.Session, cache: dict) -> np.ndarray:
    key = (z, x, y)
    if key in cache:
        return cache[key]
    n = 2**z
    x = x % n
    y = max(0, min(y, n - 1))
    url = ESRI_TILE_URL.format(z=z, y=y, x=x)
    resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    cache[key] = arr
    time.sleep(0.05)  # polite delay for tile server
    return arr


def fetch_mosaic(
    lat: float,
    lon: float,
    *,
    zoom: int = 18,
    size_px: int = 1024,
    place_name: str | None = None,
) -> tuple[np.ndarray, MosaicMeta]:
    """
    Download Esri World Imagery tiles centered on (lat, lon).
    Returns RGB uint8 array and georeferencing metadata.
    """
    if size_px < TILE_SIZE or size_px % TILE_SIZE != 0:
        raise ValueError(f"size_px must be a multiple of {TILE_SIZE}, got {size_px}")

    fx, fy = latlon_to_tile_xy(lat, lon, zoom)
    half = size_px / (2 * TILE_SIZE)
    x0 = fx - half
    y0 = fy - half

    x_start = int(math.floor(x0))
    y_start = int(math.floor(y0))
    x_end = int(math.ceil(x0 + size_px / TILE_SIZE))
    y_end = int(math.ceil(y0 + size_px / TILE_SIZE))

    session = requests.Session()
    cache: dict[tuple[int, int, int], np.ndarray] = {}
    rows = []
    for ty in range(y_start, y_end):
        row_tiles = []
        for tx in range(x_start, x_end):
            row_tiles.append(_fetch_tile(zoom, tx, ty, session, cache))
        rows.append(np.concatenate(row_tiles, axis=1))
    stitched = np.concatenate(rows, axis=0)

    off_x = int(round((x0 - x_start) * TILE_SIZE))
    off_y = int(round((y0 - y_start) * TILE_SIZE))
    mosaic = stitched[off_y : off_y + size_px, off_x : off_x + size_px]

    west_lon = tile_xy_to_latlon(x0, y0, zoom)[1]
    north_lat = tile_xy_to_latlon(x0, y0, zoom)[0]
    east_lon = tile_xy_to_latlon(x0 + size_px / TILE_SIZE, y0 + size_px / TILE_SIZE, zoom)[1]
    south_lat = tile_xy_to_latlon(x0 + size_px / TILE_SIZE, y0 + size_px / TILE_SIZE, zoom)[0]

    meta = MosaicMeta(
        width=size_px,
        height=size_px,
        west=west_lon,
        south=south_lat,
        east=east_lon,
        north=north_lat,
        zoom=zoom,
        center_lat=lat,
        center_lon=lon,
        place_name=place_name,
    )
    return mosaic, meta


def georef_geojson(pixel_fc: dict, meta: MosaicMeta) -> dict:
    """Convert pixel-space LineString GeoJSON → WGS84 (EPSG:4326)."""
    features = []
    for feat in pixel_fc.get("features", []):
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        props = dict(feat.get("properties") or {})
        props["crs"] = "EPSG:4326"

        if gtype == "LineString":
            coords = [
                [pixel_to_latlon(col, row, meta)[1], pixel_to_latlon(col, row, meta)[0]]
                for col, row in geom["coordinates"]
            ]
            new_geom = {"type": "LineString", "coordinates": coords}
        elif gtype == "MultiLineString":
            lines = []
            for line in geom["coordinates"]:
                lines.append(
                    [
                        [pixel_to_latlon(col, row, meta)[1], pixel_to_latlon(col, row, meta)[0]]
                        for col, row in line
                    ]
                )
            new_geom = {"type": "MultiLineString", "coordinates": lines}
        else:
            new_geom = geom

        features.append({"type": "Feature", "geometry": new_geom, "properties": props})

    return {
        "type": "FeatureCollection",
        "properties": {
            "crs": "EPSG:4326",
            "source_imagery": meta.source,
            "place": meta.place_name,
            "bounds": [meta.west, meta.south, meta.east, meta.north],
            "zoom": meta.zoom,
        },
        "features": features,
    }


def write_map_preview(
    out_path: str | Any,
    meta: MosaicMeta,
    geojson_wgs84: dict,
    *,
    sat_jpg_rel: str = "sat.jpg",
) -> None:
    """Write standalone Leaflet HTML preview (no extra deps)."""
    center_lat = (meta.north + meta.south) / 2
    center_lon = (meta.west + meta.east) / 2
    bounds = f"[[{meta.south}, {meta.west}], [{meta.north}, {meta.east}]]"
    import json
    from pathlib import Path

    gj = json.dumps(geojson_wgs84)
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>sat2graph — {meta.place_name or "road graph"}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>html, body, #map {{ height: 100%; margin: 0; }}</style>
</head>
<body>
  <div id="map"></div>
  <script>
    const map = L.map('map').setView([{center_lat}, {center_lon}], {meta.zoom});
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{ attribution: 'Esri World Imagery', maxZoom: 20 }}
    ).addTo(map);
    L.imageOverlay('{sat_jpg_rel}', {bounds}, {{ opacity: 0.85 }}).addTo(map);
    const gj = {gj};
    L.geoJSON(gj, {{
      style: {{ color: '#00ff66', weight: 3, opacity: 0.9 }}
    }}).addTo(map);
    map.fitBounds({bounds});
  </script>
</body>
</html>
"""
    Path(out_path).write_text(html, encoding="utf-8")
