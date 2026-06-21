import ee
import requests
import numpy as np
from pathlib import Path
from typing import Optional
from .config import PipelineConfig


def initialize(project: str = "", auth_mode: str = "colab") -> bool:
    try:
        if auth_mode == "colab":
            ee.Authenticate()
        ee.Initialize(project=project or None)
        return True
    except Exception as e:
        print(f"GEE init failed: {e}")
        return False


def get_sentinel_collection(cfg: PipelineConfig) -> ee.ImageCollection:
    bbox = cfg.bbox
    region = ee.Geometry.Rectangle([bbox.min_lon, bbox.min_lat,
                                     bbox.max_lon, bbox.max_lat])

    collection = (
        ee.ImageCollection(cfg.gee_collection)
        .filterBounds(region)
        .filterDate(cfg.date_start, cfg.date_end)
        .filter(ee.Filter.lt(cfg.cloud_field, cfg.cloud_cover_max))
        .sort(cfg.cloud_field)
    )
    return collection


def export_rgb_tile(
    image: ee.Image,
    region: ee.Geometry,
    filepath: Path,
    cfg: PipelineConfig,
) -> Optional[Path]:
    bands = list(cfg.rgb_bands)
    url = image.select(bands).getDownloadURL({
        "region": region,
        "scale": cfg.gee_scale,
        "format": "GEO_TIFF",
        "bands": bands,
    })

    resp = requests.get(url, timeout=120)
    if resp.status_code != 200:
        print(f"Download failed: {resp.status_code}")
        return None

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_bytes(resp.content)
    return filepath


def list_scenes(cfg: PipelineConfig) -> list[dict]:
    collection = get_sentinel_collection(cfg)
    size = collection.size().getInfo()
    if size == 0:
        print("No scenes found.")
        return []

    scenes = collection.limit(50).getInfo()["features"]
    result = []
    for s in scenes:
        props = s["properties"]
        result.append({
            "id": s["id"],
            "date": props.get("SENSING_TIME", props.get("DATE_ACQUIRED", "")),
            "cloud": props.get(cfg.cloud_field, 100),
            "satellite": cfg.satellite,
        })
    return result
