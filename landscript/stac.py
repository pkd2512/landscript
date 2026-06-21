import requests
import rasterio
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from .config import PipelineConfig

STAC_API = "https://earth-search.aws.element84.com/v1"


def list_scenes(cfg: PipelineConfig) -> List[dict]:
    bbox = cfg.bbox.as_tuple()
    url = f"{STAC_API}/search"
    params = {
        "collections": [cfg.stac_collection],
        "bbox": list(bbox),
        "datetime": f"{cfg.date_start}/{cfg.date_end}",
        "sortby": [{"field": cfg.cloud_field, "direction": "asc"}],
        "limit": 50,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    scenes = []
    for feat in data.get("features", []):
        props = feat["properties"]
        scenes.append({
            "id": feat["id"],
            "date": props.get("datetime", "")[:10],
            "cloud": props.get(cfg.cloud_field, 100),
            "satellite": cfg.satellite,
            "bbox": feat.get("bbox"),
        })
    return scenes


def download_scene(item: dict, out_path: Path, bands: tuple = ("B04", "B03", "B02")) -> Optional[Path]:
    item_id = item["id"]
    search_url = f"{STAC_API}/collections/{item['collection']}/items/{item_id}"

    resp = requests.get(search_url, timeout=30)
    resp.raise_for_status()
    feat = resp.json()
    assets = feat.get("assets", {})

    rgb = []
    for b in bands:
        href = assets.get(b, {}).get("href")
        if not href:
            print(f"  Band {b} not found")
            return None
        rgb.append(href)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    with rasterio.open(rgb[0]) as src:
        profile = src.profile
        width, height = src.width, src.height
        transform = src.transform
        crs = src.crs

    arr = np.zeros((len(bands), height, width), dtype=np.uint16)
    for i, href in enumerate(rgb):
        with rasterio.open(href) as src:
            arr[i] = src.read(1)

    profile.update(count=len(bands), driver="GTiff")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)

    return out_path


def download_best_scenes(cfg: PipelineConfig, max_scenes: int = 3) -> List[Path]:
    scenes = list_scenes(cfg)
    if not scenes:
        print("No scenes found.")
        return []

    print(f"Found {len(scenes)} scenes. Downloading best {max_scenes}...")
    paths = []
    for s in scenes[:max_scenes]:
        fname = f"{cfg.region_name}_{s['date']}_{cfg.satellite}.tif"
        out = cfg.source_dir / fname
        if out.exists():
            print(f"  Already cached: {fname}")
            paths.append(out)
            continue
        print(f"  Downloading {fname} (cloud: {s['cloud']:.0f}%)...")
        result = download_scene(s, out, cfg.rgb_bands)
        if result:
            paths.append(result)
    return paths


def tile_scene(src_path: Path, cfg: PipelineConfig) -> List[Path]:
    out_dir = cfg.tiles_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        width, height = src.width, src.height
        tiles = []
        tid = 0
        for y in range(0, height, cfg.tile_size):
            for x in range(0, width, cfg.tile_size):
                w = min(cfg.tile_size, width - x)
                h = min(cfg.tile_size, height - y)
                if w < cfg.tile_size or h < cfg.tile_size:
                    continue
                window = rasterio.windows.Window(x, y, w, h)
                tile = src.read(window=window)
                tile_path = out_dir / f"{src_path.stem}_tile{tid:04d}.png"
                img = np.moveaxis(tile[:3], 0, -1)
                img = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)
                import cv2
                cv2.imwrite(str(tile_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                tiles.append(tile_path)
                tid += 1
        return tiles


def download_and_tile(cfg: PipelineConfig, max_scenes: int = 3) -> List[Path]:
    paths = download_best_scenes(cfg, max_scenes)
    all_tiles = []
    for p in paths:
        tiles = tile_scene(p, cfg)
        all_tiles.extend(tiles)
    return all_tiles
