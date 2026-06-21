import time
import requests
import rasterio
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from .config import PipelineConfig

STAC_API = "https://earth-search.aws.element84.com/v1"


def list_scenes(cfg: PipelineConfig) -> List[dict]:
    log("STAC", "Searching for scenes...")
    bbox = cfg.bbox.as_tuple()
    url = f"{STAC_API}/search"
    params = {
        "collections": [cfg.stac_collection],
        "bbox": list(bbox),
        "datetime": f"{cfg.date_start}/{cfg.date_end}",
        "sortby": [{"field": cfg.cloud_field, "direction": "asc"}],
        "limit": 50,
    }
    t0 = time.time()
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    log("STAC", f"Query returned in {time.time()-t0:.1f}s")

    scenes = []
    for feat in data.get("features", []):
        props = feat["properties"]
        scenes.append({
            "id": feat["id"],
            "date": props.get("datetime", "")[:10],
            "cloud": props.get(cfg.cloud_field, 100),
            "satellite": cfg.satellite,
            "bbox": feat.get("bbox"),
            "collection": feat.get("collection", cfg.stac_collection),
        })
    return scenes


def download_scene(item: dict, out_path: Path, cfg: PipelineConfig) -> Optional[Path]:
    bands = cfg.rgb_bands
    log("STAC", f"Resolving asset URLs for {item['date']}...")
    search_url = f"{STAC_API}/collections/{item['collection']}/items/{item['id']}"

    resp = requests.get(search_url, timeout=30)
    resp.raise_for_status()
    feat = resp.json()
    assets = feat.get("assets", {})

    rgb = []
    for b in bands:
        href = assets.get(b, {}).get("href")
        if not href:
            log("STAC", f"Band {b} not found, skipping")
            return None
        rgb.append(href)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        log("STAC", f"Already cached: {out_path.name}")
        return out_path

    log("STAC", "Reading band 1/3 (detecting size)...")
    with rasterio.open(rgb[0]) as src:
        profile = src.profile
        width, height = src.width, src.height

    arr = np.zeros((len(bands), height, width), dtype=np.uint16)
    for i, href in enumerate(tqdm(rgb, desc="  Bands", unit="band")):
        with rasterio.open(href) as src:
            arr[i] = src.read(1)

    log("STAC", f"Writing GeoTIFF ({width}x{height})...")
    profile.update(count=len(bands), driver="GTiff")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)

    return out_path


def download_best_scenes(cfg: PipelineConfig, max_scenes: int = 3) -> List[Path]:
    scenes = list_scenes(cfg)
    if not scenes:
        log("STAC", "No scenes found.")
        return []

    log("STAC", f"{len(scenes)} scenes available, fetching best {max_scenes}")
    paths = []
    for s in scenes[:max_scenes]:
        fname = f"{cfg.region_name}_{s['date']}_{cfg.satellite}.tif"
        out = cfg.source_dir / fname
        log("STAC", f"Scene: {s['date']} | cloud: {s['cloud']:.0f}%")
        result = download_scene(s, out, cfg)
        if result:
            paths.append(result)
            log("STAC", f"Saved: {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    return paths


def tile_scene(src_path: Path, cfg: PipelineConfig) -> List[Path]:
    out_dir = cfg.tiles_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log("tile", f"Splitting {src_path.name} into {cfg.tile_size}x{cfg.tile_size} tiles...")
    with rasterio.open(src_path) as src:
        width, height = src.width, src.height
        n_tiles_x = (width + cfg.tile_size - 1) // cfg.tile_size
        n_tiles_y = (height + cfg.tile_size - 1) // cfg.tile_size
        total = n_tiles_x * n_tiles_y
        log("tile", f"Grid: {n_tiles_x} x {n_tiles_y} = ~{total} tiles")

        tiles = []
        tid = 0
        pbar = tqdm(total=total, desc="  Tiling", unit="tile")
        for y in range(0, height, cfg.tile_size):
            for x in range(0, width, cfg.tile_size):
                w = min(cfg.tile_size, width - x)
                h = min(cfg.tile_size, height - y)
                if w < cfg.tile_size or h < cfg.tile_size:
                    pbar.update(1)
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
                pbar.update(1)
        pbar.close()
        log("tile", f"{len(tiles)} tiles created → {out_dir}")
        return tiles


def download_and_tile(cfg: PipelineConfig, max_scenes: int = 3) -> List[Path]:
    paths = download_best_scenes(cfg, max_scenes)
    all_tiles = []
    for p in paths:
        tiles = tile_scene(p, cfg)
        all_tiles.extend(tiles)
    return all_tiles


def log(step: str, msg: str):
    print(f"  [{step}] {msg}")
