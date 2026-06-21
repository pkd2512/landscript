import time
import requests
import rasterio
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from tqdm import tqdm
from .config import PipelineConfig, STAC_PROVIDERS

SAS_CACHE: Dict[str, str] = {}


def _get_sas_token(cfg: PipelineConfig) -> Optional[str]:
    provider = cfg.satellite_config["stac_provider"]
    info = STAC_PROVIDERS[provider]
    if info.get("data_auth") != "sas-token":
        return None
    coll = cfg.stac_collection
    if coll not in SAS_CACHE:
        url = f"{info['sas_url']}/{coll}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        SAS_CACHE[coll] = resp.json()["token"]
    return SAS_CACHE[coll]


def _resolve_href(href: str, sas_token: Optional[str]) -> str:
    if sas_token and "blob.core.windows.net" in href:
        sep = "&" if "?" in href else "?"
        return f"{href}{sep}{sas_token}"
    return href


def list_scenes(cfg: PipelineConfig) -> List[dict]:
    log("STAC", "Searching for scenes...")
    bbox = cfg.bbox.as_tuple()
    url = f"{cfg.stac_url}/search"
    bbox_str = ",".join(str(v) for v in bbox)
    dt_str = f"{cfg.date_start}T00:00:00Z/{cfg.date_end}T23:59:59Z"
    params = {
        "collections": [cfg.stac_collection],
        "bbox": bbox_str,
        "datetime": dt_str,
        "limit": 50,
    }
    t0 = time.time()
    if "planetary-computer" in cfg.stac_url:
        resp = requests.post(url, json=params, timeout=30)
    else:
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
    scenes.sort(key=lambda s: s["cloud"])
    return scenes


def download_scene(item: dict, out_path: Path, cfg: PipelineConfig) -> Optional[Path]:
    bands = cfg.rgb_bands
    log("STAC", f"Resolving asset URLs for {item['date']}...")
    search_url = f"{cfg.stac_url}/collections/{item['collection']}/items/{item['id']}"

    resp = requests.get(search_url, timeout=30)
    resp.raise_for_status()
    feat = resp.json()
    assets = feat.get("assets", {})

    sas_token = _get_sas_token(cfg)

    rgb = []
    for b in bands:
        href = assets.get(b, {}).get("href")
        if not href:
            log("STAC", f"Band {b} not found, skipping")
            return None
        rgb.append(_resolve_href(href, sas_token))

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
        crs = str(src.crs)
        transform = src.transform
        log("tile", f"Grid: {n_tiles_x} x {n_tiles_y} = ~{total} tiles  |  CRS: {crs}")

        log("tile", "Computing scene-level 5%–95% stretch per band...")
        thumb_shape = (max(1, height // 20), max(1, width // 20))
        stats = src.read(out_shape=thumb_shape).astype(np.float32)
        lo_vals, hi_vals = [], []
        for b in range(min(3, stats.shape[0])):
            valid = stats[b][stats[b] > 0]
            if len(valid) < 100:
                lo_vals.append(float(stats[b].min()))
                hi_vals.append(float(stats[b].max()))
            else:
                lo_vals.append(float(np.percentile(valid, 5)))
                hi_vals.append(float(np.percentile(valid, 95)))
        log("tile", f"  Per-band low: {[f'{v:.0f}' for v in lo_vals]}")
        log("tile", f"  Per-band high: {[f'{v:.0f}' for v in hi_vals]}")

        tiles = []
        tile_index = {}
        tid = 0
        pbar = tqdm(total=total, desc="  Tiling", unit="tile")
        for row in range(0, height, cfg.tile_size):
            for col in range(0, width, cfg.tile_size):
                w = min(cfg.tile_size, width - col)
                h = min(cfg.tile_size, height - row)
                if w < cfg.tile_size or h < cfg.tile_size:
                    pbar.update(1)
                    continue
                window = rasterio.windows.Window(col, row, w, h)
                tile = src.read(window=window)
                tile_path = out_dir / f"{src_path.stem}_tile{tid:04d}.png"
                img = np.moveaxis(tile[:3], 0, -1)
                img = normalize_uint8(img, lo_vals, hi_vals)
                import cv2
                blur = cv2.GaussianBlur(img, (0, 0), 1.5)
                img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
                cv2.imwrite(str(tile_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                tile_index[tile_path.name] = {
                    "col_off": int(col),
                    "row_off": int(row),
                    "crs": crs,
                    "transform": list(transform),
                }
                tiles.append(tile_path)
                tid += 1
                pbar.update(1)
        pbar.close()

        idx_path = out_dir / "tile_index.json"
        with open(idx_path, "w", encoding="utf-8") as f:
            import json
            json.dump(tile_index, f, indent=2)

        log("tile", f"{len(tiles)} tiles created → {out_dir}")
        return tiles


def download_and_tile(cfg: PipelineConfig, max_scenes: int = 3) -> List[Path]:
    paths = download_best_scenes(cfg, max_scenes)
    all_tiles = []
    for p in paths:
        tiles = tile_scene(p, cfg)
        all_tiles.extend(tiles)
    return all_tiles


def normalize_uint8(img: np.ndarray, lo_vals=None, hi_vals=None) -> np.ndarray:
    """Percent-clip stretch to uint8 using scene-level statistics."""
    img = img.astype(np.float32)
    if lo_vals is not None and hi_vals is not None:
        for b in range(img.shape[2]):
            img[:, :, b] = np.clip(img[:, :, b], lo_vals[b], hi_vals[b])
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255
    return img.astype(np.uint8)


def log(step: str, msg: str):
    print(f"  [{step}] {msg}")
