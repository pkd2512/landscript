#!/usr/bin/env python3
"""Landscript discovery pipeline.

Run for a single region or a whole country. Output is a ranked set of
**candidate tiles** — high-interest satellite shapes that *might* look like
letters to a human. **No letter is predicted.** Humans assign letters in
the gallery.

Examples:

    python run_pipeline.py --region in-kutch-rann
    python run_pipeline.py --country india --scenes 1 --top 200
    python run_pipeline.py --region bangalore                # legacy bbox name

Each region's candidates are stored at ``data/candidates/<region|country>.json``
and the tile PNGs at ``data/glyphs/<region|country>/<id>.png``. The gallery
serves both for browsing + curation.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from landscript.config import REGIONS as LEGACY_BBOXES
from landscript.config import SATELLITES, PipelineConfig
from landscript.features import compute_features
from landscript.metadata import CandidateStore
from landscript.regions import COUNTRIES, REGIONS, Region, regions_for_country
from landscript.stac import download_and_tile
from landscript.tiles import phash, tile_should_keep, hamming


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(step: str, msg: str):
    print(f"  [{step}] {msg}")


def pixel_to_coords(tile_name: str, px: int, py: int, index: dict):
    """Convert tile pixel coords to WGS84 lat/lon via the tile_index."""
    info = index.get(tile_name)
    if not info:
        return None, None
    t = info["transform"]
    cx = t[2] + (info["col_off"] + px) * t[0] + (info["row_off"] + py) * t[1]
    cy = t[5] + (info["col_off"] + px) * t[4] + (info["row_off"] + py) * t[3]
    try:
        from rasterio.warp import transform as warp
        result = warp(info["crs"], "EPSG:4326", [cx], [cy])
        return round(result[0][0], 6), round(result[1][0], 6)
    except Exception:
        return round(cx, 6), round(cy, 6)


def resolve_region(region_id: str) -> Region:
    if region_id in REGIONS:
        return REGIONS[region_id]
    if region_id in LEGACY_BBOXES:
        return Region(
            id=region_id, name=region_id.title(), country="india",
            state="", bbox=LEGACY_BBOXES[region_id], terrain_tags=[],
        )
    raise SystemExit(
        f"Unknown region '{region_id}'. Known: "
        f"{sorted(list(REGIONS.keys()) + list(LEGACY_BBOXES.keys()))}"
    )


# ---------------------------------------------------------------------------
# Per-tile output: just the raw tile, no overlay.
# ---------------------------------------------------------------------------

def render_candidate_png(img_bgr, contour, score: float):
    """Return the raw tile. We used to draw the dominant-shape outline + a
    score badge here, but that made the PNG unusable as a clean image. The
    gallery still shows the bbox + interest score in its detail panel."""
    return img_bgr.copy()


# ---------------------------------------------------------------------------
# Per-region pipeline
# ---------------------------------------------------------------------------

def run_one_region(region: Region, args, kept_phashes) -> list:
    """Run the full pipeline for one region. Returns a list of candidate
    dicts (NOT yet added to the store)."""
    cfg = PipelineConfig(
        bbox=region.bbox,
        region_name=region.id,
        state=region.state or "",
        country=region.country or "",
        satellite=args.satellite,
        cloud_cover_max=args.cloud,
        composite=args.composite,
        date_start=args.date_start,
        date_end=args.date_end,
        tile_size=args.tile_size,
    )

    print(f"\n{'='*50}")
    print(f"  Region — {region.id}  ({region.name})")
    print(f"{'='*50}")
    log("config", f"Terrain tags: {region.terrain_tags or '—'}")
    log("config", f"Bounds: {cfg.bbox}  satellite: {cfg.satellite}")

    print(f"\n--- Step 1/3: Download & tile imagery ---")
    tile_files = download_and_tile(cfg, max_scenes=args.scenes)
    log("tile", f"{len(tile_files)} tiles ready")

    print(f"\n--- Step 2/3: Pre-filter (cloud / low-contrast / dedup) ---")
    tile_index_path = cfg.tiles_dir / "tile_index.json"
    tile_index: dict = {}
    if tile_index_path.exists():
        with open(tile_index_path, encoding="utf-8") as f:
            tile_index = json.load(f)

    survivors = []
    rejected = {"cloud": 0, "low_contrast": 0, "duplicate": 0}
    for tile_path in tqdm(tile_files, desc="  Pre-filter", unit="tile"):
        img = cv2.imread(str(tile_path))
        if img is None:
            continue
        keep, reason = tile_should_keep(
            img,
            cloud_threshold=args.cloud_threshold,
            min_stddev=args.min_stddev,
        )
        if not keep:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        ph = phash(img)
        if any(hamming(ph, kh) <= args.dedup_hamming for kh in kept_phashes):
            rejected["duplicate"] += 1
            continue
        kept_phashes.append(ph)
        survivors.append((tile_path, img, ph))
    log("pre-filter",
        f"kept {len(survivors)}/{len(tile_files)}  "
        f"(cloud={rejected['cloud']} low_contrast={rejected['low_contrast']} "
        f"dup={rejected['duplicate']})")

    print(f"\n--- Step 3/3: Score & extract dominant shape ---")
    candidates = []
    for tile_path, img, ph in tqdm(survivors, desc="  Scoring", unit="tile"):
        feats = compute_features(img)
        if feats is None:
            continue
        x, y, w, h = feats.bbox
        cx, cy = x + w // 2, y + h // 2
        lon, lat = pixel_to_coords(tile_path.name, cx, cy, tile_index)

        candidates.append({
            "phash": str(ph),
            "interest": float(feats.interest_score),
            "area_frac": float(round(feats.area_frac, 4)),
            "holes": int(feats.holes),
            "edge_density": float(round(feats.edge_density, 4)),
            "complexity": float(round(feats.complexity, 4)),
            "solidity": float(round(feats.solidity, 4)),
            "extent": float(round(feats.extent, 4)),
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "descriptor": [float(round(v, 5)) for v in feats.descriptor],
            "lat": lat,
            "lon": lon,
            "source_tile": tile_path.name,
            "region": region.id,
            "region_name": region.name,
            "state": region.state,
            "country": region.country,
            "terrain_tags": region.terrain_tags,
            "satellite": cfg.satellite,
            "composite": cfg.composite,
            "_img": img,                    # transient
            "_contour": feats.contour,      # transient
        })
    log("score", f"{len(candidates)} candidates extracted")
    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Landscript discovery pipeline")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--country", choices=sorted(COUNTRIES.keys()),
                        help="Run every region in the country, aggregate to "
                             "data/candidates/<country>.json")
    target.add_argument("--region",
                        help="Run a single region by id (see regions.py).")

    parser.add_argument("--satellite", default="sentinel-2",
                        choices=list(SATELLITES.keys()))
    parser.add_argument("--scenes", type=int, default=1,
                        help="Number of best scenes per region (default: 1)")
    parser.add_argument("--cloud", type=int, default=20,
                        help="Max scene-level cloud cover %% requested from "
                             "the STAC search (default: 20).")
    parser.add_argument("--composite", default="true-color")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--date-start", default="2023-01-01")
    parser.add_argument("--date-end", default="2024-12-31")

    parser.add_argument("--cloud-threshold", type=float, default=0.35,
                        help="Fraction of tile pixels that look cloud-like "
                             "before the whole tile is dropped (default 0.35).")
    parser.add_argument("--min-stddev", type=float, default=12.0,
                        help="Min grayscale stddev; tiles below this are "
                             "treated as low-contrast and dropped "
                             "(default 12.0).")
    parser.add_argument("--dedup-hamming", type=int, default=5,
                        help="Max pHash Hamming distance treated as duplicate "
                             "(default 5; lower = stricter).")
    parser.add_argument("--top", type=int, default=200,
                        help="Keep at most this many top-interest "
                             "candidates per region (default 200).")
    args = parser.parse_args()

    if args.country:
        regions = regions_for_country(args.country)
        if not regions:
            raise SystemExit(f"Country '{args.country}' has no regions registered.")
        out_name = args.country
    else:
        single = resolve_region(args.region or "bangalore")
        regions = [single]
        out_name = single.id

    data_dir = Path("data")
    cand_dir = data_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    img_root = data_dir / "glyphs" / out_name
    img_root.mkdir(parents=True, exist_ok=True)
    store = CandidateStore(cand_dir / f"{out_name}.json")

    # Seed pHash dedup with existing candidates so re-runs (e.g. a 512px
    # pass on top of a 1024px pass) don't add duplicates of shapes we
    # already have.
    kept_phashes: list = []
    for c in store.all():
        ph = c.get("phash")
        if ph is None:
            continue
        try:
            kept_phashes.append(int(ph))
        except (TypeError, ValueError):
            continue

    print(f"\n{'#'*52}")
    print(f"  Landscript discovery → data/candidates/{out_name}.json")
    print(f"  {len(regions)} region(s) · {args.scenes} scene(s) each "
          f"· {args.satellite}  · tile {args.tile_size}px")
    print(f"  Top {args.top} per region · NO classification, only ranking")
    if kept_phashes:
        print(f"  Seeded with {len(kept_phashes)} existing pHashes "
              f"(cross-run dedup)")
    print(f"{'#'*52}")
    region_stats = []
    for region in regions:
        try:
            cands = run_one_region(region, args, kept_phashes)
        except Exception as e:
            log("error", f"{region.id}: {e}")
            continue

        # Keep top-N per region by interest score, then write images + JSON.
        cands.sort(key=lambda c: -c["interest"])
        cands = cands[:args.top]
        for c in cands:
            img = c.pop("_img")
            contour = c.pop("_contour")
            cid = store.add(c)
            png = render_candidate_png(img, contour, c["interest"])
            out_path = img_root / f"{cid}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), png)
        region_stats.append((region.id, len(cands)))

    elapsed = time.time() - t0
    print(f"\n{'#'*52}")
    log("done", f"Total candidates: {store.count()} in {elapsed:.1f}s")
    for r, n in region_stats:
        log("done", f"  {r:<28} +{n}")
    log("done", f"Candidates: data/candidates/{out_name}.json")
    log("done", f"Images:     data/glyphs/{out_name}/")
    log("done", f"Browse:     python gallery.py --region {out_name} --open")
    print(f"{'#'*52}\n")


if __name__ == "__main__":
    main()