#!/usr/bin/env python3
"""Run the Landscript pipeline (CLIP-primary).

Single region:
    python run_pipeline.py --region in-kutch-rann
    python run_pipeline.py --region bangalore                 # legacy bbox name

Whole country (loops every region; aggregates to data/metadata/<country>.json):
    python run_pipeline.py --country india
    python run_pipeline.py --country india --scenes 2

For every tile we ask CLIP "which of A–Z does this look like, if any, beating
a small set of negative-prompt baselines". One candidate per tile maximum.
The saved PNG is the whole tile, with a small letter badge in the corner.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from landscript.config import REGIONS as LEGACY_BBOXES
from landscript.config import SATELLITES, PipelineConfig
from landscript.metadata import GlyphStore
from landscript.regions import COUNTRIES, REGIONS, Region, regions_for_country
from landscript.shape_filters import _looks_like_cloud
from landscript.stac import download_and_tile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(step: str, msg: str):
    print(f"  [{step}] {msg}")


def pixel_to_coords(tile_name: str, px: int, py: int, index: dict):
    """Convert tile pixel coords to WGS84 lat/lon using tile_index.json."""
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
# Cloud rejection (whole-tile)
# ---------------------------------------------------------------------------

def tile_is_cloudy(img_bgr, cfg) -> bool:
    """Reject the whole tile if it's mostly cloud cover.

    Reuses the HSV bright+desaturated heuristic from shape_filters but
    applied to the whole frame instead of one contour interior.
    """
    if not cfg.cloud_filter_enabled:
        return False
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bright = hsv[..., 2] >= cfg.cloud_v_min
    desat = hsv[..., 1] <= cfg.cloud_s_max
    cloud_pct = 100.0 * (bright & desat).mean()
    return cloud_pct >= cfg.cloud_pixel_pct


# ---------------------------------------------------------------------------
# Glyph rendering (whole tile + small badge)
# ---------------------------------------------------------------------------

def render_overlay_with_silhouette(img_bgr, contour, silhouette,
                                   letter: str, score: float,
                                   runner_up: str, margin: float):
    """Return a composite gallery image:
       left half  = the tile with the matched contour outlined and the
                    non-matched region dimmed
       right half = the 224x224 silhouette CLIP actually saw, scaled up

    This is the *only* way to know what CLIP actually thought was a letter.
    """
    import numpy as np

    h, w = img_bgr.shape[:2]
    # Left: dim outside the contour, outline it bright yellow.
    left = img_bgr.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    dim = (left * 0.45).astype(np.uint8)
    left = np.where(mask[..., None] == 255, left, dim)
    cv2.drawContours(left, [contour], -1, (0, 220, 255), thickness=4)

    # Right: the 224x224 silhouette as CLIP saw it, scaled to match height.
    sil_rgb = cv2.cvtColor(silhouette, cv2.COLOR_GRAY2BGR)
    sil_resized = cv2.resize(sil_rgb, (h, h), interpolation=cv2.INTER_NEAREST)

    composite = np.hstack([left, sil_resized])

    # Label badge across the bottom.
    label = f"{letter}  {score:.3f}   (vs {runner_up}: +{margin:.3f})"
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2,
    )
    pad = 12
    box_h = th + baseline + pad * 2
    H_total = composite.shape[0] + box_h
    out = np.zeros((H_total, composite.shape[1], 3), dtype=np.uint8)
    out[:composite.shape[0]] = composite
    cv2.putText(out, label,
                (pad + 4, composite.shape[0] + box_h - pad - baseline // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2,
                lineType=cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Per-region pipeline
# ---------------------------------------------------------------------------

def run_one_region(region: Region, args, store: GlyphStore) -> dict:
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
        cloud_filter_enabled=not args.no_cloud_filter,
        cloud_pixel_pct=args.cloud_pixel_pct,
    )

    print(f"\n{'='*50}")
    print(f"  Region — {region.id}  ({region.name})")
    print(f"{'='*50}")
    log("config", f"Terrain tags: {region.terrain_tags or '—'}")
    log("config", f"Bounds: {cfg.bbox}  satellite: {cfg.satellite}")

    print(f"\n--- Step 1/2: Download & tile imagery ---")
    tile_files = download_and_tile(cfg, max_scenes=args.scenes)
    log("tile", f"{len(tile_files)} tiles ready")

    print(f"\n--- Step 2/2: CLIP-match each tile ---")
    # Lazy-import keeps the cheap path independent of torch / open-clip-torch.
    from landscript.clip_matcher import classify_tile

    tile_index_path = cfg.tiles_dir / "tile_index.json"
    tile_index: dict = {}
    if tile_index_path.exists():
        with open(tile_index_path, encoding="utf-8") as f:
            tile_index = json.load(f)

    candidates_found = 0
    rejected_cloud = 0
    rejected_no_letter = 0
    rejected_low_margin = 0
    for tile_path in tqdm(tile_files, desc="  Tiles", unit="tile"):
        img = cv2.imread(str(tile_path))
        if img is None:
            continue

        if tile_is_cloudy(img, cfg):
            rejected_cloud += 1
            continue

        m = classify_tile(
            img,
            min_score=args.clip_min_score,
            min_runner_margin=args.clip_margin_runner,
        )
        if m is None:
            rejected_no_letter += 1
            continue

        # Geo-reference centre of the bbox of the matched contour.
        x, y, bw, bh = cv2.boundingRect(m.contour)
        cx, cy = x + bw // 2, y + bh // 2
        lon, lat = pixel_to_coords(tile_path.name, cx, cy, tile_index)

        margin = float(m.letter_score - m.runner_up_score)
        entry = {
            "letter": m.letter,
            "score": float(m.letter_score),
            "runner_up": m.runner_up_letter,
            "runner_up_score": float(m.runner_up_score),
            "margin_vs_runner": margin,
            "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
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
        }
        glyph_id = store.add(entry)

        composite = render_overlay_with_silhouette(
            img, m.contour, m.silhouette,
            m.letter, m.letter_score, m.runner_up_letter, margin,
        )
        glyph_path = cfg.glyphs_dir / m.letter / f"{glyph_id}.png"
        glyph_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(glyph_path), composite)
        candidates_found += 1

    return {
        "region": region.id,
        "candidates_found": candidates_found,
        "rejected_cloud": rejected_cloud,
        "rejected_no_letter": rejected_no_letter,
        "rejected_low_margin": rejected_low_margin,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Landscript CLIP pipeline")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--country", choices=sorted(COUNTRIES.keys()),
                        help="Run every region in the country, aggregate to "
                             "data/metadata/<country>.json")
    target.add_argument("--region", help="Run a single region by id")

    parser.add_argument("--satellite", default="sentinel-2",
                        choices=list(SATELLITES.keys()))
    parser.add_argument("--scenes", type=int, default=1,
                        help="Number of best scenes per region (default: 1)")
    parser.add_argument("--cloud", type=int, default=20)
    parser.add_argument("--composite", default="true-color")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--date-start", default="2023-01-01")
    parser.add_argument("--date-end", default="2024-12-31")
    parser.add_argument("--no-cloud-filter", action="store_true")
    parser.add_argument("--cloud-pixel-pct", type=float, default=35.0,
                        help="%% of tile pixels that must be cloud-like "
                             "before the whole tile is rejected (default 35).")
    parser.add_argument("--clip-min-score", type=float, default=0.75,
                        help="Min absolute CLIP image-image cosine "
                             "(candidate silhouette vs letter silhouette). "
                             "Higher = stricter (default 0.75).")
    parser.add_argument("--clip-margin-runner", type=float, default=0.02,
                        help="Min (best letter cosine − second letter cosine). "
                             "Higher = stricter (default 0.02).")
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

    data_dir = Path("data") / "metadata"
    data_dir.mkdir(parents=True, exist_ok=True)
    store = GlyphStore(data_dir / f"{out_name}.json")

    print(f"\n{'#'*52}")
    print(f"  Landscript run → data/metadata/{out_name}.json")
    print(f"  {len(regions)} region(s) · {args.scenes} scene(s) each "
          f"· {args.satellite}")
    if args.country:
        print(f"  Country: {COUNTRIES[args.country].name}")
    print(f"  Matcher: CLIP image-image  "
          f"(min_score≥{args.clip_min_score} runner_margin≥{args.clip_margin_runner})")
    print(f"{'#'*52}")

    stats = []
    for region in regions:
        try:
            result = run_one_region(region, args, store)
        except Exception as e:
            log("error", f"{region.id}: {e}")
            continue
        stats.append(result)

    elapsed = time.time() - t0
    print(f"\n{'#'*52}")
    log("done", f"Total: {store.count()} glyphs in {elapsed:.1f}s "
                f"across {len(regions)} region(s)")
    for s in stats:
        log(
            "done",
            f"  {s['region']:<28} +{s['candidates_found']:>3} glyphs  "
            f"(cloud={s['rejected_cloud']}  no_letter={s['rejected_no_letter']})",
        )
    log("done", f"Metadata: data/metadata/{out_name}.json")
    log("done", f"Glyphs:   data/glyphs/")
    print(f"{'#'*52}\n")


if __name__ == "__main__":
    main()