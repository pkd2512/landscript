#!/usr/bin/env python3
"""Run the Landscript pipeline.

Single region:
    python run_pipeline.py --region in-kutch-rann --scenes 2
    python run_pipeline.py --region bangalore               # legacy bbox-only

Whole country (loops every region; aggregates to data/metadata/<country>.json):
    python run_pipeline.py --country india
    python run_pipeline.py --country india --scenes 1

Each tile produces **at most one glyph per (mask, scale) combination** — the
dominant shape of that mask is what's matched. The saved PNG is the whole
tile (or the whole downsampled tile, upscaled back), so the letter dominates
the frame.
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
from landscript.cv_pipeline import resolve_font
from landscript.metadata import GlyphStore
from landscript.regions import COUNTRIES, REGIONS, Region, regions_for_country
from landscript.shape_filters import _looks_like_cloud
from landscript.stac import download_and_tile
from landscript.tile_matcher import (
    candidate_descriptors_for_tile,
    render_letter_templates,
    score_against_letter,
)


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
# Per-tile matching
# ---------------------------------------------------------------------------

def process_tile_best_letter(img_bgr, templates, cfg, scales=(1.0, 0.5),
                             margin: float = 0.15,
                             min_area_frac: float = 0.18,
                             max_area_frac: float = 0.55):
    """Return at most ONE (letter, score, candidate, scale) for the tile.

    Across all (scale × mask × letter) combinations, we compute the score of
    every plausible match, then pick the single best. We also require the
    top score to beat the second-best by at least ``margin`` — otherwise the
    tile is ambiguous (the same shape "looks like" two different letters)
    and we reject the whole tile rather than picking one arbitrarily.

    Area gates are tightened so the candidate mask occupies a letter-like
    fraction of the tile (default 18–55%): too small and it's a detail in
    the picture; too large and it's basically the whole tile.
    """
    all_results = []
    for s in scales:
        if s == 1.0:
            tile = img_bgr
        else:
            tile = cv2.resize(
                img_bgr, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA,
            )
        cands = candidate_descriptors_for_tile(
            tile, min_area_frac=min_area_frac, max_area_frac=max_area_frac,
            k_landuse=5,
        )
        for cand in cands:
            if cfg.cloud_filter_enabled and _looks_like_cloud(
                tile, cand.contour,
                v_min=cfg.cloud_v_min, s_max=cfg.cloud_s_max,
                pct=cfg.cloud_pixel_pct,
            ):
                continue
            # Score against every letter; gates: hole count + descriptor
            # bands + pixel-IoU >= min_iou. Returns (score, iou) or None.
            for letter, tmpl in templates.items():
                r = score_against_letter(
                    cand, tmpl, letter, min_iou=cfg.min_iou,
                )
                if r is None:
                    continue
                sc, iou = r
                if sc > cfg.similarity_threshold:
                    continue
                all_results.append({
                    "letter": letter,
                    "score": sc,
                    "iou": iou,
                    "scale": s,
                    "candidate": cand,
                })
    if not all_results:
        return None
    all_results.sort(key=lambda r: r["score"])
    best = all_results[0]
    # Ambiguity gate: if the second-best score is too close, this tile
    # is matching multiple letters and probably doesn't look like *any*
    # particular one — skip rather than pick at random.
    for r in all_results[1:]:
        if r["letter"] != best["letter"] and (r["score"] - best["score"]) < margin:
            return None
    return best


def render_overlay(img_bgr, candidate, scale: float, letter: str,
                   score: float) -> "np.ndarray":
    """Return a copy of ``img_bgr`` with the matched silhouette overlaid in
    a bright contrasting colour, plus a small label badge.

    This is what the gallery actually shows the human, so they can tell at a
    glance which shape the matcher thought was the letter.
    """
    overlay = img_bgr.copy()
    h, w = overlay.shape[:2]

    # Re-scale the contour back to the original tile if it was found at a
    # downscaled scale.
    contour = candidate.contour
    if scale != 1.0:
        contour = np.round(contour.astype(np.float32) / scale).astype(np.int32)

    # Dim the non-matched region.
    mask_full = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_full, [contour], -1, 255, thickness=cv2.FILLED)
    dim = (overlay * 0.45).astype(np.uint8)
    overlay = np.where(mask_full[..., None] == 255, overlay, dim)

    # Outline the matched contour in a bright yellow-orange.
    cv2.drawContours(overlay, [contour], -1, (0, 220, 255), thickness=4)

    # Label badge in the bottom-left.
    label = f"{letter}  {score:.2f}"
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2,
    )
    pad = 12
    box_x1, box_y1 = 16, h - th - baseline - pad * 2 - 8
    box_x2, box_y2 = box_x1 + tw + pad * 2, h - 16
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.putText(overlay, label,
                (box_x1 + pad, box_y2 - pad - baseline // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 255), 2,
                lineType=cv2.LINE_AA)
    return overlay


# ---------------------------------------------------------------------------
# Per-region pipeline
# ---------------------------------------------------------------------------

def run_one_region(region: Region, args, store: GlyphStore,
                   templates=None) -> dict:
    cfg = PipelineConfig(
        bbox=region.bbox,
        region_name=region.id,
        state=region.state or "",
        country=region.country or "",
        satellite=args.satellite,
        cloud_cover_max=args.cloud,
        similarity_threshold=args.threshold,
        composite=args.composite,
        date_start=args.date_start,
        date_end=args.date_end,
        tile_size=args.tile_size,
        cloud_filter_enabled=not args.no_cloud_filter,
        cloud_pixel_pct=args.cloud_pixel_pct,
    )
    cfg.min_iou = args.min_iou

    print(f"\n{'='*50}")
    print(f"  Region — {region.id}  ({region.name})")
    print(f"{'='*50}")
    log("config", f"Terrain tags: {region.terrain_tags or '—'}")
    log("config", f"Bounds: {cfg.bbox}  satellite: {cfg.satellite}")

    print(f"\n--- Step 1/3: Download & tile imagery ---")
    tile_files = download_and_tile(cfg, max_scenes=args.scenes)
    log("tile", f"{len(tile_files)} tiles ready")

    if templates is None:
        print(f"\n--- Step 2/3: Render letter templates ---")
        font_path = resolve_font(cfg.font)
        templates = render_letter_templates(
            font_path, font_size=cfg.font.size, canvas=256,
        )
        log("fonts", f"{len(templates)} letter templates ready")

    print(f"\n--- Step 3/3: Match dominant shapes (tile-as-glyph) ---")
    tile_index_path = cfg.tiles_dir / "tile_index.json"
    tile_index: dict = {}
    if tile_index_path.exists():
        with open(tile_index_path, encoding="utf-8") as f:
            tile_index = json.load(f)

    # Lazy-load CLIP only if requested. Keeps the cheap path independent of
    # torch / open-clip-torch.
    clip_score = None
    if args.rerank_clip:
        from landscript.clip_rerank import score_candidate as clip_score
        log("clip", f"CLIP reranker enabled  min margin {args.clip_min_margin}")

    candidates_found = 0
    clip_rejected = 0
    for tile_path in tqdm(tile_files, desc="  Tiles", unit="tile"):
        img = cv2.imread(str(tile_path))
        if img is None:
            continue

        best = process_tile_best_letter(img, templates, cfg)
        if best is None:
            continue

        cand = best["candidate"]
        scale = best["scale"]
        cx_s = cand.bbox[0] + cand.bbox[2] // 2
        cy_s = cand.bbox[1] + cand.bbox[3] // 2
        cx = int(cx_s / scale)
        cy = int(cy_s / scale)
        lon, lat = pixel_to_coords(tile_path.name, cx, cy, tile_index)

        # Optional CLIP semantic check — only if --rerank-clip and we have
        # a clip scorer available. Skips the tile if CLIP doesn't agree
        # that the picture looks like the chosen letter.
        entry_clip = None
        if clip_score is not None:
            overlay_preview = render_overlay(
                img, cand, scale, best["letter"], best["score"],
            )
            cs = clip_score(overlay_preview, best["letter"])
            if cs["margin"] < args.clip_min_margin:
                clip_rejected += 1
                continue
            entry_clip = {
                "pos_mean": round(cs["pos_mean"], 4),
                "neg_mean": round(cs["neg_mean"], 4),
                "margin": round(cs["margin"], 4),
            }

        entry = {
            "letter": best["letter"],
            "score": float(best["score"]),
            "iou": float(round(best.get("iou", 0.0), 3)),
            "scale": float(scale),
            "holes": int(cand.holes),
            "solidity": float(round(cand.solidity, 3)),
            "aspect": float(round(cand.aspect, 3)),
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
        if entry_clip is not None:
            entry["clip"] = entry_clip
        glyph_id = store.add(entry)

        overlay = render_overlay(img, cand, scale, best["letter"], best["score"])
        glyph_path = cfg.glyphs_dir / best["letter"] / f"{glyph_id}.png"
        glyph_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(glyph_path), overlay)
        candidates_found += 1

    return {
        "region": region.id,
        "candidates_found": candidates_found,
        "clip_rejected": clip_rejected,
        "_templates": templates,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Landscript glyph extraction pipeline")
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
    parser.add_argument("--threshold", type=float, default=2.5,
                        help="Letter-match score threshold (lower = stricter). "
                             "tile_matcher uses Hu+descriptors so the scale is "
                             "different from before. Default: 2.5")
    parser.add_argument("--composite", default="true-color")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--date-start", default="2023-01-01")
    parser.add_argument("--date-end", default="2024-12-31")
    parser.add_argument("--no-cloud-filter", action="store_true")
    parser.add_argument("--cloud-pixel-pct", type=float, default=35.0)
    parser.add_argument("--min-iou", type=float, default=0.45,
                        help="Pixel-IoU threshold against the rendered letter "
                             "silhouette (default 0.45). Lower=more permissive.")
    parser.add_argument("--rerank-clip", action="store_true",
                        help="Filter candidates with CLIP zero-shot. CLIP only "
                             "scores existing imagery — no images are generated.")
    parser.add_argument("--clip-min-margin", type=float, default=0.02,
                        help="Min (positive - negative) cosine margin for "
                             "CLIP to keep a candidate (default 0.02).")
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
    print(f"{'#'*52}")

    templates = None
    stats = []
    for region in regions:
        try:
            result = run_one_region(region, args, store, templates=templates)
        except Exception as e:
            log("error", f"{region.id}: {e}")
            continue
        templates = result.pop("_templates", templates)
        stats.append(result)

    elapsed = time.time() - t0
    print(f"\n{'#'*52}")
    log("done", f"Total: {store.count()} glyphs in {elapsed:.1f}s "
                f"across {len(regions)} region(s)")
    for s in stats:
        log("done", f"  {s['region']:<28} +{s['candidates_found']} glyphs")
    log("done", f"Metadata: data/metadata/{out_name}.json")
    log("done", f"Glyphs:   data/glyphs/")
    print(f"{'#'*52}\n")


if __name__ == "__main__":
    main()