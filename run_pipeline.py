#!/usr/bin/env python3
"""Run the full Landscript pipeline for a region.

Usage:
    python run_pipeline.py                    # Bangalore defaults
    python run_pipeline.py --region mumbai --satellite sentinel-2 --scenes 5
"""

import argparse
import cv2
from pathlib import Path
from landscript.config import REGIONS, SATELLITES, PipelineConfig
from landscript.stac import download_and_tile
from landscript.cv_pipeline import (
    load_letter_templates, to_grayscale, apply_threshold,
    find_contours, filter_contours, match_shape,
    contour_to_polygon, extract_glyph_crop
)
from landscript.metadata import GlyphStore


def main():
    parser = argparse.ArgumentParser(description="Landscript glyph extraction pipeline")
    parser.add_argument("--region", default="bangalore", choices=list(REGIONS.keys()))
    parser.add_argument("--satellite", default="sentinel-2", choices=list(SATELLITES.keys()))
    parser.add_argument("--scenes", type=int, default=3, help="Number of best scenes to download")
    parser.add_argument("--cloud", type=int, default=20, help="Max cloud cover %")
    parser.add_argument("--threshold", type=float, default=0.15, help="Shape match threshold (lower = stricter)")
    parser.add_argument("--date-start", default="2023-01-01")
    parser.add_argument("--date-end", default="2024-12-31")
    args = parser.parse_args()

    cfg = PipelineConfig(
        bbox=REGIONS[args.region],
        region_name=args.region,
        satellite=args.satellite,
        cloud_cover_max=args.cloud,
        similarity_threshold=args.threshold,
        date_start=args.date_start,
        date_end=args.date_end,
    )

    print(f"Region: {cfg.region_name} ({cfg.satellite})")
    print(f"Bounds: {cfg.bbox}")

    print("\n=== Step 1: Download & tile imagery ===")
    tile_files = download_and_tile(cfg, max_scenes=args.scenes)
    print(f"Tiles: {len(tile_files)}")

    print("\n=== Step 2: Load letter templates ===")
    templates = load_letter_templates(cfg)
    print(f"Templates: {len(templates)} letters")

    print("\n=== Step 3: Process tiles for glyph matches ===")
    store = GlyphStore(cfg.metadata_dir / f"{cfg.region_name}.json")
    print(f"Existing glyphs: {store.count()}")

    candidates_found = 0
    for tile_path in tile_files:
        img = cv2.imread(str(tile_path))
        if img is None:
            continue

        gray = to_grayscale(img)
        binary = apply_threshold(gray, cfg.threshold_method)
        contours = find_contours(binary)
        contours = filter_contours(contours, cfg)

        for contour in contours:
            poly = contour_to_polygon(contour, cfg.epsilon)
            for tmpl in templates:
                score = match_shape(poly, tmpl.contour)
                if score >= cfg.similarity_threshold:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                glyph_id = store.add({
                    "letter": tmpl.letter,
                    "score": float(score),
                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                    "source_tile": tile_path.name,
                    "region": cfg.region_name,
                    "state": cfg.state,
                    "country": cfg.country,
                    "satellite": cfg.satellite,
                })
                glyph_path = cfg.glyphs_dir / tmpl.letter / f"{glyph_id}.png"
                extract_glyph_crop(img, contour, glyph_path)
                candidates_found += 1

    print(f"\nDone! {candidates_found} glyph candidates saved.")
    print(f"Glyphs: {cfg.glyphs_dir}")
    print(f"Metadata: {cfg.metadata_dir / (cfg.region_name + '.json')}")


if __name__ == "__main__":
    main()
