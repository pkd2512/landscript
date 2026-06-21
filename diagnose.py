#!/usr/bin/env python3
"""Diagnose why no glyphs are matching. Runs detection stages on existing tiles
and prints contour counts + best match scores so we can see where things drop."""

import cv2
import numpy as np
from pathlib import Path
from landscript.config import REGIONS, PipelineConfig
from landscript.cv_pipeline import (
    load_letter_templates, to_grayscale, apply_threshold,
    find_contours, filter_contours, contour_to_polygon, match_shape,
)


def main(region: str = "bangalore", sample_tiles: int = 20):
    cfg = PipelineConfig(bbox=REGIONS[region], region_name=region)
    print(f"Config: thr={cfg.similarity_threshold} area=[{cfg.min_contour_area},{cfg.max_contour_area}] "
          f"method={cfg.threshold_method} eps={cfg.epsilon}")

    tile_files = sorted(cfg.tiles_dir.glob("*.png"))[:sample_tiles]
    print(f"Sampling {len(tile_files)} of {len(list(cfg.tiles_dir.glob('*.png')))} tiles")

    print("Loading templates...")
    templates = load_letter_templates(cfg)

    totals = {
        "tiles": 0,
        "contours_raw": 0,
        "contours_filtered": 0,
        "best_score_min": float("inf"),
        "best_score_max": 0.0,
        "scores_under_0.10": 0,
        "scores_under_0.20": 0,
        "scores_under_0.50": 0,
    }
    score_samples = []

    for tile_path in tile_files:
        img = cv2.imread(str(tile_path))
        if img is None:
            continue
        totals["tiles"] += 1
        gray = to_grayscale(img)
        binary = apply_threshold(gray, cfg.threshold_method)
        contours_raw = find_contours(binary)
        totals["contours_raw"] += len(contours_raw)
        contours = filter_contours(contours_raw, cfg)
        totals["contours_filtered"] += len(contours)

        for c in contours:
            poly = contour_to_polygon(c, cfg.epsilon)
            best = min(match_shape(poly, t.contour) for t in templates)
            score_samples.append(best)
            totals["best_score_min"] = min(totals["best_score_min"], best)
            totals["best_score_max"] = max(totals["best_score_max"], best)
            if best < 0.10:
                totals["scores_under_0.10"] += 1
            if best < 0.20:
                totals["scores_under_0.20"] += 1
            if best < 0.50:
                totals["scores_under_0.50"] += 1

    print("\n=== Diagnosis ===")
    for k, v in totals.items():
        print(f"  {k}: {v}")
    if score_samples:
        arr = np.array(score_samples)
        print(f"\n  score distribution (per-contour best match across all 26 letters):")
        for p in [10, 25, 50, 75, 90, 99]:
            print(f"    p{p:>2}: {np.percentile(arr, p):.3f}")
    else:
        print("\n  No contours survived filtering — area limits too strict, "
              "or thresholding produced no shapes.")


if __name__ == "__main__":
    import sys
    region = sys.argv[1] if len(sys.argv) > 1 else "bangalore"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    main(region, n)