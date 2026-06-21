"""Image-image CLIP diagnostic: for cached tiles, generate candidate
silhouettes and report best-letter cosine + runner-up margin. Tells us
what score thresholds make sense before running the full pipeline."""

import sys
import cv2
import numpy as np
from pathlib import Path

from landscript.config import PipelineConfig
from landscript.regions import REGIONS
from landscript.clip_matcher import (
    dominant_contours, render_candidate_silhouette,
    letter_silhouette_embeddings, _embed_gray,
)

region_id = sys.argv[1] if len(sys.argv) > 1 else "in-kutch-rann"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 30

region = REGIONS[region_id]
cfg = PipelineConfig(bbox=region.bbox, region_name=region.id)
tile_files = sorted(cfg.tiles_dir.glob("*.png"))[:limit]
print(f"Scoring {len(tile_files)} tiles from {region_id}")

letter_embs, letters = letter_silhouette_embeddings()

rows = []
for tp in tile_files:
    img = cv2.imread(str(tp))
    if img is None:
        continue
    contours = dominant_contours(img)
    if not contours:
        continue
    for c in contours:
        sil = render_candidate_silhouette(None, c)
        if sil.sum() == 0:
            continue
        emb = _embed_gray(sil)
        scores = letter_embs @ emb
        order = np.argsort(-scores)
        top = float(scores[order[0]])
        second = float(scores[order[1]])
        rows.append({
            "tile": tp.name,
            "letter": letters[order[0]],
            "top": top,
            "second": second,
            "margin": top - second,
        })

# Sort by best score
rows.sort(key=lambda r: -r["top"])
print(f"\n{'tile':<55} {'L':<3} {'top':<6} {'2nd':<6} {'margin':<8}")
for r in rows[:30]:
    print(f"{r['tile']:<55} {r['letter']:<3} {r['top']:<6.3f} {r['second']:<6.3f} {r['margin']:<+8.4f}")

scores = np.array([r["top"] for r in rows])
margins = np.array([r["margin"] for r in rows])
print(f"\nstats across {len(rows)} candidates:")
print(f"  top score: max={scores.max():.3f} p90={np.percentile(scores,90):.3f} median={np.median(scores):.3f}")
print(f"  margin:    max={margins.max():+.4f} p90={np.percentile(margins,90):+.4f} median={np.median(margins):+.4f}")
print(f"\nTiles with top>=0.85 and margin>=0.02: {((scores>=0.85)&(margins>=0.02)).sum()}")
print(f"Tiles with top>=0.80 and margin>=0.02: {((scores>=0.80)&(margins>=0.02)).sum()}")
print(f"Tiles with top>=0.75 and margin>=0.02: {((scores>=0.75)&(margins>=0.02)).sum()}")
print(f"Tiles with top>=0.70 and margin>=0.02: {((scores>=0.70)&(margins>=0.02)).sum()}")