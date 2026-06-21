"""Sample Kutch tiles and print the best IoU each candidate achieves against
any letter template, ignoring all other gates. Tells us what IoU range is
actually achievable so we can set a realistic threshold."""

import sys
import cv2
import numpy as np
from pathlib import Path

from landscript.config import PipelineConfig
from landscript.cv_pipeline import resolve_font
from landscript.regions import REGIONS
from landscript.tile_matcher import (
    candidate_descriptors_for_tile,
    iou_against_template,
    LETTER_DESCRIPTORS,
    render_letter_templates,
)

region_id = sys.argv[1] if len(sys.argv) > 1 else "in-kutch-rann"
region = REGIONS[region_id]
cfg = PipelineConfig(bbox=region.bbox, region_name=region.id)

tile_dir = cfg.tiles_dir
tile_files = sorted(tile_dir.glob("*.png"))[:20]
print(f"Sampling {len(tile_files)} tiles from {tile_dir}")

font_path = resolve_font(cfg.font)
templates = render_letter_templates(font_path, font_size=cfg.font.size, canvas=256)

# First just print descriptor distributions WITHOUT gating, so we can see
# what values candidates actually have.
all_descriptors = []
for tp in tile_files:
    img = cv2.imread(str(tp))
    if img is None:
        continue
    cands = candidate_descriptors_for_tile(
        img, min_area_frac=0.18, max_area_frac=0.55, k_landuse=5,
    )
    for cand in cands:
        all_descriptors.append((tp.name, cand.holes,
                                round(cand.solidity, 3),
                                round(cand.aspect, 3),
                                cand))

print(f"\nTotal candidates: {len(all_descriptors)}")
if not all_descriptors:
    sys.exit(0)
print("\nDescriptor distribution:")
import collections
holes_dist = collections.Counter(d[1] for d in all_descriptors)
print(f"  holes: {dict(sorted(holes_dist.items()))}")
sols = [d[2] for d in all_descriptors]
asps = [d[3] for d in all_descriptors]
print(f"  solidity: min={min(sols):.2f} median={np.median(sols):.2f} max={max(sols):.2f}")
print(f"  aspect:   min={min(asps):.2f} median={np.median(asps):.2f} max={max(asps):.2f}")

# Now best IoU vs ANY template (no gates) — what is achievable at all?
print("\nComputing best-IoU per candidate (NO gates)...")
ious_any = []
for tile, holes, sol, asp, cand in all_descriptors[:60]:
    best_iou = 0.0
    best_letter = "-"
    for letter, tmpl in templates.items():
        iou = iou_against_template(
            cand.contour, cand.hole_contours or [], tmpl.canvas_mask,
        )
        if iou > best_iou:
            best_iou = iou
            best_letter = letter
    ious_any.append((tile, best_letter, best_iou, holes, sol, asp))

ious_any.sort(key=lambda r: -r[2])
print(f"\nTop 30 IoU (ignoring hole/band gates):\n")
print(f"  {'tile':<55} {'letter':<6} {'iou':<6} {'holes':<6} {'sol':<6} {'asp':<6}")
for row in ious_any[:30]:
    t, l, iou, h, s, a = row
    print(f"  {t:<55} {l:<6} {iou:<6.3f} {h:<6} {s:<6} {a:<6}")
all_ious = [r[2] for r in ious_any]
print(f"\nIoU stats across {len(all_ious)} candidates:")
print(f"  max={max(all_ious):.3f}  p90={np.percentile(all_ious, 90):.3f}  "
      f"p75={np.percentile(all_ious, 75):.3f}  median={np.median(all_ious):.3f}")

# Now also report which candidates pass current LETTER_DESCRIPTORS gates.
print("\nWith current LETTER_DESCRIPTORS gates:")
best_per_tile = []
for tile, holes, sol, asp, cand in all_descriptors:
    best_iou = 0.0
    best_letter = "-"
    for letter, tmpl in templates.items():
        spec = LETTER_DESCRIPTORS[letter]
        holes_req, sol_lo, sol_hi, asp_lo, asp_hi = spec
        if cand.holes != holes_req:
            continue
        if not (sol_lo <= cand.solidity <= sol_hi):
            continue
        if not (asp_lo <= cand.aspect <= asp_hi):
            continue
        iou = iou_against_template(
            cand.contour, cand.hole_contours or [], tmpl.canvas_mask,
        )
        if iou > best_iou:
            best_iou = iou
            best_letter = letter
    if best_letter != "-":
        best_per_tile.append((tile, best_letter, best_iou,
                              cand.holes, round(cand.solidity, 2),
                              round(cand.aspect, 2)))

if not best_per_tile:
    print("No candidates passed even the hole/band gates.")
    sys.exit(0)

best_per_tile.sort(key=lambda r: -r[2])
print(f"\nTop 30 IoU achievements (best-matching letter per candidate):\n")
print(f"  {'tile':<55} {'letter':<6} {'iou':<6} {'holes':<6} {'sol':<6} {'asp':<6}")
for row in best_per_tile[:30]:
    tile, letter, iou, holes, sol, asp = row
    print(f"  {tile:<55} {letter:<6} {iou:<6.3f} {holes:<6} {sol:<6} {asp:<6}")

ious = [r[2] for r in best_per_tile]
print(f"\n{len(ious)} candidates passed pre-gates; IoU stats:")
print(f"  max={max(ious):.3f}  p90={np.percentile(ious, 90):.3f}  "
      f"p75={np.percentile(ious, 75):.3f}  median={np.median(ious):.3f}")