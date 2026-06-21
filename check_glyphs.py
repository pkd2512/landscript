#!/usr/bin/env python3
"""Audit remaining glyphs: report per-glyph mean V, mean S, and a verdict."""
import cv2
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
META = ROOT / "data" / "metadata" / "bangalore.json"
TILES = ROOT / "data" / "tiles" / "bangalore"

with open(META, encoding="utf-8") as _f:
    glyphs = json.load(_f)
print(f"{'letter':<6} {'score':<7} {'meanV':<7} {'meanS':<7} {'bright_low_sat%':<16} verdict")
print("-" * 70)

suspects = []
for g in glyphs:
    tile = cv2.imread(str(TILES / g["source_tile"]))
    if tile is None:
        continue
    x, y, w, h = g["bbox"]["x"], g["bbox"]["y"], g["bbox"]["w"], g["bbox"]["h"]
    crop = tile[y:y+h, x:x+w]
    if crop.size == 0:
        continue
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v_mean = hsv[..., 2].mean()
    s_mean = hsv[..., 1].mean()
    bright_low_sat = ((hsv[..., 2] >= 180) & (hsv[..., 1] <= 60)).mean() * 100
    verdict = "CLOUD" if bright_low_sat > 40 else ("suspicious" if bright_low_sat > 20 else "ok")
    print(f"{g['letter']:<6} {g['score']:<7.3f} {v_mean:<7.1f} {s_mean:<7.1f} {bright_low_sat:<16.1f} {verdict}")
    if verdict != "ok":
        suspects.append(g["id"])

print(f"\n{len(suspects)} suspect glyphs of {len(glyphs)} total")