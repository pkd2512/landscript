"""Visual feature extraction for satellite tiles.

Given a tile (BGR uint8) we extract:

  - the **dominant shape**: the largest connected region from a colour-cluster
    segmentation, with its outer contour and any internal holes;
  - a small set of **interpretable features** (edge density, dominant-component
    fraction, shape complexity, hole count, Hu-moments);
  - a single ``interest_score`` combining the features — used purely to **rank**
    tiles, never to label them. High score = visually interesting; low score =
    farmland / texture / blank.

Nothing in this module classifies anything. It produces numbers and shapes
for the gallery to display and humans to evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Dominant-shape extraction
# ---------------------------------------------------------------------------

def _kmeans_label_map(img_bgr: np.ndarray, k: int = 4,
                      downscale: int = 4) -> np.ndarray:
    """Return a per-pixel cluster label map (uint8) using HSV k-means.
    The image is downsampled for speed and the labels upsampled back."""
    h, w = img_bgr.shape[:2]
    small = cv2.resize(
        img_bgr,
        (max(1, w // downscale), max(1, h // downscale)),
        interpolation=cv2.INTER_AREA,
    )
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    pixels = hsv.reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(
        pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(small.shape[:2]).astype(np.uint8)
    return cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Smooth + fill speckle so the largest component has letter-sized
    topology rather than thousands of pinhole holes."""
    h, w = mask.shape
    k = max(5, min(h, w) // 48) | 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)
    # Fill tiny holes (<0.6% of tile area) so the topology is meaningful.
    min_area = max(64, int(0.006 * h * w))
    inv = cv2.bitwise_not(m)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            m[labels == i] = 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            m[labels == i] = 0
    return m


def dominant_shape(img_bgr: np.ndarray, k: int = 4,
                   min_area_frac: float = 0.08,
                   max_area_frac: float = 0.85,
                   ) -> Optional[dict]:
    """Find the dominant shape in the tile by checking each k-means cluster
    and picking the cluster whose largest connected component best satisfies:

      - area is between ``min_area_frac`` and ``max_area_frac`` of the tile
      - among those that qualify, pick the one with the *highest area*
        (the most dominant shape that isn't basically the whole tile).

    Returns a dict with:
        contour       outer contour (np.ndarray)
        hole_contours list of interior contours (may be empty)
        bbox          (x, y, w, h)
        area          int
        holes         int
        mask          full-resolution binary mask (uint8)
      Or ``None`` if no cluster yields a qualifying shape.
    """
    h, w = img_bgr.shape[:2]
    tile_area = h * w
    lo = int(tile_area * min_area_frac)
    hi = int(tile_area * max_area_frac)

    label_map = _kmeans_label_map(img_bgr, k=k)

    best: Optional[dict] = None
    for ci in range(k):
        mask = (label_map == ci).astype(np.uint8) * 255
        mask = _clean_mask(mask)
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours or hierarchy is None:
            continue
        hierarchy = hierarchy[0]
        # Pick the biggest outer contour in this cluster.
        outers = [(i, c) for i, c in enumerate(contours)
                  if hierarchy[i][3] == -1]
        if not outers:
            continue
        idx, contour = max(outers, key=lambda ic: cv2.contourArea(ic[1]))
        area = int(cv2.contourArea(contour))
        if area < lo or area > hi:
            continue
        if best is not None and area <= best["area"]:
            continue
        hole_contours = [
            contours[i] for i in range(len(contours))
            if hierarchy[i][3] == idx
        ]
        x, y, cw, ch = cv2.boundingRect(contour)
        best = {
            "contour": contour,
            "hole_contours": hole_contours,
            "bbox": (x, y, cw, ch),
            "area": area,
            "holes": len(hole_contours),
            "mask": mask,
        }
    return best


# ---------------------------------------------------------------------------
# Feature numbers
# ---------------------------------------------------------------------------

def edge_density(img_bgr: np.ndarray) -> float:
    """Fraction of pixels that are Canny edges. ~0.02 for uniform; ~0.4+ for
    chaotic texture; recognisable shapes land between."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(edges.mean() / 255.0)


def shape_complexity(contour: np.ndarray) -> float:
    """Isoperimetric-ratio-like complexity: ``perimeter² / (4π × area)``.

    A perfect circle = 1.0; a square ≈ 1.27; jagged coastlines and complex
    letterforms 3.0–10.0; very wiggly noise 20+. Used as both a feature and
    a tie-breaker — letters tend to live in the 2–8 range.
    """
    area = max(1.0, cv2.contourArea(contour))
    peri = cv2.arcLength(contour, True)
    return (peri * peri) / (4.0 * np.pi * area)


def hu_log(contour: np.ndarray) -> np.ndarray:
    """7 Hu moments, log-scaled and sign-preserved. One component of the
    descriptor vector used for similarity search (NOT for classification)."""
    m = cv2.moments(contour)
    h = cv2.HuMoments(m).flatten()
    return -np.sign(h) * np.log10(np.abs(h) + 1e-30)


# ---------------------------------------------------------------------------
# Combined features + interest score
# ---------------------------------------------------------------------------

@dataclass
class TileFeatures:
    """Everything we compute for one tile that has a dominant shape."""
    bbox: Tuple[int, int, int, int]
    area: int
    area_frac: float
    holes: int
    edge_density: float
    complexity: float
    solidity: float
    extent: float
    hu: np.ndarray
    descriptor: np.ndarray   # the vector used for similarity (11-D)
    interest_score: float
    contour: np.ndarray = field(repr=False)
    hole_contours: List[np.ndarray] = field(repr=False, default_factory=list)


def _complexity_score(complexity: float) -> float:
    """Map shape_complexity to 0–1: letters live around 2–8.
    Bell curve centred ~5, with FWHM ~6. Plain blobs (~1) and noise (>20)
    both score low."""
    return float(np.exp(-((complexity - 5.0) / 4.0) ** 2))


def compute_features(img_bgr: np.ndarray) -> Optional[TileFeatures]:
    """Compute everything for one tile. Returns None if no dominant shape
    qualifies."""
    h, w = img_bgr.shape[:2]
    tile_area = h * w

    shape = dominant_shape(img_bgr)
    if shape is None:
        return None
    contour = shape["contour"]
    area = shape["area"]
    bbox = shape["bbox"]
    bx, by, bw, bh = bbox
    bbox_area = max(1, bw * bh)
    hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(contour))))

    feat_edge = edge_density(img_bgr)
    feat_complexity = shape_complexity(contour)
    feat_solidity = area / hull_area
    feat_extent = area / bbox_area
    feat_area_frac = area / tile_area
    feat_hu = hu_log(contour)

    # Interest score: visual quality, NOT letter-ness.
    # Reward a dominant shape that fills a recognisable fraction of the
    # tile; reward letter-like complexity (~3–8); penalise both blobs (low
    # complexity) and chaotic noise (high complexity).
    score = (
        0.45 * min(1.0, feat_area_frac / 0.4)   # dominant shape fraction
        + 0.30 * _complexity_score(feat_complexity)  # letter-like complexity
        + 0.15 * min(1.0, feat_edge / 0.20)     # has visible edges
        + 0.10 * feat_solidity                  # not a fractal
    )
    score = float(np.clip(score, 0.0, 1.0))

    descriptor = np.concatenate([
        feat_hu,
        np.array([feat_area_frac, feat_edge, feat_complexity,
                  feat_solidity], dtype=np.float64),
    ])

    return TileFeatures(
        bbox=bbox,
        area=area,
        area_frac=feat_area_frac,
        holes=shape["holes"],
        edge_density=feat_edge,
        complexity=feat_complexity,
        solidity=feat_solidity,
        extent=feat_extent,
        hu=feat_hu,
        descriptor=descriptor,
        interest_score=score,
        contour=contour,
        hole_contours=shape["hole_contours"],
    )