"""Phase A: saliency pre-filters.

Cheap per-contour checks that run BEFORE template shape matching to reject
contours that obviously aren't letter-shaped. Each filter returns
``(passed, reason)`` and the dispatcher returns ``(passed, reason)`` for the
first failing check (or ``(True, "")`` if all pass).

Filters in this module:
- aspect ratio gate (rejects long thin rivers, square farmland blocks)
- solidity gate         area / convex_hull_area
- extent gate           area / bbox_area
- edge-density gate     fraction of Canny edge pixels in the bbox window
- cloud gate            HSV brightness + saturation (moved from cv_pipeline)

All thresholds live on :class:`landscript.config.PipelineConfig` so they can
be tuned per-region without code changes.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import PipelineConfig


def _aspect_ratio(contour: np.ndarray) -> float:
    """Return w / h of the contour's axis-aligned bounding box."""
    _, _, w, h = cv2.boundingRect(contour)
    if h == 0:
        return 0.0
    return w / h


def _solidity(contour: np.ndarray) -> float:
    """Return area / convex_hull_area in [0, 1]."""
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0
    return float(area / hull_area)


def _extent(contour: np.ndarray) -> float:
    """Return area / bbox_area in [0, 1]."""
    area = cv2.contourArea(contour)
    _, _, w, h = cv2.boundingRect(contour)
    bbox_area = float(w * h)
    if bbox_area <= 0:
        return 0.0
    return float(area / bbox_area)


def _edge_density(img: np.ndarray, contour: np.ndarray) -> float:
    """Return the fraction of Canny-edge pixels in the contour's bbox window.

    Letters sit on structured edges (roads, coastlines, building outlines).
    Flat regions (desert, water) have near-zero edge density; chaotic
    texture (dense canopy) has near-one. Letter candidates are in between.
    """
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return 0.0
    tile_h, tile_w = img.shape[:2]
    # Pad the window slightly so the contour's own edges aren't the only
    # thing contributing to the density.
    pad = max(4, min(w, h) // 8)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(tile_w, x + w + pad)
    y1 = min(tile_h, y + h + pad)
    window = img[y0:y1, x0:x1]
    if window.size == 0:
        return 0.0
    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY) if window.ndim == 3 else window
    edges = cv2.Canny(gray, 80, 160)
    return float(edges.mean() / 255.0)


def _looks_like_cloud(img: np.ndarray, contour: np.ndarray,
                      v_min: int, s_max: int, pct: float) -> bool:
    """Return True if the contour interior is mostly bright + desaturated.

    Same logic that previously lived as ``is_cloud_region`` in
    ``cv_pipeline.py``; centralised here so the whole filter pipeline lives
    in one place.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    pixels = img[mask == 255]
    if pixels.size == 0:
        return False
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    s = hsv[:, 1]
    v = hsv[:, 2]
    cloud_pixels = ((v >= v_min) & (s <= s_max)).sum()
    return 100.0 * cloud_pixels / len(hsv) >= pct


def is_candidate_shape(img: np.ndarray, contour: np.ndarray,
                       cfg: PipelineConfig) -> Tuple[bool, str]:
    """Run all enabled saliency filters and return ``(passed, reason)``.

    Filters are evaluated in cheapest-first order so we short-circuit on the
    most common rejection reason without paying for expensive checks
    (Canny edge density, HSV conversion). ``reason`` is the empty string on
    success or a short identifier (``"aspect"``, ``"solidity"``,
    ``"extent"``, ``"edge_density"``, ``"cloud"``) on failure.
    """
    # 1. Aspect ratio — cheapest possible check.
    if cfg.shape_filter_aspect_enabled:
        ar = _aspect_ratio(contour)
        if not (cfg.min_aspect_ratio <= ar <= cfg.max_aspect_ratio):
            return False, "aspect"

    # 2. Extent — single area + bbox arithmetic.
    if cfg.shape_filter_extent_enabled:
        ex = _extent(contour)
        if not (cfg.min_extent <= ex <= cfg.max_extent):
            return False, "extent"

    # 3. Solidity — needs convex hull.
    if cfg.shape_filter_solidity_enabled:
        sol = _solidity(contour)
        if not (cfg.min_solidity <= sol <= cfg.max_solidity):
            return False, "solidity"

    # 4. Edge density — Canny over a window; more expensive.
    if cfg.shape_filter_edge_density_enabled:
        ed = _edge_density(img, contour)
        if not (cfg.min_edge_density <= ed <= cfg.max_edge_density):
            return False, "edge_density"

    # 5. Cloud — HSV conversion of all masked pixels; most expensive.
    if cfg.cloud_filter_enabled and _looks_like_cloud(
        img, contour,
        v_min=cfg.cloud_v_min, s_max=cfg.cloud_s_max,
        pct=cfg.cloud_pixel_pct,
    ):
        return False, "cloud"

    return True, ""