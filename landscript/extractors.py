"""Multi-lens contour extraction.

The original pipeline collapsed RGB → grayscale → Otsu, which loses all the
*color contrast* that makes farmland letters readable (think NASA's "Your
Name in Landsat" — the L-shaped field is yellow against green; in grayscale
the contrast is much weaker). This module exposes several alternative
"lenses" that each emit a binary mask; the pipeline can run any subset and
take the union of contours, deduplicated by bbox IoU.

Lenses provided:

- ``otsu``      — global grayscale Otsu (the original)
- ``landuse``   — k-means cluster the HSV pixels into K groups; emit a mask
                  per cluster so each "land use type" becomes its own
                  candidate region. Surfaces farmland/forest/water blocks.
- ``water``     — blue-dominant pixels (water / shadow / dark land).
- ``vegetation``— green-dominant pixels (NDVI-like proxy for true-color).

Each function takes an RGB-as-BGR uint8 image and returns a list of binary
masks. (We return a *list* because ``landuse`` produces several masks per
tile — one per cluster — and the caller wants to find contours in each.)
"""

from __future__ import annotations

from typing import Callable, Dict, List

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Individual lenses
# ---------------------------------------------------------------------------

def lens_otsu(img_bgr: np.ndarray) -> List[np.ndarray]:
    """Grayscale Otsu — the original behaviour."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [m, cv2.bitwise_not(m)]


def lens_landuse(img_bgr: np.ndarray, k: int = 5,
                 downscale: int = 4) -> List[np.ndarray]:
    """K-means cluster HSV pixels into ``k`` land-use groups and emit one
    binary mask per cluster.

    The image is downscaled before clustering (``downscale``× per side) so
    k-means runs in tens of ms instead of seconds; the per-cluster mask is
    then upsampled to the original resolution. This is the lens that
    surfaces letter shapes hiding inside contrasting field blocks (the L in
    NASA's "Your Name in Landsat" example).
    """
    h, w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (max(1, w // downscale), max(1, h // downscale)),
                       interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    pixels = hsv.reshape(-1, 3)
    # k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(
        pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(small.shape[:2]).astype(np.uint8)
    masks: List[np.ndarray] = []
    for ci in range(k):
        m = (labels == ci).astype(np.uint8) * 255
        m_full = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        # A tiny morphological clean-up so we don't get pepper-noise contours.
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m_full = cv2.morphologyEx(m_full, cv2.MORPH_OPEN, kern)
        masks.append(m_full)
    return masks


def lens_water(img_bgr: np.ndarray) -> List[np.ndarray]:
    """Mask water / dark pixels via blue-dominance + low brightness.

    Water tends to be the darkest band in true-color and disproportionately
    blue. We threshold ``B − 0.5*(R+G)`` together with low V.
    """
    b, g, r = cv2.split(img_bgr.astype(np.int16))
    blue_dom = b - (r + g) // 2          # signed
    blue_mask = (blue_dom > 8).astype(np.uint8) * 255
    v = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., 2]
    dark_mask = (v < 110).astype(np.uint8) * 255
    m = cv2.bitwise_or(blue_mask, dark_mask)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)
    return [m]


def lens_vegetation(img_bgr: np.ndarray) -> List[np.ndarray]:
    """Green-dominant pixels (NDVI-like proxy for true-color imagery).

    True-color has no NIR band, so this is just ``(G − R) / (G + R + 1)``
    thresholded. Reasonable approximation for distinguishing healthy
    vegetation from bare ground / urban.
    """
    f = img_bgr.astype(np.float32) + 1.0
    b, g, r = cv2.split(f)
    ratio = (g - r) / (g + r)
    m = ((ratio > 0.05) & (g > 60)).astype(np.uint8) * 255
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)
    return [m]


# Lens registry. Order matters only for log readability.
LENSES: Dict[str, Callable[[np.ndarray], List[np.ndarray]]] = {
    "otsu": lens_otsu,
    "landuse": lens_landuse,
    "water": lens_water,
    "vegetation": lens_vegetation,
}


# ---------------------------------------------------------------------------
# Multi-lens dispatcher + contour dedup
# ---------------------------------------------------------------------------

def _bbox(contour: np.ndarray):
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, x + w, y + h)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def extract_contours_multilens(
    img_bgr: np.ndarray,
    enabled: List[str],
    min_area: int,
    max_area: int,
    dedup_iou: float = 0.6,
) -> List[np.ndarray]:
    """Run every enabled lens, find contours in each mask, deduplicate.

    Returns a list of unique contours (one per physical feature), with each
    contour represented by the version emitted by the lens that found it
    *first* in ``enabled`` order. So if ``enabled = ["otsu", "landuse"]``
    and Otsu already found a lake outline, the landuse mask's version of
    the same lake is suppressed.
    """
    seen_boxes: List[tuple] = []
    out: List[np.ndarray] = []
    for lens_name in enabled:
        lens = LENSES.get(lens_name)
        if lens is None:
            continue
        for mask in lens(img_bgr):
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            for c in contours:
                a = cv2.contourArea(c)
                if a <= min_area or a >= max_area:
                    continue
                bx = _bbox(c)
                if any(_iou(bx, b) >= dedup_iou for b in seen_boxes):
                    continue
                seen_boxes.append(bx)
                out.append(c)
    return out