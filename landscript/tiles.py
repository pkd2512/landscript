"""Tile management: cloud filtering + perceptual-hash deduplication.

Given a set of already-tiled PNGs from ``landscript.stac.download_and_tile``,
this module:

  - Rejects tiles that are mostly cloud (bright + desaturated in HSV).
  - Rejects tiles with very low overall contrast (uniform desert / ocean /
    farmland that won't surface any shape).
  - Deduplicates near-identical tiles via a 64-bit perceptual hash
    (8x8 mean-threshold pHash). Useful because adjacent scenes overlap.

The whole point of this layer is "throw out the obviously-unusable stuff
before we waste any compute on it."
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Cloud filter (whole tile)
# ---------------------------------------------------------------------------

def cloud_fraction(img_bgr: np.ndarray, v_min: int = 170,
                   s_max: int = 70) -> float:
    """Return the fraction of pixels that look like cloud (bright + low
    saturation). Range 0.0–1.0."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bright = hsv[..., 2] >= v_min
    desat = hsv[..., 1] <= s_max
    return float((bright & desat).mean())


def is_cloudy(img_bgr: np.ndarray, threshold: float = 0.35,
              v_min: int = 170, s_max: int = 70) -> bool:
    """True if ``cloud_fraction`` is at least ``threshold``."""
    return cloud_fraction(img_bgr, v_min=v_min, s_max=s_max) >= threshold


# ---------------------------------------------------------------------------
# Low-contrast filter
# ---------------------------------------------------------------------------

def grayscale_stddev(img_bgr: np.ndarray) -> float:
    """A simple "how much variation is in this tile" number. Uniform desert
    or ocean comes out around 4–10; complex coastlines are 35+."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def is_low_contrast(img_bgr: np.ndarray, min_stddev: float = 12.0) -> bool:
    return grayscale_stddev(img_bgr) < min_stddev


# ---------------------------------------------------------------------------
# Perceptual hash for dedup
# ---------------------------------------------------------------------------

def phash(img_bgr: np.ndarray, size: int = 8) -> int:
    """8x8 mean-threshold perceptual hash. Returns a 64-bit int.

    Two tiles whose Hamming distance is small are visually almost identical;
    we use this to drop overlapping tiles from adjacent scenes.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    bits = (small > mean).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedup_phashes(items: Iterable[Tuple[Path, int]],
                  max_hamming: int = 5) -> List[Path]:
    """Greedy deduplication: walk the list once, keep an item iff no earlier
    kept item has a phash within ``max_hamming`` bits. ~5 bits is "looks
    the same to a human"; ~10 is "loosely similar".
    """
    kept_paths: List[Path] = []
    kept_hashes: List[int] = []
    for path, h in items:
        if any(hamming(h, kh) <= max_hamming for kh in kept_hashes):
            continue
        kept_paths.append(path)
        kept_hashes.append(h)
    return kept_paths


# ---------------------------------------------------------------------------
# Composite filter
# ---------------------------------------------------------------------------

def tile_should_keep(img_bgr: np.ndarray,
                     cloud_threshold: float = 0.35,
                     min_stddev: float = 12.0) -> Tuple[bool, str]:
    """Return ``(keep, reason)``. If ``keep`` is False, ``reason`` is one of
    ``"cloud"`` / ``"low_contrast"`` / ``""``."""
    if is_cloudy(img_bgr, threshold=cloud_threshold):
        return False, "cloud"
    if is_low_contrast(img_bgr, min_stddev=min_stddev):
        return False, "low_contrast"
    return True, ""