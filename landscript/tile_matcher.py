"""Tile-as-glyph matching.

The original pipeline detected hundreds of small contours per tile and saved
each as a 512x512 crop centred on the contour. Result: the "letter" was a
small contour inside a much larger crop — visually it didn't look like a
letter at all.

This module takes a different approach (NASA "Your Name in Landsat" style):

  - For each tile, find the **dominant shape** at the tile's scale — i.e. the
    largest connected region under a colour-cluster lens.
  - Match that dominant shape against the A–Z templates using a richer
    descriptor than Hu-moments-alone:
        * **hole count** must match exactly (kills 80%+ of false positives)
        * **solidity** must be in a sensible band per letter
        * **aspect ratio** must be in a sensible band per letter
        * Hu moments distance breaks ties
  - The **crop is the entire tile**, so the letter dominates the frame.
  - Run the matcher at 1.0× and 0.5× scales so we catch shapes that fill the
    tile *and* shapes that fill half the tile.

The output is at most one glyph per (tile × scale × cluster) combination,
not per contour, so we get tens of clean candidates per region rather than
hundreds of noisy ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Per-letter descriptor expectations
# ---------------------------------------------------------------------------
# Solidity = area / convex_hull_area. Aspect = w / h.
# Bands tightened (Dec 2026) after Kutch test showed too-loose bands let
# random blobs through. These ranges come from measuring rendered Noto Sans
# templates ±15%; further tuning happens via gallery curation.
LETTER_DESCRIPTORS = {
    # letter: (holes, solidity_lo, solidity_hi, aspect_lo, aspect_hi)
    "A": (1, 0.50, 0.68, 0.65, 1.00),
    "B": (2, 0.55, 0.72, 0.50, 0.80),
    "C": (0, 0.38, 0.58, 0.75, 1.15),
    "D": (1, 0.60, 0.78, 0.65, 0.95),
    "E": (0, 0.35, 0.50, 0.50, 0.75),
    "F": (0, 0.28, 0.45, 0.45, 0.70),
    "G": (0, 0.45, 0.65, 0.75, 1.10),
    "H": (0, 0.38, 0.55, 0.75, 1.15),
    "I": (0, 0.20, 0.40, 0.15, 0.40),
    "J": (0, 0.28, 0.50, 0.35, 0.65),
    "K": (0, 0.40, 0.58, 0.65, 1.00),
    "L": (0, 0.30, 0.48, 0.50, 0.78),
    "M": (0, 0.50, 0.70, 0.95, 1.30),
    "N": (0, 0.45, 0.65, 0.75, 1.10),
    "O": (1, 0.70, 0.88, 0.80, 1.20),
    "P": (1, 0.50, 0.65, 0.50, 0.78),
    "Q": (1, 0.65, 0.82, 0.80, 1.20),
    "R": (1, 0.50, 0.68, 0.55, 0.85),
    "S": (0, 0.40, 0.60, 0.55, 0.85),
    "T": (0, 0.25, 0.45, 0.75, 1.10),
    "U": (0, 0.45, 0.65, 0.75, 1.10),
    "V": (0, 0.45, 0.65, 0.80, 1.20),
    "W": (0, 0.50, 0.70, 1.25, 1.70),
    "X": (0, 0.40, 0.58, 0.85, 1.20),
    "Y": (0, 0.40, 0.60, 0.65, 1.00),
    "Z": (0, 0.45, 0.65, 0.75, 1.10),
}


@dataclass
class ShapeDescriptors:
    """All descriptors we care about for a connected component."""
    contour: np.ndarray          # the outer contour
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    area: int
    holes: int                   # number of interior contours
    solidity: float              # area / convex_hull_area
    aspect: float                # w / h
    hu: np.ndarray               # 7 Hu moments (already log-scaled)
    hole_contours: List[np.ndarray] = None  # interior child contours


def _hu_log(contour: np.ndarray) -> np.ndarray:
    m = cv2.moments(contour)
    h = cv2.HuMoments(m).flatten()
    # log-scale, sign-preserving — same form cv2.matchShapes uses internally.
    h = -np.sign(h) * np.log10(np.abs(h) + 1e-30)
    return h


def _make_filled_mask(contour: np.ndarray, h: int, w: int) -> np.ndarray:
    """Return a uint8 mask the same size as the tile with the contour drawn
    filled (with all its holes carved out — via even-odd polygon fill is too
    expensive; we approximate by filling the outer then subtracting any
    children, but in practice describe_mask passes us only outer contours
    so we just fill it solid)."""
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(m, [contour], -1, 255, thickness=cv2.FILLED)
    return m


def iou_against_template(cand_contour: np.ndarray,
                         cand_holes_contours: List[np.ndarray],
                         tmpl_mask_canvas: np.ndarray,
                         try_rotations: Tuple[int, ...] = (0, 90, 180, 270),
                         try_mirror: bool = True) -> float:
    """Resize candidate filled silhouette to template canvas, try a small set
    of orientations, return the best IoU.

    The whole point: Hu moments can be fooled by coastlines that happen to
    have B-ish solidity / hole count. Pixel IoU asks the harder question —
    "if I overlay the candidate silhouette on the rendered letter, how much
    do they actually overlap?". A coastline gets ~0.2, a real letter ~0.6+.

    ``tmpl_mask_canvas`` is the template's filled binary mask on a square
    canvas. We resize the candidate's filled mask to the same canvas and
    score IoU.
    """
    # Build the candidate's filled mask within a tight square canvas the same
    # size as the template, so the comparison is shape-only (translation /
    # scale already normalised).
    canvas_size = tmpl_mask_canvas.shape[0]
    x, y, w, h = cv2.boundingRect(cand_contour)
    if w <= 1 or h <= 1:
        return 0.0
    # Render the candidate into a tight canvas, preserving aspect ratio,
    # centred on a square of the template canvas size.
    aspect = w / h
    if aspect >= 1.0:
        new_w = canvas_size
        new_h = max(1, int(round(canvas_size / aspect)))
    else:
        new_h = canvas_size
        new_w = max(1, int(round(canvas_size * aspect)))
    # Translate the contour to the bbox origin.
    cand_local = cand_contour.copy()
    cand_local[..., 0] -= x
    cand_local[..., 1] -= y
    # Scale.
    sx = new_w / w
    sy = new_h / h
    cand_local = cand_local.astype(np.float32)
    cand_local[..., 0] *= sx
    cand_local[..., 1] *= sy
    cand_local = np.round(cand_local).astype(np.int32)

    # Centre on a square canvas.
    cand_filled = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    off_x = (canvas_size - new_w) // 2
    off_y = (canvas_size - new_h) // 2
    cand_local[..., 0] += off_x
    cand_local[..., 1] += off_y
    cv2.drawContours(cand_filled, [cand_local], -1, 255, thickness=cv2.FILLED)
    # Carve out holes (interior children) the same way.
    for hc in cand_holes_contours:
        if hc.size == 0:
            continue
        hc_local = hc.copy()
        hc_local[..., 0] -= x
        hc_local[..., 1] -= y
        hc_local = hc_local.astype(np.float32)
        hc_local[..., 0] *= sx
        hc_local[..., 1] *= sy
        hc_local = np.round(hc_local).astype(np.int32)
        hc_local[..., 0] += off_x
        hc_local[..., 1] += off_y
        cv2.drawContours(cand_filled, [hc_local], -1, 0, thickness=cv2.FILLED)

    tmpl_bool = tmpl_mask_canvas > 127
    best = 0.0
    for rot in try_rotations:
        rotated = cand_filled
        if rot == 90:
            rotated = cv2.rotate(cand_filled, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            rotated = cv2.rotate(cand_filled, cv2.ROTATE_180)
        elif rot == 270:
            rotated = cv2.rotate(cand_filled, cv2.ROTATE_90_COUNTERCLOCKWISE)
        variants = [rotated]
        if try_mirror:
            variants.append(cv2.flip(rotated, 1))
        for v in variants:
            v_bool = v > 127
            inter = np.logical_and(v_bool, tmpl_bool).sum()
            union = np.logical_or(v_bool, tmpl_bool).sum()
            if union == 0:
                continue
            iou = inter / union
            if iou > best:
                best = float(iou)
    return best


def describe_mask(mask: np.ndarray) -> Optional[ShapeDescriptors]:
    """Describe the largest connected component of a binary mask.

    Returns None if there's no component or it has zero area.
    """
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours or hierarchy is None:
        return None
    hierarchy = hierarchy[0]
    # Outer contours have parent == -1.
    outers = [(i, c) for i, c in enumerate(contours) if hierarchy[i][3] == -1]
    if not outers:
        return None
    # Pick the biggest outer contour.
    idx, contour = max(outers, key=lambda ic: cv2.contourArea(ic[1]))
    area = int(cv2.contourArea(contour))
    if area <= 0:
        return None
    # Holes = children of this outer contour. Keep both the count and the
    # actual hole contours so pixel-IoU can carve them out of the silhouette.
    hole_contours = [
        contours[i] for i in range(len(contours))
        if hierarchy[i][3] == idx
    ]
    holes = len(hole_contours)
    x, y, w, h = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour))) or 1.0
    solidity = area / hull_area
    aspect = w / h if h > 0 else 0.0
    return ShapeDescriptors(
        contour=contour, bbox=(x, y, w, h), area=area, holes=holes,
        solidity=solidity, aspect=aspect, hu=_hu_log(contour),
        hole_contours=hole_contours,
    )


def describe_template_image(letter_mask: np.ndarray) -> ShapeDescriptors:
    """Same as describe_mask but for a rendered letter template (guaranteed
    to have at least one component)."""
    d = describe_mask(letter_mask)
    if d is None:
        # Should never happen for a properly-rendered letter, but be safe.
        empty = np.array([[[0, 0]], [[1, 0]], [[0, 1]]], dtype=np.int32)
        return ShapeDescriptors(contour=empty, bbox=(0, 0, 1, 1), area=1,
                                holes=0, solidity=1.0, aspect=1.0,
                                hu=np.zeros(7))
    return d


# ---------------------------------------------------------------------------
# Tile → candidates
# ---------------------------------------------------------------------------

def _clean_mask(mask: np.ndarray, tile_size: int) -> np.ndarray:
    """Aggressively clean a binary mask so its largest component has
    letter-like topology (a handful of holes, not hundreds).

    The k-means / Otsu masks are full of small speckle holes that make
    ``cv2.findContours(..., RETR_CCOMP)`` return enormous hole counts. To
    get the hole-count gate to mean anything we have to first fill in
    speckle smaller than ~1% of the tile and remove pepper.

    Kernel size scales with tile size so a 1024-tile uses a ~21px kernel
    while a 256-tile uses a ~5px one.
    """
    k = max(5, tile_size // 48) | 1  # odd, ~21 for 1024, ~11 for 512
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)
    # Drop holes smaller than ~0.6% of tile area: paint them in.
    h, w = m.shape
    min_hole_area = max(64, int(0.006 * h * w))
    inv = cv2.bitwise_not(m)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_hole_area:
            m[labels == i] = 255
    # Also drop tiny foreground islands.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_island_area = max(64, int(0.006 * h * w))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_island_area:
            m[labels == i] = 0
    return m


def _kmeans_cluster_masks(img_bgr: np.ndarray, k: int = 5,
                          downscale: int = 4) -> List[np.ndarray]:
    """HSV k-means → one binary mask per cluster, upsampled and aggressively
    cleaned so hole counts reflect topology, not speckle."""
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
    masks: List[np.ndarray] = []
    for ci in range(k):
        m = (labels == ci).astype(np.uint8) * 255
        m_full = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m_full = _clean_mask(m_full, tile_size=max(h, w))
        masks.append(m_full)
    return masks


def _otsu_masks(img_bgr: np.ndarray) -> List[np.ndarray]:
    """Otsu (light vs dark), aggressively cleaned."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = m.shape
    m = _clean_mask(m, tile_size=max(h, w))
    inv = _clean_mask(cv2.bitwise_not(m), tile_size=max(h, w))
    return [m, inv]


def candidate_descriptors_for_tile(
    img_bgr: np.ndarray,
    min_area_frac: float = 0.04,
    max_area_frac: float = 0.85,
    k_landuse: int = 5,
) -> List[ShapeDescriptors]:
    """Return up to ~(k+2) candidate shape descriptors from one tile.

    For each binary mask (Otsu light/dark + k_landuse k-means clusters), we
    take the **largest connected component** and compute descriptors for it.
    Components that are too small or too large relative to the tile are
    dropped: the goal is "dominant shape at this scale", not "tiny detail"
    nor "the whole tile minus border".
    """
    h, w = img_bgr.shape[:2]
    tile_area = h * w
    min_area = int(tile_area * min_area_frac)
    max_area = int(tile_area * max_area_frac)

    candidates: List[ShapeDescriptors] = []
    for mask in _otsu_masks(img_bgr) + _kmeans_cluster_masks(img_bgr, k=k_landuse):
        d = describe_mask(mask)
        if d is None:
            continue
        if d.area < min_area or d.area > max_area:
            continue
        candidates.append(d)
    return candidates


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_against_letter(
    cand: ShapeDescriptors,
    tmpl,  # LetterTemplate
    letter: str,
    min_iou: float = 0.45,
) -> Optional[Tuple[float, float]]:
    """Score ``cand`` against the letter template.

    Pipeline of hard gates → soft score:
      1. hole count must match exactly
      2. solidity / aspect must be in the per-letter band
      3. pixel IoU vs the template silhouette must be >= ``min_iou``
      4. final score combines (1 - IoU), Hu distance, and band centring

    Returns (score, iou) or None if any hard gate fails. Lower score = better.
    """
    spec = LETTER_DESCRIPTORS.get(letter)
    if spec is None:
        return None
    holes_req, sol_lo, sol_hi, asp_lo, asp_hi = spec

    if cand.holes != holes_req:
        return None
    if cand.solidity < sol_lo or cand.solidity > sol_hi:
        return None
    if cand.aspect < asp_lo or cand.aspect > asp_hi:
        return None

    # Pixel-IoU gate — the hard geometric check that distinguishes a B from a
    # B-ish coastline.
    iou = iou_against_template(
        cand.contour, cand.hole_contours or [], tmpl.canvas_mask,
    )
    if iou < min_iou:
        return None

    # Hu moments distance (smaller = better).
    hu_dist = float(np.sum(np.abs(1.0 / cand.hu - 1.0 / tmpl.descriptors.hu)))

    # Band centring penalties.
    sol_centre = 0.5 * (sol_lo + sol_hi)
    asp_centre = 0.5 * (asp_lo + asp_hi)
    sol_pen = abs(cand.solidity - sol_centre)
    asp_pen = abs(cand.aspect - asp_centre)

    # Combine: IoU dominates (higher IoU = lower score), Hu / band centring
    # break ties.
    score = (1.0 - iou) * 4.0 + 0.4 * hu_dist + 0.5 * sol_pen + 0.3 * asp_pen
    return score, iou


def best_letter_for(
    cand: ShapeDescriptors,
    templates: dict,  # letter -> LetterTemplate
    threshold: float,
    min_iou: float = 0.45,
) -> Optional[Tuple[str, float, float]]:
    """Return (letter, score, iou) for the best-matching letter, or None."""
    best: Optional[Tuple[str, float, float]] = None
    for letter, tmpl in templates.items():
        r = score_against_letter(cand, tmpl, letter, min_iou=min_iou)
        if r is None:
            continue
        s, iou = r
        if s > threshold:
            continue
        if best is None or s < best[1]:
            best = (letter, s, iou)
    return best


# ---------------------------------------------------------------------------
# Letter templates with full descriptors
# ---------------------------------------------------------------------------

@dataclass
class LetterTemplate:
    """A rendered letter: descriptors + a fixed-size filled silhouette mask
    used for pixel-IoU comparison."""
    descriptors: ShapeDescriptors
    canvas_mask: np.ndarray   # square binary mask, letter centred & tight


def _render_letter_canvas(font_path, font_size: int, canvas: int,
                          letter: str) -> np.ndarray:
    """Render ``letter`` centred in a square canvas, return binary mask."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(font_path), font_size)
    img = Image.new("L", (canvas, canvas), 0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), letter, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas - tw) // 2 - bbox[0]
    y = (canvas - th) // 2 - bbox[1]
    draw.text((x, y), letter, fill=255, font=font)
    arr = np.array(img)
    _, mask = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY)
    return mask


def _tight_letter_canvas(font_path, font_size: int, target_size: int,
                         letter: str) -> np.ndarray:
    """Render the letter as tightly as possible into a square canvas of
    ``target_size``, preserving aspect ratio.

    Used as the template for pixel-IoU: the candidate is resized to the
    same target_size and then compared.
    """
    # Render large, find tight bbox, scale to target.
    big = _render_letter_canvas(font_path, font_size=200, canvas=512,
                                letter=letter)
    ys, xs = np.where(big > 0)
    if len(xs) == 0:
        return big
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    cropped = big[y0:y1, x0:x1]
    h, w = cropped.shape
    aspect = w / h
    if aspect >= 1.0:
        new_w = target_size
        new_h = max(1, int(round(target_size / aspect)))
    else:
        new_h = target_size
        new_w = max(1, int(round(target_size * aspect)))
    resized = cv2.resize(cropped, (new_w, new_h),
                         interpolation=cv2.INTER_AREA)
    canvas_img = np.zeros((target_size, target_size), dtype=np.uint8)
    off_x = (target_size - new_w) // 2
    off_y = (target_size - new_h) // 2
    canvas_img[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    _, canvas_img = cv2.threshold(canvas_img, 128, 255, cv2.THRESH_BINARY)
    return canvas_img


def render_letter_templates(font_path, font_size: int = 140,
                            canvas: int = 256) -> dict:
    """Render A–Z and return ``{letter: LetterTemplate}``.

    Each LetterTemplate carries both the ShapeDescriptors (for the cheap
    Hu/solidity/aspect gates) and a tight square canvas mask for the
    pixel-IoU stage.
    """
    out: dict = {}
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        mask = _render_letter_canvas(font_path, font_size, canvas, letter)
        descriptors = describe_template_image(mask)
        canvas_mask = _tight_letter_canvas(font_path, font_size, canvas, letter)
        out[letter] = LetterTemplate(descriptors=descriptors,
                                     canvas_mask=canvas_mask)
    return out
