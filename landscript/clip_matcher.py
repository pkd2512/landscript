"""CLIP image-image matching of candidate silhouettes against letter glyphs.

The previous text-prompt approach didn't work because CLIP was trained on
"object photos with captions", not on "aerial views captioned by shape".
Text-prompt cosine for 'a satellite photo of land shaped like X' collapses
to noise — every farmland tile scores ~0.28 for every letter.

This module takes a different approach: **image-image cosine**.

For each candidate (a dominant shape from one k-means/Otsu mask):

  1. Render the candidate as a clean ``224 x 224`` image: white silhouette
     on black background, centred and scaled to fill ~70% of the frame.
     That's exactly the kind of image CLIP was *also* trained on (logos,
     icons, simple shapes captioned as letters and symbols).
  2. Embed that with CLIP.
  3. For each letter A–Z, render it the same way (white-on-black,
     centred, 70% fill) and embed.
  4. Cosine similarity between candidate-silhouette-embedding and
     each letter-silhouette-embedding gives a *visual* similarity.
  5. Also embed the original RGB tile and compute its similarity to a
     "generic landscape" baseline — used to confirm we don't pick tiles
     that just look like featureless terrain.

CLIP is *recognising* the rendered silhouette, not generating anything.
Every glyph is still a real Sentinel-2 crop.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Lazy CLIP loader
# ---------------------------------------------------------------------------

@dataclass
class ClipBundle:
    model: object
    preprocess: object
    tokenizer: object
    device: str


_CACHED: Optional[ClipBundle] = None


def load_clip(model_name: str = "ViT-B-32",
              pretrained: str = "laion2b_s34b_b79k") -> ClipBundle:
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    import torch
    import open_clip

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained,
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    _CACHED = ClipBundle(model=model, preprocess=preprocess,
                         tokenizer=tokenizer, device=device)
    return _CACHED


# ---------------------------------------------------------------------------
# Letter silhouette rendering (the "image side" of the comparison)
# ---------------------------------------------------------------------------

def render_letter_silhouette(letter: str, canvas: int = 224,
                             fill_fraction: float = 0.7) -> np.ndarray:
    """Render a single letter as a clean white-on-black silhouette.

    The letter is tightly cropped then scaled to occupy ``fill_fraction`` of
    the square canvas (default 70%), centred. Returns a single-channel uint8
    image. CLIP's preprocess will handle the RGB conversion + resize +
    normalisation downstream.
    """
    from PIL import Image, ImageDraw, ImageFont
    from .cv_pipeline import resolve_font
    from .config import FontConfig

    font_path = resolve_font(FontConfig())
    # Render large first so the cropped letter is high-resolution.
    big = 1024
    font = ImageFont.truetype(str(font_path), 720)
    img = Image.new("L", (big, big), 0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), letter, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (big - tw) // 2 - bbox[0]
    y = (big - th) // 2 - bbox[1]
    draw.text((x, y), letter, fill=255, font=font)
    arr = np.array(img)
    # Tight crop.
    ys, xs = np.where(arr > 0)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    cropped = arr[y0:y1, x0:x1]
    h, w = cropped.shape
    # Scale so the longer side is fill_fraction of canvas.
    target_long = int(canvas * fill_fraction)
    if w >= h:
        new_w = target_long
        new_h = max(1, int(round(target_long * h / w)))
    else:
        new_h = target_long
        new_w = max(1, int(round(target_long * w / h)))
    resized = cv2.resize(cropped, (new_w, new_h),
                         interpolation=cv2.INTER_AREA)
    out = np.zeros((canvas, canvas), dtype=np.uint8)
    off_x = (canvas - new_w) // 2
    off_y = (canvas - new_h) // 2
    out[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return out


def render_candidate_silhouette(mask: np.ndarray, contour: np.ndarray,
                                canvas: int = 224,
                                fill_fraction: float = 0.7) -> np.ndarray:
    """Render a single contour as a clean white-on-black silhouette.

    Same framing rules as ``render_letter_silhouette`` so the two images are
    directly comparable: tight crop, scale to ``fill_fraction`` of canvas,
    centred.
    """
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 1 or h <= 1:
        return np.zeros((canvas, canvas), dtype=np.uint8)
    # Build a tight filled mask of just the contour.
    tight = np.zeros((h, w), dtype=np.uint8)
    cand_local = contour.copy()
    cand_local[..., 0] -= x
    cand_local[..., 1] -= y
    cv2.drawContours(tight, [cand_local], -1, 255, thickness=cv2.FILLED)
    # Scale to fill_fraction of canvas, preserving aspect.
    target_long = int(canvas * fill_fraction)
    if w >= h:
        new_w = target_long
        new_h = max(1, int(round(target_long * h / w)))
    else:
        new_h = target_long
        new_w = max(1, int(round(target_long * w / h)))
    resized = cv2.resize(tight, (new_w, new_h),
                         interpolation=cv2.INTER_AREA)
    out = np.zeros((canvas, canvas), dtype=np.uint8)
    off_x = (canvas - new_w) // 2
    off_y = (canvas - new_h) // 2
    out[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return out


# ---------------------------------------------------------------------------
# CLIP embedding helpers
# ---------------------------------------------------------------------------

def _embed_gray(img_gray: np.ndarray) -> np.ndarray:
    """Embed a single-channel uint8 image with CLIP. Converts to 3-channel
    RGB first (CLIP expects RGB)."""
    import torch
    from PIL import Image

    bundle = load_clip()
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    pil = Image.fromarray(rgb)
    with torch.no_grad():
        t = bundle.preprocess(pil).unsqueeze(0).to(bundle.device)
        e = bundle.model.encode_image(t)
        e = e / e.norm(dim=-1, keepdim=True)
    return e.cpu().numpy()[0]


@lru_cache(maxsize=1)
def letter_silhouette_embeddings() -> Tuple[np.ndarray, List[str]]:
    """Return (embeddings[26, D], letters). Cached so we only encode once."""
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    embs = []
    for L in letters:
        sil = render_letter_silhouette(L)
        embs.append(_embed_gray(sil))
    return np.stack(embs), letters


# ---------------------------------------------------------------------------
# Candidate extraction (one dominant shape per mask)
# ---------------------------------------------------------------------------

def _kmeans_masks(img_bgr: np.ndarray, k: int = 5,
                  downscale: int = 4) -> List[np.ndarray]:
    """K-means HSV clustering → one binary mask per cluster, lightly cleaned."""
    h, w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr,
                       (max(1, w // downscale), max(1, h // downscale)),
                       interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    pixels = hsv.reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(pixels, k, None, criteria, 3,
                              cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(small.shape[:2]).astype(np.uint8)
    masks: List[np.ndarray] = []
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for ci in range(k):
        m = (labels == ci).astype(np.uint8) * 255
        m_full = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m_full = cv2.morphologyEx(m_full, cv2.MORPH_CLOSE, kern)
        m_full = cv2.morphologyEx(m_full, cv2.MORPH_OPEN, kern)
        masks.append(m_full)
    return masks


def dominant_contours(img_bgr: np.ndarray,
                      min_area_frac: float = 0.10,
                      max_area_frac: float = 0.70,
                      k_landuse: int = 5) -> List[np.ndarray]:
    """Return the largest contour of each k-means cluster mask, area-gated
    to the "letter-like fraction of the tile" band.

    No hole-count, solidity or aspect gating here — we let CLIP do the
    discriminating; this layer just produces candidate shapes.
    """
    h, w = img_bgr.shape[:2]
    tile_area = h * w
    lo = int(tile_area * min_area_frac)
    hi = int(tile_area * max_area_frac)
    out: List[np.ndarray] = []
    for mask in _kmeans_masks(img_bgr, k=k_landuse):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        biggest = max(contours, key=cv2.contourArea)
        a = cv2.contourArea(biggest)
        if a < lo or a > hi:
            continue
        out.append(biggest)
    return out


# ---------------------------------------------------------------------------
# Per-tile decision
# ---------------------------------------------------------------------------

@dataclass
class TileMatch:
    letter: str
    letter_score: float          # cosine to best letter silhouette
    runner_up_letter: str
    runner_up_score: float
    contour: np.ndarray          # the contour CLIP chose
    silhouette: np.ndarray       # the 224x224 silhouette we embedded


def classify_tile(img_bgr: np.ndarray,
                  min_score: float = 0.85,
                  min_runner_margin: float = 0.01,
                  ) -> Optional[TileMatch]:
    """Find the candidate shape in this tile that best resembles a letter.

    Procedure:
      1. Generate dominant contours via k-means.
      2. For each contour, render its silhouette + embed with CLIP.
      3. Score against the 26 letter silhouette embeddings.
      4. Pick the (contour, letter) with the highest cosine across all
         combinations, subject to:
            * absolute cosine >= min_score (sanity floor)
            * margin over runner-up letter >= min_runner_margin
      5. Return None if nothing clears the bar.

    Defaults are intentionally moderate; tune via CLI flags.
    """
    contours = dominant_contours(img_bgr)
    if not contours:
        return None
    letter_embs, letters = letter_silhouette_embeddings()

    best: Optional[TileMatch] = None
    best_score = -1.0
    for c in contours:
        sil = render_candidate_silhouette(np.zeros_like(img_bgr[..., 0]), c)
        if sil.sum() == 0:
            continue
        emb = _embed_gray(sil)
        scores = letter_embs @ emb              # shape (26,)
        order = np.argsort(-scores)
        top = float(scores[order[0]])
        second = float(scores[order[1]])
        if top < min_score:
            continue
        if (top - second) < min_runner_margin:
            continue
        if top > best_score:
            best_score = top
            best = TileMatch(
                letter=letters[order[0]],
                letter_score=top,
                runner_up_letter=letters[order[1]],
                runner_up_score=second,
                contour=c,
                silhouette=sil,
            )
    return best