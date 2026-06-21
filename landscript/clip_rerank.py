"""CLIP-based reranking of glyph candidates.

The classical-CV pipeline (tile_matcher) is *fast* but only measures outline
statistics — it can't tell a coastline from the letter B. CLIP can: it was
trained on hundreds of millions of captioned images and has a real prior on
"does this picture look like the letter X".

We use CLIP **only as a reranker**:

  1. The classical pipeline produces ~10–100 candidates per region (cheap, no
     ML).
  2. For each candidate we ask CLIP: how well does this *image* match the
     prompt "a satellite photo where the land forms the shape of the letter
     <X>" compared to a small set of negative prompts?
  3. Candidates whose positive-prompt margin over the negatives is below a
     threshold are dropped.

CLIP does **not** generate any image. It only scores the satellite imagery
that the existing pipeline already extracted. Every glyph still comes from
real Sentinel-2 data; CLIP just filters which ones we keep.

Dependencies (optional extras):
    pip install open_clip_torch torch

The first call to ``load_clip()`` downloads the model weights (~150 MB for
ViT-B/32) into the standard HF cache. On Apple Silicon we use MPS, otherwise
CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Model loading (lazy / cached)
# ---------------------------------------------------------------------------

@dataclass
class ClipBundle:
    model: object       # open_clip model
    preprocess: object  # transform
    tokenizer: object
    device: str


_CACHED: Optional[ClipBundle] = None


def load_clip(model_name: str = "ViT-B-32",
              pretrained: str = "laion2b_s34b_b79k") -> ClipBundle:
    """Load (and cache) an open-clip model. Picks MPS on macOS Apple Silicon."""
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
# Prompts
# ---------------------------------------------------------------------------

def positive_prompts(letter: str) -> List[str]:
    """Several phrasings to average over — CLIP scores are noisy on a single
    prompt; averaging several stabilises the signal."""
    L = letter.upper()
    return [
        f"a satellite photo where the land naturally forms the shape of the letter {L}",
        f"an aerial view of a landscape that looks like the letter {L}",
        f"a satellite image of a region shaped exactly like the capital letter {L}",
        f"a top-down photograph of land arranged in the form of the letter {L}",
    ]


def negative_prompts() -> List[str]:
    """Negatives that look superficially similar to letter candidates but
    aren't letters. The reranker keeps a candidate only if positives beat
    these by a comfortable margin."""
    return [
        "a satellite photo of farmland with no recognisable shape",
        "an aerial photo of a coastline with no particular letter shape",
        "a satellite photo of a river bend",
        "an aerial photo of a lake or reservoir",
        "a satellite photo of fields, forests and roads",
        "an abstract aerial landscape image with no letter visible",
    ]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _text_embeddings(letter: str) -> Tuple["np.ndarray", "np.ndarray"]:
    """Return (positive_embeddings, negative_embeddings) for a letter, each
    normalised. Cached: text encoding is the cheap part but cheaper still
    when reused across regions."""
    import torch

    bundle = load_clip()
    pos = positive_prompts(letter)
    neg = negative_prompts()
    with torch.no_grad():
        tok_pos = bundle.tokenizer(pos).to(bundle.device)
        tok_neg = bundle.tokenizer(neg).to(bundle.device)
        emb_pos = bundle.model.encode_text(tok_pos)
        emb_neg = bundle.model.encode_text(tok_neg)
        emb_pos = emb_pos / emb_pos.norm(dim=-1, keepdim=True)
        emb_neg = emb_neg / emb_neg.norm(dim=-1, keepdim=True)
    return emb_pos.cpu().numpy(), emb_neg.cpu().numpy()


def _image_embedding(img_bgr: "np.ndarray") -> "np.ndarray":
    """Encode an OpenCV BGR image with CLIP and L2-normalise."""
    import torch
    import cv2
    from PIL import Image

    bundle = load_clip()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    with torch.no_grad():
        t = bundle.preprocess(pil).unsqueeze(0).to(bundle.device)
        e = bundle.model.encode_image(t)
        e = e / e.norm(dim=-1, keepdim=True)
    return e.cpu().numpy()[0]


def score_candidate(img_bgr: "np.ndarray", letter: str) -> dict:
    """Return CLIP scores for ``img_bgr`` against ``letter`` prompts.

    Output dict:
        pos_mean   mean cosine vs positive prompts for ``letter``
        neg_mean   mean cosine vs the negative prompts
        margin     pos_mean - neg_mean (the actual reranking signal)
    """
    emb = _image_embedding(img_bgr)
    pos, neg = _text_embeddings(letter)
    pos_scores = pos @ emb
    neg_scores = neg @ emb
    return {
        "pos_mean": float(pos_scores.mean()),
        "neg_mean": float(neg_scores.mean()),
        "margin": float(pos_scores.mean() - neg_scores.mean()),
    }


def rerank_with_clip(items: list, get_image, get_letter,
                     min_margin: float = 0.02) -> list:
    """Filter a list of items by CLIP-derived ``margin``.

    ``items`` is any iterable; ``get_image(item)`` returns a BGR ndarray and
    ``get_letter(item)`` returns the candidate letter. Items whose margin is
    below ``min_margin`` are dropped. Surviving items get an extra
    ``clip`` key with the scores attached so the caller can record them.
    """
    kept = []
    for it in items:
        scores = score_candidate(get_image(it), get_letter(it))
        if scores["margin"] >= min_margin:
            it_out = dict(it)
            it_out["clip"] = scores
            kept.append(it_out)
    return kept