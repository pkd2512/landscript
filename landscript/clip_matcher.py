"""CLIP as the primary matcher.

Instead of "find a contour, score it against templates", we treat each
satellite tile as one image and ask CLIP directly:

    "Which capital letter, if any, does this picture most resemble?"

The procedure for one tile:

  1. Encode the image once.
  2. Compare to 26 letter prompts and ~6 negative prompts (farmland,
     coastline-no-letter, river bend, lake, fields, generic landscape).
  3. The winning letter is the one with highest cosine similarity, but it
     must beat **every** negative by a margin to be kept. If the top
     negative beats every letter, the tile is rejected.
  4. Also require the top letter to beat the second-best letter by a
     smaller margin — so a tile that's "kind of B and kind of D" gets
     dropped as ambiguous.

This is pure CLIP — no Hu moments, no descriptor bands, no IoU. The crop is
the entire tile and gets saved as-is, with a small letter-and-score badge.

CLIP is *recognising* the imagery, not generating anything; every glyph is
still a real Sentinel-2 crop.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Lazy CLIP loader (shared with clip_rerank.py if both are imported)
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
# Prompt sets
# ---------------------------------------------------------------------------

def _letter_prompts(letter: str) -> List[str]:
    """A few phrasings averaged together to denoise CLIP's score for
    ``letter``."""
    L = letter.upper()
    return [
        f"a satellite photo of land naturally arranged in the shape of the capital letter {L}",
        f"an aerial view where the landscape clearly forms the letter {L}",
        f"a top-down photograph of terrain that looks exactly like the letter {L}",
    ]


def _negative_prompts() -> List[str]:
    return [
        "a satellite photo of farmland with no recognisable letter",
        "an aerial view of a coastline with no letter shape",
        "a satellite photo of a river bend",
        "an aerial photo of a lake or reservoir",
        "a satellite image of fields, forests and roads",
        "a generic top-down landscape photo with no letter visible",
    ]


# ---------------------------------------------------------------------------
# Embedding helpers (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _all_text_embeddings() -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Return:
        - letter_embs   shape (26, D); each row is the mean of that letter's
                         prompt embeddings, L2-normalised
        - letters       list of 26 letters in the same order
        - neg_embs      shape (N_neg, D); each row is one negative prompt
                         embedding, L2-normalised
    """
    import torch

    bundle = load_clip()
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    all_letter_texts: List[str] = []
    counts: List[int] = []
    for L in letters:
        texts = _letter_prompts(L)
        all_letter_texts.extend(texts)
        counts.append(len(texts))
    neg_texts = _negative_prompts()

    with torch.no_grad():
        tok = bundle.tokenizer(all_letter_texts + neg_texts).to(bundle.device)
        emb = bundle.model.encode_text(tok)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb_np = emb.cpu().numpy()

    # Average each letter's per-prompt embeddings, then re-normalise.
    letter_embs = np.zeros((26, emb_np.shape[1]), dtype=np.float32)
    cursor = 0
    for i, n in enumerate(counts):
        chunk = emb_np[cursor:cursor + n]
        v = chunk.mean(axis=0)
        v /= (np.linalg.norm(v) + 1e-12)
        letter_embs[i] = v
        cursor += n
    neg_embs = emb_np[cursor:]
    return letter_embs, letters, neg_embs


def _image_embedding(img_bgr: np.ndarray) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Per-tile decision
# ---------------------------------------------------------------------------

@dataclass
class TileMatch:
    letter: str
    letter_score: float       # cosine vs that letter's prompt centroid
    runner_up_letter: str
    runner_up_score: float
    best_negative: float      # max cosine vs any negative prompt
    margin_letter_vs_runner: float    # letter_score - runner_up_score
    margin_letter_vs_negative: float  # letter_score - best_negative


def classify_tile(img_bgr: np.ndarray,
                  min_margin_vs_negative: float = 0.02,
                  min_margin_vs_runner: float = 0.005) -> Optional[TileMatch]:
    """Return the best letter for ``img_bgr`` or None if no letter clearly
    beats both the negatives and the runner-up letter.

    Defaults are intentionally permissive — we want a decent candidate set
    that humans then curate. Tighten to be stricter.
    """
    letter_embs, letters, neg_embs = _all_text_embeddings()
    img_emb = _image_embedding(img_bgr)

    letter_scores = letter_embs @ img_emb          # shape (26,)
    neg_scores = neg_embs @ img_emb                # shape (N_neg,)

    order = np.argsort(-letter_scores)
    top_i = int(order[0])
    second_i = int(order[1])
    best_neg = float(neg_scores.max())
    top_score = float(letter_scores[top_i])
    second_score = float(letter_scores[second_i])

    margin_neg = top_score - best_neg
    margin_runner = top_score - second_score
    if margin_neg < min_margin_vs_negative:
        return None
    if margin_runner < min_margin_vs_runner:
        return None
    return TileMatch(
        letter=letters[top_i],
        letter_score=top_score,
        runner_up_letter=letters[second_i],
        runner_up_score=second_score,
        best_negative=best_neg,
        margin_letter_vs_runner=margin_runner,
        margin_letter_vs_negative=margin_neg,
    )