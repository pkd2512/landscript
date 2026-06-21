import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from skimage import exposure
from PIL import Image, ImageDraw, ImageFont
from .config import PipelineConfig, FontConfig


def load_tile(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_threshold(gray: np.ndarray, method: str = "otsu") -> np.ndarray:
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    elif method == "triangle":
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    else:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def find_contours(binary: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contours(contours: List[np.ndarray],
                    cfg: PipelineConfig) -> List[np.ndarray]:
    return [
        c for c in contours
        if cfg.min_contour_area < cv2.contourArea(c) < cfg.max_contour_area
    ]


def contour_to_polygon(contour: np.ndarray,
                       epsilon: float = 0.02) -> np.ndarray:
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon * peri, True)


def match_shape(contour: np.ndarray,
                template: np.ndarray) -> float:
    return cv2.matchShapes(contour, template, cv2.CONTOURS_MATCH_I2, 0)


def resolve_font(font_cfg: FontConfig) -> Path:
    local = font_cfg.local_path
    if local is not None and local.exists():
        print(f"  Using local font: {local}")
        return local

    local = local or Path("fonts") / font_cfg.url.rstrip("/").split("/")[-1]
    print(f"  Font not found locally. Downloading from {font_cfg.url}...")
    import requests
    local.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(font_cfg.url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to download font from {font_cfg.url}: {resp.status_code}"
        )
    local.write_bytes(resp.content)
    print(f"  Saved font to {local}")
    return local


class LetterTemplate:
    def __init__(self, letter: str, contour: np.ndarray):
        self.letter = letter
        self.contour = contour


def load_letter_templates(cfg: PipelineConfig) -> List[LetterTemplate]:
    from tqdm import tqdm
    log("fonts", "Resolving font...")
    font_path = resolve_font(cfg.font)
    log("fonts", f"Loading font: {cfg.font.family} ({font_path.name})")
    font = ImageFont.truetype(str(font_path), cfg.font.size)

    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    templates = []
    size = 200

    for letter in tqdm(letters, desc="  Letters", unit="letter"):
        img = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), letter, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size - tw) // 2 - bbox[0]
        y = (size - th) // 2 - bbox[1]
        draw.text((x, y), letter, fill=255, font=font)

        np_img = np.array(img)
        _, binary = cv2.threshold(np_img, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            templates.append(LetterTemplate(letter, largest))

    log("fonts", f"{len(templates)} templates ready")
    return templates


def log(step: str, msg: str):
    print(f"  [{step}] {msg}")


def process_tile(
    tile_path: Path,
    cfg: PipelineConfig,
    templates: List[LetterTemplate],
) -> List[dict]:
    img = load_tile(tile_path)
    if img is None:
        return []

    gray = to_grayscale(img)
    binary = apply_threshold(gray, cfg.threshold_method)
    contours = find_contours(binary)
    contours = filter_contours(contours, cfg)

    candidates = []
    for i, contour in enumerate(contours):
        poly = contour_to_polygon(contour, cfg.epsilon)
        x, y, w, h = cv2.boundingRect(contour)

        for tmpl in templates:
            score = match_shape(poly, tmpl.contour)
            if score < cfg.similarity_threshold:
                candidates.append({
                    "letter": tmpl.letter,
                    "score": float(score),
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "contour_index": i,
                })

    return candidates


def extract_glyph_crop(
    img: np.ndarray,
    contour: np.ndarray,
    output_path: Path,
    out_w: int = 1024,
    out_h: int = 2048,
    max_black_pct: float = 20.0,
) -> Optional[Path]:
    x, y, w, h = cv2.boundingRect(contour)
    cx, cy = x + w // 2, y + h // 2

    left = cx - out_w // 2
    top = cy - out_h // 2
    right = left + out_w
    bottom = top + out_h

    tile_h, tile_w = img.shape[:2]
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    mask = np.zeros((out_h, out_w), dtype=np.uint8)

    src_l = max(0, left)
    src_t = max(0, top)
    src_r = min(tile_w, right)
    src_b = min(tile_h, bottom)
    dst_l = src_l - left
    dst_t = src_t - top
    dst_r = dst_l + (src_r - src_l)
    dst_b = dst_t + (src_b - src_t)

    if dst_r > dst_l and dst_b > dst_t:
        canvas[dst_t:dst_b, dst_l:dst_r] = img[src_t:src_b, src_l:src_r]
        mask[dst_t:dst_b, dst_l:dst_r] = 255

    black_pct = 100.0 * (1.0 - mask.mean() / 255.0)
    if black_pct > max_black_pct:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def cleanup_glyph(glyph_path: Path, store, glyph_id: str):
    """Remove a glyph file and its metadata entry."""
    if glyph_path.exists():
        glyph_path.unlink()
    store.delete(glyph_id)
