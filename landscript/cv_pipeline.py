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


def is_cloud_region(img: np.ndarray, contour: np.ndarray,
                    v_min: int = 200, s_max: int = 35,
                    bright_pixel_pct: float = 60.0) -> bool:
    """Return True if the contour interior looks like cloud cover.

    Clouds in true-color satellite imagery are bright and desaturated:
    high V (value/brightness) and low S (saturation) in HSV space.
    We mask the contour interior, then check what fraction of those
    pixels are both bright AND desaturated. If the majority are,
    we call it a cloud and reject.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    pixels = img[mask == 255]
    if pixels.size == 0:
        return False

    # cv2.imread returns BGR; convert to HSV
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    s = hsv[:, 1]
    v = hsv[:, 2]
    cloud_pixels = ((v >= v_min) & (s <= s_max)).sum()
    pct = 100.0 * cloud_pixels / len(hsv)
    return pct >= bright_pixel_pct


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
    min_crop: int = 512,
    padding: float = 0.4,
    max_black_pct: float = 20.0,
) -> Optional[Path]:
    """Crop a square region around the contour and save as PNG at native resolution.

    The crop side length is `max(contour_bbox * (1 + 2*padding), min_crop)`,
    capped to the tile dimensions. We **never upscale** — the saved PNG is
    the actual source pixels, so glyphs from small features stay sharp
    instead of becoming blurry interpolated mush.

    Returns None if too much of the requested square falls outside the tile.
    """
    tile_h, tile_w = img.shape[:2]
    x, y, w, h = cv2.boundingRect(contour)
    cx, cy = x + w // 2, y + h // 2

    # Square source crop: bbox extent + padding, with a floor so tiny contours
    # still get pulled out as a reasonably-sized image with context.
    bbox_extent = max(w, h)
    src_size = max(int(bbox_extent * (1.0 + 2.0 * padding)), min_crop)
    src_size = min(src_size, tile_w, tile_h)

    left = cx - src_size // 2
    top = cy - src_size // 2

    # Shift the window so it stays fully inside the tile when possible.
    left = max(0, min(left, tile_w - src_size))
    top = max(0, min(top, tile_h - src_size))
    right = left + src_size
    bottom = top + src_size

    canvas = np.zeros((src_size, src_size, 3), dtype=np.uint8)
    mask = np.zeros((src_size, src_size), dtype=np.uint8)

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
