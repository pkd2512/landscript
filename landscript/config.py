from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class BBox:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.min_lat + self.max_lat) / 2,
                (self.min_lon + self.max_lon) / 2)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat,
                self.max_lon, self.max_lat)


REGIONS = {
    "bangalore": BBox(min_lat=12.82, max_lat=13.12,
                      min_lon=77.42, max_lon=77.82),
    "mumbai": BBox(min_lat=18.87, max_lat=19.27,
                   min_lon=72.77, max_lon=73.02),
    "delhi": BBox(min_lat=28.38, max_lat=28.78,
                  min_lon=77.02, max_lon=77.42),
    "chennai": BBox(min_lat=12.87, max_lat=13.27,
                    min_lon=80.12, max_lon=80.42),
    "kolkata": BBox(min_lat=22.44, max_lat=22.74,
                    min_lon=88.22, max_lon=88.52),
}


SATELLITES = {
    "sentinel-2": {
        "stac_collection": "sentinel-2-l2a",
        "rgb_bands": ("B04", "B03", "B02"),
        "scale": 10,
        "cloud_field": "eo:cloud_cover",
    },
    "landsat-8": {
        "stac_collection": "landsat-8-l2-c2",
        "rgb_bands": ("B04", "B03", "B02"),
        "scale": 30,
        "cloud_field": "eo:cloud_cover",
    },
    "landsat-9": {
        "stac_collection": "landsat-9-l2-c2",
        "rgb_bands": ("B04", "B03", "B02"),
        "scale": 30,
        "cloud_field": "eo:cloud_cover",
    },
}


@dataclass
class FontConfig:
    url: str = "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans-Regular.ttf"
    local_path: Optional[Path] = None
    family: str = "Noto Sans"
    size: int = 140

    def resolve_path(self) -> Path:
        if self.local_path and self.local_path.exists():
            return self.local_path
        return Path(PROJECT_ROOT / "fonts" / self.url.rstrip("/").split("/")[-1])


@dataclass
class PipelineConfig:
    bbox: BBox = REGIONS["bangalore"]
    region_name: str = "bangalore"
    state: str = "Karnataka"
    country: str = "India"
    satellite: str = "sentinel-2"
    font: FontConfig = field(default_factory=FontConfig)

    data_dir: Path = PROJECT_ROOT / "data"
    fonts_dir: Path = field(init=False)
    source_dir: Path = field(init=False)
    tiles_dir: Path = field(init=False)
    glyphs_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)

    cloud_cover_max: int = 20
    date_start: str = "2023-01-01"
    date_end: str = "2024-12-31"

    tile_size: int = 256
    threshold_method: str = "otsu"
    min_contour_area: int = 50
    max_contour_area: int = 50000
    epsilon: float = 0.02

    similarity_threshold: float = 0.15

    def __post_init__(self):
        self.fonts_dir = PROJECT_ROOT / "fonts"
        self.source_dir = self.data_dir / "source" / self.region_name
        self.tiles_dir = self.data_dir / "tiles" / self.region_name
        self.glyphs_dir = self.data_dir / "glyphs"
        self.metadata_dir = self.data_dir / "metadata"
        for d in [self.fonts_dir, self.source_dir, self.tiles_dir,
                  self.glyphs_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

        if self.font.local_path is None:
            fname = self.font.url.rstrip("/").split("/")[-1]
            self.font.local_path = self.fonts_dir / fname

    @property
    def satellite_config(self) -> dict:
        return SATELLITES.get(self.satellite, SATELLITES["sentinel-2"])

    @property
    def stac_collection(self) -> str:
        return self.satellite_config["stac_collection"]

    @property
    def rgb_bands(self) -> Tuple[str, ...]:
        return self.satellite_config["rgb_bands"]

    @property
    def cloud_field(self) -> str:
        return self.satellite_config["cloud_field"]
