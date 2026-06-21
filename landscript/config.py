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


COMPOSITES = {
    "true-color": {"rgb_bands": ("red", "green", "blue"), "desc": "Natural colors"},
    "false-color": {"rgb_bands": ("nir", "red", "green"), "desc": "Vegetation in red, water dark"},
    "swir": {"rgb_bands": ("swir16", "nir", "red"), "desc": "Geology, soil moisture"},
    "agriculture": {"rgb_bands": ("nir", "rededge1", "rededge2"), "desc": "Crop health"},
}


STAC_PROVIDERS = {
    "earth-search": {
        "url": "https://earth-search.aws.element84.com/v1",
        "data_auth": None,
    },
    "planetary-computer": {
        "url": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "data_auth": "sas-token",
        "sas_url": "https://planetarycomputer.microsoft.com/api/sas/v1/token",
    },
}

SATELLITES = {
    "sentinel-2": {
        "stac_collection": "sentinel-2-l2a",
        "scale": 10,
        "cloud_field": "eo:cloud_cover",
        "band_aliases": {},
        "stac_provider": "earth-search",
    },
    "landsat-8": {
        "stac_collection": "landsat-c2-l2",
        "scale": 30,
        "cloud_field": "eo:cloud_cover",
        "band_aliases": {"nir": "nir08"},
        "stac_provider": "planetary-computer",
    },
    "landsat-9": {
        "stac_collection": "landsat-c2-l2",
        "scale": 30,
        "cloud_field": "eo:cloud_cover",
        "band_aliases": {"nir": "nir08"},
        "stac_provider": "planetary-computer",
    },
}


@dataclass
class FontConfig:
    url: str = "https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/NotoSans%5Bwdth,wght%5D.ttf"
    local_path: Optional[Path] = None
    family: str = "Noto Sans"
    size: int = 140

    def resolve_path(self) -> Path:
        if self.local_path and self.local_path.exists():
            return self.local_path
        return PROJECT_ROOT / "fonts" / "NotoSans-Regular.ttf"


@dataclass
class PipelineConfig:
    bbox: BBox = field(default_factory=lambda: REGIONS["bangalore"])
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

    tile_size: int = 1024
    threshold_method: str = "otsu"
    min_contour_area: int = 500
    max_contour_area: int = 500000
    epsilon: float = 0.02

    similarity_threshold: float = 0.10
    composite: str = "true-color"

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
            self.font.local_path = self.fonts_dir / "NotoSans-Regular.ttf"

    @property
    def satellite_config(self) -> dict:
        return SATELLITES.get(self.satellite, SATELLITES["sentinel-2"])

    @property
    def stac_collection(self) -> str:
        return self.satellite_config["stac_collection"]

    @property
    def rgb_bands(self) -> Tuple[str, ...]:
        base = COMPOSITES.get(self.composite, COMPOSITES["true-color"])["rgb_bands"]
        alias = self.satellite_config.get("band_aliases", {})
        return tuple(alias.get(b, b) for b in base)

    @property
    def composite_desc(self) -> str:
        return COMPOSITES.get(self.composite, COMPOSITES["true-color"])["desc"]

    @property
    def cloud_field(self) -> str:
        return self.satellite_config["cloud_field"]

    @property
    def stac_url(self) -> str:
        provider = self.satellite_config["stac_provider"]
        return STAC_PROVIDERS[provider]["url"]
