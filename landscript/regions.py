"""Country / region definitions for Landscript.

A *region* is a small geographic bbox (typically 0.3°×0.3°) with terrain tags
that hint at what letter shapes might appear there. A *country* groups
regions and provides a macro bbox for reference.

To add a new country: append a `Country(...)` to ``COUNTRIES`` and one or
more `Region(...)` entries with that country's id. No other code changes
required — ``python run_pipeline.py --country <id>`` will pick them up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .config import BBox


# Terrain tags — purely documentary; the gallery filters use these.
TERRAIN_TAGS = (
    "farmland",        # bright/dark field patches → L, T, I, F, E, H
    "salt-flat",       # isolated bright closed shapes → O, D, P, B
    "dune",            # parallel ridges → I, N, M, W
    "coastline",       # curved shorelines → C, S, U, J, V
    "delta",           # branching channels → Y, T, K, H
    "river-bend",      # meanders, oxbows → S, U, C
    "lake",            # closed water bodies → O, D
    "mountain-ridge",  # folded ridges → V, A, M, W
    "urban-radial",    # roundabouts, radial street plans → O, P, B
)


@dataclass
class Region:
    """A small bbox that's a promising hunting ground for letter shapes."""
    id: str
    name: str
    country: str
    state: str
    bbox: BBox
    terrain_tags: List[str] = field(default_factory=list)


@dataclass
class Country:
    """A country: macro bbox + the ids of regions that belong to it."""
    id: str
    name: str
    bbox: BBox
    region_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------
# Each region is ~0.3° × 0.3° (~33×33 km at India's latitude), which fits
# inside a single Sentinel-2 scene. Selected for variety of terrain tags
# rather than tourism fame — see the issue tracker for the rationale.

REGIONS: Dict[str, Region] = {
    # --- India ---
    "in-punjab-farmland": Region(
        id="in-punjab-farmland", name="Punjab fields", country="india",
        state="Punjab",
        bbox=BBox(min_lat=30.55, max_lat=30.85, min_lon=75.35, max_lon=75.65),
        terrain_tags=["farmland"],
    ),
    "in-kutch-rann": Region(
        id="in-kutch-rann", name="Rann of Kutch", country="india",
        state="Gujarat",
        bbox=BBox(min_lat=23.7, max_lat=24.0, min_lon=69.85, max_lon=70.15),
        terrain_tags=["salt-flat", "coastline"],
    ),
    "in-thar-dunes": Region(
        id="in-thar-dunes", name="Thar dunes (Jaisalmer)", country="india",
        state="Rajasthan",
        bbox=BBox(min_lat=26.85, max_lat=27.15, min_lon=70.85, max_lon=71.15),
        terrain_tags=["dune"],
    ),
    "in-goa-coast": Region(
        id="in-goa-coast", name="Goa coastline", country="india",
        state="Goa",
        bbox=BBox(min_lat=15.25, max_lat=15.55, min_lon=73.75, max_lon=74.05),
        terrain_tags=["coastline"],
    ),
    "in-kerala-backwaters": Region(
        id="in-kerala-backwaters", name="Kerala backwaters", country="india",
        state="Kerala",
        bbox=BBox(min_lat=9.45, max_lat=9.75, min_lon=76.25, max_lon=76.55),
        terrain_tags=["delta", "coastline"],
    ),
    "in-sundarbans": Region(
        id="in-sundarbans", name="Sundarbans delta", country="india",
        state="West Bengal",
        bbox=BBox(min_lat=21.8, max_lat=22.1, min_lon=88.7, max_lon=89.0),
        terrain_tags=["delta", "coastline"],
    ),
    "in-chilika": Region(
        id="in-chilika", name="Chilika lagoon", country="india",
        state="Odisha",
        bbox=BBox(min_lat=19.65, max_lat=19.95, min_lon=85.25, max_lon=85.55),
        terrain_tags=["lake", "coastline"],
    ),
    "in-ganga-bihar": Region(
        id="in-ganga-bihar", name="Ganga meanders (Bihar)", country="india",
        state="Bihar",
        bbox=BBox(min_lat=25.45, max_lat=25.75, min_lon=85.05, max_lon=85.35),
        terrain_tags=["river-bend"],
    ),
    "in-brahmaputra-assam": Region(
        id="in-brahmaputra-assam", name="Brahmaputra braided river",
        country="india", state="Assam",
        bbox=BBox(min_lat=26.15, max_lat=26.45, min_lon=91.55, max_lon=91.85),
        terrain_tags=["river-bend", "delta"],
    ),
    "in-ladakh-pangong": Region(
        id="in-ladakh-pangong", name="Pangong Tso & ridges",
        country="india", state="Ladakh",
        bbox=BBox(min_lat=33.65, max_lat=33.95, min_lon=78.25, max_lon=78.55),
        terrain_tags=["lake", "mountain-ridge"],
    ),
    "in-himalaya-spiti": Region(
        id="in-himalaya-spiti", name="Spiti valley", country="india",
        state="Himachal Pradesh",
        bbox=BBox(min_lat=32.05, max_lat=32.35, min_lon=78.05, max_lon=78.35),
        terrain_tags=["mountain-ridge", "river-bend"],
    ),
    "in-himalaya-zanskar": Region(
        id="in-himalaya-zanskar", name="Zanskar ridges", country="india",
        state="Ladakh",
        bbox=BBox(min_lat=33.45, max_lat=33.75, min_lon=76.85, max_lon=77.15),
        terrain_tags=["mountain-ridge", "river-bend"],
    ),
    "in-himalaya-kinnaur": Region(
        id="in-himalaya-kinnaur", name="Kinnaur folded ridges",
        country="india", state="Himachal Pradesh",
        bbox=BBox(min_lat=31.55, max_lat=31.85, min_lon=78.35, max_lon=78.65),
        terrain_tags=["mountain-ridge"],
    ),
    "in-western-ghats-nilgiri": Region(
        id="in-western-ghats-nilgiri", name="Nilgiri ridges", country="india",
        state="Tamil Nadu",
        bbox=BBox(min_lat=11.25, max_lat=11.55, min_lon=76.55, max_lon=76.85),
        terrain_tags=["mountain-ridge"],
    ),
    "in-deccan-lakes-vidarbha": Region(
        id="in-deccan-lakes-vidarbha",
        name="Vidarbha lakes (incl. Lonar crater)",
        country="india", state="Maharashtra",
        bbox=BBox(min_lat=19.85, max_lat=20.15, min_lon=76.35, max_lon=76.65),
        terrain_tags=["lake"],
    ),
}


# ---------------------------------------------------------------------------
# Countries
# ---------------------------------------------------------------------------

COUNTRIES: Dict[str, Country] = {
    "india": Country(
        id="india", name="India",
        bbox=BBox(min_lat=6.5, max_lat=36.0, min_lon=68.0, max_lon=98.0),
        region_ids=[r.id for r in REGIONS.values() if r.country == "india"],
    ),
    # To add a new country, append a Country(...) entry and one or more
    # Region(...) entries above. Example:
    # "sri-lanka": Country(
    #     id="sri-lanka", name="Sri Lanka",
    #     bbox=BBox(min_lat=5.9, max_lat=9.85, min_lon=79.5, max_lon=81.9),
    #     region_ids=["lk-jaffna-lagoon", ...],
    # ),
}


def regions_for_country(country_id: str) -> List[Region]:
    """Return all `Region`s registered under the given country id, in
    declaration order."""
    if country_id not in COUNTRIES:
        raise KeyError(
            f"Unknown country '{country_id}'. "
            f"Available: {sorted(COUNTRIES.keys())}"
        )
    ids = COUNTRIES[country_id].region_ids
    return [REGIONS[i] for i in ids if i in REGIONS]