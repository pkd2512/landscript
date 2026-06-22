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
    # Natural
    "farmland",        # bright/dark field patches → L, T, I, F, E, H
    "salt-flat",       # isolated bright closed shapes → O, D, P, B
    "dune",            # parallel ridges → I, N, M, W
    "coastline",       # curved shorelines → C, S, U, J, V
    "delta",           # branching channels → Y, T, K, H
    "river-bend",      # meanders, oxbows → S, U, C
    "lake",            # closed water bodies → O, D
    "mountain-ridge",  # folded ridges → V, A, M, W
    "terraced",        # curved contour terraces — natural calligraphy
    # Human-made (best surfaced at 512px tiles)
    "urban-radial",    # roundabouts, radial street plans → O, P, B
    "solar-farm",      # parallel dark rectangles (barcode)
    "brick-kiln",      # circular chimneys + rectangular ovens
    "fish-farm",       # honeycomb of pond rectangles
    "interchange",     # cloverleaf road junctions → D, O, B
    "mining-pit",      # concentric / spiral terraces
    "wind-farm",       # line/grid of dots (turbine towers)
    "rail-yard",       # parallel stripes fanning in / out
    "container-port",  # sharp colour-block grid of containers
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
    # ====================================================================
    # FARMLAND — bright colour-contrast field blocks (L/T/I/F/E shapes)
    # ====================================================================
    "in-punjab-farmland": Region(
        id="in-punjab-farmland", name="Punjab fields", country="india",
        state="Punjab",
        bbox=BBox(min_lat=30.55, max_lat=30.85, min_lon=75.35, max_lon=75.65),
        terrain_tags=["farmland"],
    ),
    "in-haryana-farmland": Region(
        id="in-haryana-farmland", name="Haryana wheat belt", country="india",
        state="Haryana",
        bbox=BBox(min_lat=29.20, max_lat=29.50, min_lon=76.30, max_lon=76.60),
        terrain_tags=["farmland"],
    ),
    "in-up-canal-grid": Region(
        id="in-up-canal-grid", name="UP canal-irrigation grid",
        country="india", state="Uttar Pradesh",
        bbox=BBox(min_lat=28.40, max_lat=28.70, min_lon=78.40, max_lon=78.70),
        terrain_tags=["farmland"],
    ),
    "in-godavari-farmland": Region(
        id="in-godavari-farmland", name="Godavari delta farmland",
        country="india", state="Andhra Pradesh",
        bbox=BBox(min_lat=16.55, max_lat=16.85, min_lon=81.55, max_lon=81.85),
        terrain_tags=["farmland", "delta"],
    ),

    # ====================================================================
    # SALT-FLAT — bright isolated closed shapes against dark background
    # ====================================================================
    "in-kutch-rann": Region(
        id="in-kutch-rann", name="Rann of Kutch", country="india",
        state="Gujarat",
        bbox=BBox(min_lat=23.7, max_lat=24.0, min_lon=69.85, max_lon=70.15),
        terrain_tags=["salt-flat", "coastline"],
    ),
    "in-sambhar-saltlake": Region(
        id="in-sambhar-saltlake", name="Sambhar Salt Lake",
        country="india", state="Rajasthan",
        bbox=BBox(min_lat=26.85, max_lat=27.05, min_lon=74.95, max_lon=75.20),
        terrain_tags=["salt-flat", "lake"],
    ),

    # ====================================================================
    # DUNE — parallel ridge lines (I/N/M/W)
    # ====================================================================
    "in-thar-dunes": Region(
        id="in-thar-dunes", name="Thar dunes (Jaisalmer)", country="india",
        state="Rajasthan",
        bbox=BBox(min_lat=26.85, max_lat=27.15, min_lon=70.85, max_lon=71.15),
        terrain_tags=["dune"],
    ),

    # ====================================================================
    # COASTLINE — curved shorelines (C/S/U/J/V)
    # ====================================================================
    "in-goa-coast": Region(
        id="in-goa-coast", name="Goa coastline", country="india", state="Goa",
        bbox=BBox(min_lat=15.25, max_lat=15.55, min_lon=73.75, max_lon=74.05),
        terrain_tags=["coastline"],
    ),
    "in-mumbai-coast": Region(
        id="in-mumbai-coast", name="Mumbai harbour & creeks",
        country="india", state="Maharashtra",
        bbox=BBox(min_lat=18.92, max_lat=19.22, min_lon=72.80, max_lon=73.05),
        terrain_tags=["coastline", "urban-radial"],
    ),
    "in-konkan-coast": Region(
        id="in-konkan-coast", name="Konkan scalloped coast",
        country="india", state="Maharashtra",
        bbox=BBox(min_lat=17.00, max_lat=17.30, min_lon=73.10, max_lon=73.40),
        terrain_tags=["coastline"],
    ),
    "in-gulf-of-mannar": Region(
        id="in-gulf-of-mannar", name="Gulf of Mannar islands",
        country="india", state="Tamil Nadu",
        bbox=BBox(min_lat=9.05, max_lat=9.30, min_lon=79.05, max_lon=79.35),
        terrain_tags=["coastline"],
    ),

    # ====================================================================
    # DELTA — branching channels (Y/T/K/H)
    # ====================================================================
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
    "in-krishna-delta": Region(
        id="in-krishna-delta", name="Krishna delta", country="india",
        state="Andhra Pradesh",
        bbox=BBox(min_lat=15.85, max_lat=16.15, min_lon=80.85, max_lon=81.15),
        terrain_tags=["delta", "coastline"],
    ),
    "in-mahanadi-delta": Region(
        id="in-mahanadi-delta", name="Mahanadi delta", country="india",
        state="Odisha",
        bbox=BBox(min_lat=20.20, max_lat=20.50, min_lon=86.55, max_lon=86.85),
        terrain_tags=["delta"],
    ),

    # ====================================================================
    # LAKE — closed water bodies (O/D)
    # ====================================================================
    "in-chilika": Region(
        id="in-chilika", name="Chilika lagoon", country="india",
        state="Odisha",
        bbox=BBox(min_lat=19.65, max_lat=19.95, min_lon=85.25, max_lon=85.55),
        terrain_tags=["lake", "coastline"],
    ),
    "in-deccan-lakes-vidarbha": Region(
        id="in-deccan-lakes-vidarbha",
        name="Vidarbha lakes (incl. Lonar crater)",
        country="india", state="Maharashtra",
        bbox=BBox(min_lat=19.85, max_lat=20.15, min_lon=76.35, max_lon=76.65),
        terrain_tags=["lake"],
    ),
    "in-loktak-lake": Region(
        id="in-loktak-lake", name="Loktak Lake (floating phumdis)",
        country="india", state="Manipur",
        bbox=BBox(min_lat=24.45, max_lat=24.65, min_lon=93.70, max_lon=93.95),
        terrain_tags=["lake"],
    ),

    # ====================================================================
    # RIVER-BEND — meanders, oxbows (S/U/C)
    # ====================================================================
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
    "in-chambal-ravines": Region(
        id="in-chambal-ravines", name="Chambal ravines & meanders",
        country="india", state="Madhya Pradesh",
        bbox=BBox(min_lat=26.40, max_lat=26.70, min_lon=78.20, max_lon=78.50),
        terrain_tags=["river-bend"],
    ),

    # ====================================================================
    # MOUNTAIN-RIDGE — folded ridges (V/A/M/W)
    # ====================================================================
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
    "in-western-ghats-coorg": Region(
        id="in-western-ghats-coorg", name="Coorg / Kodagu hills",
        country="india", state="Karnataka",
        bbox=BBox(min_lat=12.30, max_lat=12.60, min_lon=75.65, max_lon=75.95),
        terrain_tags=["mountain-ridge"],
    ),
    "in-aravalli-ridges": Region(
        id="in-aravalli-ridges", name="Aravalli ridges (Rajasthan)",
        country="india", state="Rajasthan",
        bbox=BBox(min_lat=24.55, max_lat=24.85, min_lon=73.55, max_lon=73.85),
        terrain_tags=["mountain-ridge"],
    ),

    # ====================================================================
    # TERRACED — curved contour terraces on hillsides
    # ====================================================================
    "in-tehri-terraces": Region(
        id="in-tehri-terraces", name="Tehri terraces (Uttarakhand)",
        country="india", state="Uttarakhand",
        bbox=BBox(min_lat=30.30, max_lat=30.45, min_lon=78.40, max_lon=78.55),
        terrain_tags=["terraced", "mountain-ridge"],
    ),
    "in-munnar-terraces": Region(
        id="in-munnar-terraces", name="Munnar tea terraces",
        country="india", state="Kerala",
        bbox=BBox(min_lat=10.05, max_lat=10.20, min_lon=77.00, max_lon=77.15),
        terrain_tags=["terraced", "mountain-ridge"],
    ),

    # ====================================================================
    # URBAN-RADIAL — roundabouts, radial street plans (O/P/B)
    # ====================================================================
    "in-delhi-lutyens": Region(
        id="in-delhi-lutyens", name="Lutyens' Delhi radial roads",
        country="india", state="Delhi",
        bbox=BBox(min_lat=28.58, max_lat=28.66, min_lon=77.18, max_lon=77.26),
        terrain_tags=["urban-radial"],
    ),
    "in-chandigarh-grid": Region(
        id="in-chandigarh-grid", name="Chandigarh sector grid",
        country="india", state="Punjab",
        bbox=BBox(min_lat=30.68, max_lat=30.80, min_lon=76.72, max_lon=76.84),
        terrain_tags=["urban-radial", "farmland"],
    ),

    # ====================================================================
    # SOLAR-FARM — long parallel dark rectangles
    # ====================================================================
    "in-bhadla-solar": Region(
        id="in-bhadla-solar", name="Bhadla Solar Park", country="india",
        state="Rajasthan",
        bbox=BBox(min_lat=27.50, max_lat=27.65, min_lon=71.85, max_lon=72.00),
        terrain_tags=["solar-farm"],
    ),
    "in-pavagada-solar": Region(
        id="in-pavagada-solar", name="Pavagada Solar Park", country="india",
        state="Karnataka",
        bbox=BBox(min_lat=14.05, max_lat=14.20, min_lon=77.20, max_lon=77.35),
        terrain_tags=["solar-farm", "farmland"],
    ),

    # ====================================================================
    # BRICK-KILN — circular chimneys + rectangular ovens
    # ====================================================================
    "in-lucknow-kilns": Region(
        id="in-lucknow-kilns", name="Lucknow outskirts brick kilns",
        country="india", state="Uttar Pradesh",
        bbox=BBox(min_lat=26.75, max_lat=26.90, min_lon=80.80, max_lon=80.95),
        terrain_tags=["brick-kiln", "farmland"],
    ),
    "in-noida-kilns": Region(
        id="in-noida-kilns", name="Greater Noida brick kilns",
        country="india", state="Uttar Pradesh",
        bbox=BBox(min_lat=28.40, max_lat=28.55, min_lon=77.55, max_lon=77.70),
        terrain_tags=["brick-kiln", "urban-radial"],
    ),

    # ====================================================================
    # SALT-PAN (industrial) — distinct from natural salt-flats
    # ====================================================================
    "in-tuticorin-saltpans": Region(
        id="in-tuticorin-saltpans", name="Tuticorin salt pans",
        country="india", state="Tamil Nadu",
        bbox=BBox(min_lat=8.70, max_lat=8.85, min_lon=78.05, max_lon=78.20),
        terrain_tags=["salt-flat", "coastline"],
    ),
    "in-bhavnagar-saltpans": Region(
        id="in-bhavnagar-saltpans", name="Bhavnagar coastal salt pans",
        country="india", state="Gujarat",
        bbox=BBox(min_lat=21.65, max_lat=21.80, min_lon=72.10, max_lon=72.25),
        terrain_tags=["salt-flat", "coastline"],
    ),

    # ====================================================================
    # FISH-FARM — honeycomb of pond rectangles
    # ====================================================================
    "in-andhra-fishponds": Region(
        id="in-andhra-fishponds", name="Coastal AP shrimp/fish ponds",
        country="india", state="Andhra Pradesh",
        bbox=BBox(min_lat=16.20, max_lat=16.35, min_lon=81.20, max_lon=81.35),
        terrain_tags=["fish-farm", "delta"],
    ),

    # ====================================================================
    # INTERCHANGE — cloverleaf road junctions
    # ====================================================================
    "in-mumbai-pune-interchange": Region(
        id="in-mumbai-pune-interchange",
        name="Mumbai-Pune expressway interchanges",
        country="india", state="Maharashtra",
        bbox=BBox(min_lat=18.85, max_lat=19.00, min_lon=73.30, max_lon=73.45),
        terrain_tags=["interchange", "urban-radial"],
    ),

    # ====================================================================
    # MINING-PIT — concentric stepped craters / spirals
    # ====================================================================
    "in-jharia-mines": Region(
        id="in-jharia-mines", name="Jharia coalfield",
        country="india", state="Jharkhand",
        bbox=BBox(min_lat=23.70, max_lat=23.85, min_lon=86.30, max_lon=86.45),
        terrain_tags=["mining-pit"],
    ),
    "in-bailadila-mines": Region(
        id="in-bailadila-mines", name="Bailadila iron mines",
        country="india", state="Chhattisgarh",
        bbox=BBox(min_lat=18.65, max_lat=18.80, min_lon=81.20, max_lon=81.35),
        terrain_tags=["mining-pit", "mountain-ridge"],
    ),

    # ====================================================================
    # WIND-FARM — line/grid of small dots (turbine bases)
    # ====================================================================
    "in-muppandal-wind": Region(
        id="in-muppandal-wind", name="Muppandal wind farm",
        country="india", state="Tamil Nadu",
        bbox=BBox(min_lat=8.20, max_lat=8.35, min_lon=77.50, max_lon=77.65),
        terrain_tags=["wind-farm"],
    ),

    # ====================================================================
    # RAIL-YARD — parallel stripes fanning in / out
    # ====================================================================
    "in-mughalsarai-railyard": Region(
        id="in-mughalsarai-railyard", name="Mughalsarai rail yard",
        country="india", state="Uttar Pradesh",
        bbox=BBox(min_lat=25.25, max_lat=25.35, min_lon=83.05, max_lon=83.20),
        terrain_tags=["rail-yard", "urban-radial"],
    ),

    # ====================================================================
    # CONTAINER-PORT — sharp colour-block grids
    # ====================================================================
    "in-jnpt-port": Region(
        id="in-jnpt-port", name="JNPT (Nhava Sheva) container port",
        country="india", state="Maharashtra",
        bbox=BBox(min_lat=18.92, max_lat=19.02, min_lon=72.92, max_lon=73.02),
        terrain_tags=["container-port", "coastline"],
    ),
    "in-mundra-port": Region(
        id="in-mundra-port", name="Mundra container port",
        country="india", state="Gujarat",
        bbox=BBox(min_lat=22.70, max_lat=22.80, min_lon=69.65, max_lon=69.75),
        terrain_tags=["container-port", "coastline"],
    ),

    # ====================================================================
    # RIVER-BED — beds, meanders, sandbars along major Indian rivers
    # Big sweep along each river so we capture the variety of contours
    # ====================================================================
    # Ganga — already have Bihar bend; add upstream + downstream + branches
    "in-ganga-allahabad": Region(
        id="in-ganga-allahabad", name="Ganga–Yamuna confluence (Prayagraj)",
        country="india", state="Uttar Pradesh",
        bbox=BBox(min_lat=25.35, max_lat=25.55, min_lon=81.75, max_lon=82.00),
        terrain_tags=["river-bend"],
    ),
    "in-ganga-varanasi": Region(
        id="in-ganga-varanasi", name="Ganga at Varanasi",
        country="india", state="Uttar Pradesh",
        bbox=BBox(min_lat=25.20, max_lat=25.40, min_lon=82.95, max_lon=83.20),
        terrain_tags=["river-bend"],
    ),
    "in-ganga-farakka": Region(
        id="in-ganga-farakka", name="Ganga at Farakka barrage",
        country="india", state="West Bengal",
        bbox=BBox(min_lat=24.75, max_lat=24.95, min_lon=87.85, max_lon=88.10),
        terrain_tags=["river-bend"],
    ),
    "in-hooghly-bend": Region(
        id="in-hooghly-bend", name="Hooghly meanders (West Bengal)",
        country="india", state="West Bengal",
        bbox=BBox(min_lat=23.05, max_lat=23.25, min_lon=88.30, max_lon=88.50),
        terrain_tags=["river-bend"],
    ),

    # Brahmaputra — already have Assam braided; add the Majuli stretch + Tezpur
    "in-brahmaputra-majuli": Region(
        id="in-brahmaputra-majuli",
        name="Brahmaputra at Majuli island",
        country="india", state="Assam",
        bbox=BBox(min_lat=26.85, max_lat=27.05, min_lon=94.05, max_lon=94.35),
        terrain_tags=["river-bend", "delta"],
    ),
    "in-brahmaputra-tezpur": Region(
        id="in-brahmaputra-tezpur", name="Brahmaputra at Tezpur",
        country="india", state="Assam",
        bbox=BBox(min_lat=26.55, max_lat=26.75, min_lon=92.65, max_lon=92.90),
        terrain_tags=["river-bend"],
    ),

    # Indus / its tributaries in Punjab–Ladakh
    "in-indus-leh": Region(
        id="in-indus-leh", name="Indus near Leh",
        country="india", state="Ladakh",
        bbox=BBox(min_lat=34.05, max_lat=34.25, min_lon=77.45, max_lon=77.70),
        terrain_tags=["river-bend", "mountain-ridge"],
    ),
    "in-beas-himachal": Region(
        id="in-beas-himachal", name="Beas meanders (Kullu valley)",
        country="india", state="Himachal Pradesh",
        bbox=BBox(min_lat=31.85, max_lat=32.05, min_lon=77.05, max_lon=77.25),
        terrain_tags=["river-bend", "mountain-ridge"],
    ),

    # Narmada (Western India)
    "in-narmada-jabalpur": Region(
        id="in-narmada-jabalpur",
        name="Narmada at Jabalpur (marble rocks)",
        country="india", state="Madhya Pradesh",
        bbox=BBox(min_lat=23.05, max_lat=23.25, min_lon=79.85, max_lon=80.10),
        terrain_tags=["river-bend"],
    ),
    "in-narmada-bharuch": Region(
        id="in-narmada-bharuch",
        name="Narmada estuary (Bharuch)",
        country="india", state="Gujarat",
        bbox=BBox(min_lat=21.65, max_lat=21.85, min_lon=72.85, max_lon=73.10),
        terrain_tags=["river-bend", "coastline"],
    ),

    # Mahanadi upstream
    "in-mahanadi-hirakud": Region(
        id="in-mahanadi-hirakud",
        name="Hirakud reservoir & Mahanadi",
        country="india", state="Odisha",
        bbox=BBox(min_lat=21.55, max_lat=21.80, min_lon=83.75, max_lon=84.05),
        terrain_tags=["river-bend", "lake"],
    ),

    # Godavari upstream (we already have delta farmland)
    "in-godavari-bend": Region(
        id="in-godavari-bend",
        name="Godavari at Polavaram",
        country="india", state="Andhra Pradesh",
        bbox=BBox(min_lat=17.20, max_lat=17.40, min_lon=81.60, max_lon=81.85),
        terrain_tags=["river-bend"],
    ),

    # Krishna upstream
    "in-krishna-nagarjuna": Region(
        id="in-krishna-nagarjuna",
        name="Nagarjuna Sagar (Krishna reservoir)",
        country="india", state="Telangana",
        bbox=BBox(min_lat=16.55, max_lat=16.75, min_lon=79.25, max_lon=79.50),
        terrain_tags=["river-bend", "lake"],
    ),

    # Kaveri (Cauvery)
    "in-kaveri-mettur": Region(
        id="in-kaveri-mettur",
        name="Mettur dam reservoir (Kaveri)",
        country="india", state="Tamil Nadu",
        bbox=BBox(min_lat=11.75, max_lat=11.95, min_lon=77.75, max_lon=78.00),
        terrain_tags=["river-bend", "lake"],
    ),
    "in-kaveri-srirangam": Region(
        id="in-kaveri-srirangam",
        name="Kaveri at Srirangam (island fork)",
        country="india", state="Tamil Nadu",
        bbox=BBox(min_lat=10.80, max_lat=11.00, min_lon=78.65, max_lon=78.90),
        terrain_tags=["river-bend", "delta"],
    ),

    # Tapi
    "in-tapi-surat": Region(
        id="in-tapi-surat", name="Tapi estuary at Surat",
        country="india", state="Gujarat",
        bbox=BBox(min_lat=21.10, max_lat=21.30, min_lon=72.65, max_lon=72.90),
        terrain_tags=["river-bend", "coastline"],
    ),

    # ====================================================================
    # LAKES — big closed waterbodies, more contour-rich shorelines
    # ====================================================================
    "in-vembanad-lake": Region(
        id="in-vembanad-lake",
        name="Vembanad Lake (Kerala backwaters main body)",
        country="india", state="Kerala",
        bbox=BBox(min_lat=9.55, max_lat=9.85, min_lon=76.30, max_lon=76.55),
        terrain_tags=["lake", "coastline"],
    ),
    "in-pulicat-lake": Region(
        id="in-pulicat-lake", name="Pulicat Lake",
        country="india", state="Andhra Pradesh",
        bbox=BBox(min_lat=13.50, max_lat=13.75, min_lon=80.10, max_lon=80.35),
        terrain_tags=["lake", "coastline"],
    ),
    "in-wular-lake": Region(
        id="in-wular-lake",
        name="Wular Lake (Kashmir)",
        country="india", state="Jammu & Kashmir",
        bbox=BBox(min_lat=34.30, max_lat=34.45, min_lon=74.50, max_lon=74.75),
        terrain_tags=["lake", "mountain-ridge"],
    ),
    "in-dal-lake": Region(
        id="in-dal-lake", name="Dal Lake (Srinagar)",
        country="india", state="Jammu & Kashmir",
        bbox=BBox(min_lat=34.05, max_lat=34.20, min_lon=74.80, max_lon=74.95),
        terrain_tags=["lake", "urban-radial"],
    ),
    "in-bhopal-lakes": Region(
        id="in-bhopal-lakes", name="Bhopal upper & lower lakes",
        country="india", state="Madhya Pradesh",
        bbox=BBox(min_lat=23.20, max_lat=23.35, min_lon=77.30, max_lon=77.50),
        terrain_tags=["lake", "urban-radial"],
    ),
    "in-tso-moriri": Region(
        id="in-tso-moriri", name="Tso Moriri (Ladakh)",
        country="india", state="Ladakh",
        bbox=BBox(min_lat=32.85, max_lat=33.05, min_lon=78.20, max_lon=78.40),
        terrain_tags=["lake", "mountain-ridge"],
    ),
    "in-tso-kar": Region(
        id="in-tso-kar", name="Tso Kar basin",
        country="india", state="Ladakh",
        bbox=BBox(min_lat=33.20, max_lat=33.40, min_lon=77.95, max_lon=78.20),
        terrain_tags=["lake", "salt-flat", "mountain-ridge"],
    ),

    # ====================================================================
    # OXBOWS — abandoned river bends that often look like closed loops
    # ====================================================================
    "in-kosi-oxbows": Region(
        id="in-kosi-oxbows",
        name="Kosi river abandoned channels",
        country="india", state="Bihar",
        bbox=BBox(min_lat=25.95, max_lat=26.15, min_lon=86.95, max_lon=87.20),
        terrain_tags=["river-bend"],
    ),
    "in-yamuna-oxbows": Region(
        id="in-yamuna-oxbows",
        name="Yamuna oxbows north of Mathura",
        country="india", state="Uttar Pradesh",
        bbox=BBox(min_lat=27.65, max_lat=27.85, min_lon=77.65, max_lon=77.90),
        terrain_tags=["river-bend"],
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