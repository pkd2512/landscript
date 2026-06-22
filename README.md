# landscript

Hunt for accidental typography in satellite imagery.

Landscript scans satellite tiles for **visually interesting shapes** —
the kind of things that *might* look like letters from a writing system,
if you squint. **The pipeline never predicts letters.** It scores tiles
purely on visual properties (dominant-shape extraction, edge density,
shape complexity, hole topology, etc.) and ranks them by an
"interest score". Humans browse the gallery and assign letters by hand.

Every shape is a real place on Earth — pulled from the
**Earth Search STAC API** (AWS-hosted Sentinel-2 / Landsat). No API
keys, no registration.

## What it does

1. For each curated region (small bbox with terrain tags — see
   `landscript/regions.py`), download the cleanest 1–N scenes from STAC.
2. Slice each scene into 1024px tiles and reject the obviously useless
   ones: cloud-heavy, low-contrast/featureless, or pHash-duplicate of
   a tile we already have.
3. For each surviving tile, extract its dominant shape via HSV k-means
   clustering, compute features (area fraction, edge density, shape
   complexity, solidity, Hu moments) and combine them into a single
   **interest score**.
4. Save the raw tile PNG to `data/glyphs/<region>/` and its metadata
   (lat/lon, source scene, feature vector, descriptor, …) to
   `data/candidates/<region>.json`.
5. Browse them in the local gallery, mark candidates as
   accepted/rejected, optionally assign a letter A–Z, and use
   "Find similar shapes" to pivot to look-alikes via descriptor
   nearest-neighbour.

## Quick start

The easiest path is the bundled `run.sh` wrapper — it creates the venv,
installs the package, and dispatches to whichever pipeline step you want:

```bash
./run.sh setup                       # one-time: venv + pip install -e .
./run.sh region  in-kutch-rann       # discover for a single region
./run.sh country india               # discover for every region in a country
./run.sh regen   india               # rebuild glyph PNGs from cached tiles
./run.sh gallery india               # serve + open the gallery
./run.sh help                        # see all subcommands
```

The legacy positional form still works:

```bash
./run.sh in-kutch-rann 1 true-color  # same as `./run.sh region in-kutch-rann 1 true-color`
```

Or call the Python entry points directly if you prefer:

```bash
pip install -e .
python run_pipeline.py --region in-kutch-rann
python run_pipeline.py --country india --scenes 1 --top 200
python regen_pngs.py    --region india
python gallery.py       --region india --open
```

## Gallery

The gallery is a single-file static HTML page served by a tiny
stdlib HTTP server. It loads `data/candidates/<region>.json` and renders
the saved PNGs from `data/glyphs/<region>/`.

- **Each card shows a serial number** (`#1`, `#2`, …) based on the
  current sort/filter order, the interest score, and the region id.
- **The detail panel** (click any card) opens a side drawer with the
  full tile, all metadata, and a "📍 Open <lat>, <lon> in Google Maps
  (satellite)" link so you can verify the shape on the ground.
- **The `Limit` field** defaults to `0` (= show all candidates). Set a
  positive number to cap how many cards render at once — useful for
  very large catalogues.
- **Keyboard** in detail view: `A–Z` assign a letter, `Space` toggles
  accept, `X` rejects, `Del` deletes, `Esc` closes.
- **"Find similar shapes"** computes nearest-neighbour over the
  candidate's 11-D descriptor and re-orders the grid.

Decisions persist back to `data/candidates/<region>.json`.

## Pipeline options

```bash
python run_pipeline.py [options]

  --region          single region id (see landscript/regions.py)
  --country         run every region for a country (e.g. india)
  --satellite       sentinel-2 / landsat-8 / landsat-9
  --scenes          best scenes per region (default 1)
  --cloud           max scene-level cloud cover %% (default 20)
  --top             keep top-N candidates per region (default 200)
  --composite       true-color / false-color / swir / agriculture
  --tile-size       tile side in pixels (default 1024)
  --date-start      ISO date (default 2023-01-01)
  --date-end        ISO date (default 2024-12-31)
  --cloud-threshold tile-level cloud fraction cutoff (default 0.35)
  --min-stddev      min grayscale stddev to keep a tile (default 12.0)
  --dedup-hamming   pHash Hamming distance for dup detection (default 5)
```

## Project layout

```
landscript/
├── run.sh                   # multi-subcommand convenience wrapper
├── run_pipeline.py          # discovery pipeline entry point
├── gallery.py               # web UI for browsing + labelling candidates
├── regen_pngs.py            # rewrite glyph PNGs from cached source tiles
├── landscript/
│   ├── config.py            # bounding boxes, satellites, composites
│   ├── regions.py           # curated region catalogue (per-country)
│   ├── stac.py              # STAC search + scene download + tiling
│   ├── tiles.py             # cloud / contrast / pHash filters
│   ├── features.py          # dominant-shape extraction + interest score
│   └── metadata.py          # JSON candidate store + similarity search
├── data/
│   ├── source/              # downloaded GeoTIFFs        (gitignored)
│   ├── tiles/               # PNG tiles + tile_index.json (gitignored)
│   ├── candidates/          # <region|country>.json      (tracked)
│   └── glyphs/<region>/     # candidate PNGs              (tracked)
├── fonts/
└── pyproject.toml
```

## Regenerating glyph PNGs without redownloading

The heavy `data/source/` (GeoTIFFs) and `data/tiles/` (PNG tiles) trees
are gitignored. The lighter `data/candidates/*.json` plus
`data/glyphs/<region>/*.png` are committed.

If you copy or re-create `data/tiles/` on another machine (or move it
to a different path), you can recreate every candidate PNG without
touching the network:

```bash
./run.sh regen india                            # convenience wrapper

# …or call regen_pngs.py directly for more options
python regen_pngs.py india                      # uses ./data/
python regen_pngs.py india --dry-run            # report only, write nothing
python regen_pngs.py --region india --data-dir /elsewhere/data
```

The script reads `data/candidates/<region>.json`, looks up each
candidate's `source_tile` in `data/tiles/<region_id>/`, and overwrites
`data/glyphs/<region>/<id>.png` with the raw tile (no overlay, no
annotations — just the satellite image).

If a source tile is missing, the script prints its path so you know
exactly which scene to re-tile.

## Moving heavy data to personal storage

```bash
mv data/source /path/to/your/storage/
mv data/tiles /path/to/your/storage/
ln -s /path/to/your/storage/source data/source
ln -s /path/to/your/storage/tiles  data/tiles
```

## Principles

- Real satellite imagery only. No AI generation.
- Fully reproducible. Zero auth. Open source.
- Data provenance for every candidate (lat/lon + scene + tile).
- **The pipeline does not classify letters.** It only finds and ranks
  interesting shapes. The human in the gallery decides what's a "T".

## License

MIT
