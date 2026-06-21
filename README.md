# landscript

Discover typography hidden in satellite imagery.

Landscript searches satellite imagery for shapes resembling letters from human writing systems. Every glyph is sourced from a real place on Earth.

Uses the **Earth Search STAC API** (AWS-hosted Sentinel-2 / Landsat) — no API keys, no registration required.

## Quick Start

```bash
pip install -e .
python run_pipeline.py --region bangalore --satellite sentinel-2 --scenes 3
```

This downloads the 3 best (lowest cloud) scenes for Bangalore, tiles them, runs contour shape-matching against letter templates, and saves glyphs to `data/glyphs/`.

## Usage

```bash
python run_pipeline.py --region bangalore [options]

Options:
  --region        bangalore, mumbai, delhi, chennai, kolkata
  --satellite     sentinel-2, landsat-8, landsat-9
  --scenes        number of best scenes to download (default: 3)
  --cloud         max cloud cover %% (default: 20)
  --threshold     shape match threshold, lower = stricter (default: 0.15)
  --date-start    start date (default: 2023-01-01)
  --date-end      end date (default: 2024-12-31)
```

## Project Structure

```
landscript/
├── run_pipeline.py          # One-command pipeline
├── landscript/
│   ├── config.py            # Region/satellite/font configuration
│   ├── stac.py              # STAC API downloader + tiler
│   ├── cv_pipeline.py       # OpenCV thresholding, contours, shape matching
│   └── metadata.py          # JSON-based glyph metadata store
├── notebooks/
│   └── extract_glyphs.ipynb # Step-by-step exploration notebook
├── data/
│   ├── source/              # Downloaded scenes (gitignored — move to personal storage)
│   ├── tiles/               # 256×256 tiles (gitignored)
│   ├── glyphs/              # Extracted glyph crops (tracked)
│   └── metadata/            # JSON stores per region (tracked)
├── fonts/                   # Downloaded fonts (tracked)
└── pyproject.toml
```

## Moving Data to Personal Storage

The heavy files (`data/source/`, `data/tiles/`) are gitignored. Move them elsewhere and symlink:

```bash
mv data/source /path/to/your/storage/
mv data/tiles /path/to/your/storage/
ln -s /path/to/your/storage/source data/source
ln -s /path/to/your/storage/tiles data/tiles
```

## Principles

- Real satellite imagery only. No AI generation.
- Fully reproducible. Zero auth. Open source.
- Data provenance for every glyph.

## License

MIT
