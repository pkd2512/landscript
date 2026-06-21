# landscript

Discover typography hidden in satellite imagery.

Landscript is an open-source computational art project that finds shapes resembling letters from human writing systems in satellite imagery of India. Every glyph is sourced from a real place on Earth — no AI generation.

## Project Structure

```
landscript/
├── landscript/          # Python package
│   ├── config.py        # Pipeline configuration (Bangalore defaults)
│   ├── gee.py           # Google Earth Engine integration
│   ├── cv_pipeline.py   # OpenCV thresholding, contours, shape matching
│   └── metadata.py      # JSON-based glyph metadata store
├── notebooks/
│   └── extract_glyphs.ipynb  # Colab-ready pipeline notebook
├── data/                # Local data directory (gitignored)
│   ├── source/          # Raw Sentinel-2 downloads
│   ├── tiles/           # 256×256 tiles for processing
│   ├── glyphs/          # Extracted letter candidate images
│   └── metadata/        # JSON store files
├── pyproject.toml
└── README.md
```

## Getting Started (Colab)

Open the notebook:

```
https://colab.research.google.com/github/pkd2512/landscript/blob/main/notebooks/extract_glyphs.ipynb
```

The pipeline:
1. Authenticate Earth Engine
2. Search Sentinel-2 scenes for Bangalore
3. Download and tile imagery
4. Run contour analysis + shape matching against letter templates
5. Save glyph images + metadata to Google Drive

## Design Principles

- Real satellite imagery only.
- No AI-generated images.
- Fully reproducible pipeline.
- Open source.
- Data provenance for every glyph.

## License

MIT
