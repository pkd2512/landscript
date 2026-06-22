#!/usr/bin/env python3
"""Regenerate candidate PNGs from cached source tiles.

The pipeline (``run_pipeline.py``) downloads imagery to ``data/source/`` and
slices it into ``data/tiles/<region_id>/``. The candidate PNGs in
``data/glyphs/<out>/`` are derived from those tiles. Some earlier runs drew
a yellow outline / score overlay on each glyph; this script rewrites every
glyph from its cached source tile so the saved PNG is a clean, raw image
of the area.

It uses ONLY locally cached data — it does not call any STAC API or
download anything. So if you copy ``data/tiles/`` to another machine
alongside ``data/candidates/`` you can regenerate all glyphs without
re-running the full pipeline.

Usage:
    python regen_pngs.py                       # default region: india
    python regen_pngs.py india
    python regen_pngs.py --region india --force
    python regen_pngs.py --region india --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("region", nargs="?", default=None,
                   help="Region/country id (basename of "
                        "data/candidates/<region>.json). Default: india.")
    p.add_argument("--region", dest="region_flag", default=None,
                   help="Same as positional; takes precedence if given.")
    p.add_argument("--force", action="store_true",
                   help="Rewrite every PNG, even if it already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be done without writing anything.")
    p.add_argument("--data-dir", default="data",
                   help="Path to the data root (default: ./data).")
    args = p.parse_args()

    region = args.region_flag or args.region or "india"
    root = Path(args.data_dir).resolve()
    cand_path = root / "candidates" / f"{region}.json"
    tiles_root = root / "tiles"
    out_dir = root / "glyphs" / region

    if not cand_path.exists():
        print(f"ERROR: no candidate file at {cand_path}", file=sys.stderr)
        return 2
    if not tiles_root.exists():
        print(f"ERROR: no tile cache at {tiles_root}\n"
              f"Copy data/tiles/ from your other machine into {root}/ and retry.",
              file=sys.stderr)
        return 2

    with open(cand_path) as f:
        items = json.load(f)

    print(f"Region:       {region}")
    print(f"Candidates:   {len(items)}  ({cand_path})")
    print(f"Source tiles: {tiles_root}")
    print(f"Output:       {out_dir}")
    if args.force:
        print("Mode:         FORCE (rewrite every PNG)")
    elif args.dry_run:
        print("Mode:         DRY-RUN (no files will be written)")
    else:
        print("Mode:         normal (rewrite all PNGs)")
    print()

    # We always rewrite by default — the whole point is to flatten any
    # stale overlay-baked PNGs. --force is kept for symmetry but isn't
    # actually different in this version.
    missing_src: list[str] = []
    unreadable: list[str] = []
    written = 0
    skipped = 0

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for c in tqdm(items, unit="png"):
        src_tile_name = c.get("source_tile")
        region_id = c.get("region")
        cid = c.get("id")
        if not (src_tile_name and region_id and cid):
            unreadable.append(repr(c)[:80])
            continue

        src = tiles_root / region_id / src_tile_name
        if not src.exists():
            missing_src.append(str(src))
            continue

        out_path = out_dir / f"{cid}.png"
        if args.dry_run:
            written += 1
            continue

        img = cv2.imread(str(src))
        if img is None:
            unreadable.append(str(src))
            continue

        cv2.imwrite(str(out_path), img)
        written += 1

    print()
    print(f"  wrote        {written:>6} png(s)")
    print(f"  skipped      {skipped:>6}")
    print(f"  missing src  {len(missing_src):>6}")
    print(f"  unreadable   {len(unreadable):>6}")
    if missing_src:
        print("\nMissing source tiles (first 5):")
        for m in missing_src[:5]:
            print(f"  {m}")
        print("  If you're on a machine without the cached tiles, copy "
              "data/tiles/ over from the machine that produced them.")
    if args.dry_run:
        print("\n(dry-run: nothing was written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())