"""JSON-backed candidate store for Landscript.

Keeps a flat list of candidate dicts and provides:

  - ``add`` / ``get`` / ``all`` / ``delete`` / ``count``
  - tri-state status ``pending`` / ``accepted`` / ``rejected``
  - **manual letter assignment** (``set_letter``) — the *only* place a
    letter gets attached to a candidate. The pipeline doesn't predict
    letters; humans assign them in the gallery.
  - **similarity search** over the per-candidate descriptor vector (used
    by the gallery's "find similar" feature).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class CandidateStore:
    """Lightweight JSON store of candidate tiles."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self):
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self._items = json.load(f)
        else:
            self._items = []

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._items, f, indent=2, default=str,
                      ensure_ascii=False)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add(self, entry: Dict[str, Any]) -> str:
        entry.setdefault("id", f"cand_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}")
        entry.setdefault("created_at", datetime.utcnow().isoformat())
        entry.setdefault("status", "pending")     # pending / accepted / rejected
        entry.setdefault("letter", None)          # assigned by human via UI
        self._items.append(entry)
        self._save()
        return entry["id"]

    def get(self, cid: str) -> Optional[Dict[str, Any]]:
        for c in self._items:
            if c["id"] == cid:
                return c
        return None

    def all(self, limit: int = 100000) -> List[Dict[str, Any]]:
        return self._items[:limit]

    def count(self) -> int:
        return len(self._items)

    def delete(self, cid: str) -> bool:
        for i, c in enumerate(self._items):
            if c["id"] == cid:
                self._items.pop(i)
                self._save()
                return True
        return False

    # ------------------------------------------------------------------
    # Status + letter assignment
    # ------------------------------------------------------------------
    def set_status(self, cid: str, status: str) -> bool:
        if status not in ("pending", "accepted", "rejected"):
            return False
        c = self.get(cid)
        if c is None:
            return False
        c["status"] = status
        self._save()
        return True

    def set_letter(self, cid: str, letter: Optional[str]) -> bool:
        """Assign an A–Z label to a candidate (or pass None to clear).

        This is the *only* mechanism that attaches a letter; the matcher
        never does.
        """
        c = self.get(cid)
        if c is None:
            return False
        if letter is not None:
            letter = letter.upper().strip() or None
            if letter is not None and (len(letter) != 1 or not letter.isalpha()):
                return False
        c["letter"] = letter
        self._save()
        return True

    # ------------------------------------------------------------------
    # Similarity search (for "find similar" in the gallery)
    # ------------------------------------------------------------------
    def find_similar(self, cid: str, k: int = 20) -> List[Dict[str, Any]]:
        """Return the ``k`` candidates with the smallest descriptor distance
        to the candidate identified by ``cid``. The candidate itself is
        excluded from the result. Returns an empty list if descriptors are
        unavailable.
        """
        target = self.get(cid)
        if target is None or "descriptor" not in target:
            return []
        tv = np.asarray(target["descriptor"], dtype=np.float64)
        scored: List[tuple] = []
        for c in self._items:
            if c["id"] == cid:
                continue
            if "descriptor" not in c:
                continue
            v = np.asarray(c["descriptor"], dtype=np.float64)
            if v.shape != tv.shape:
                continue
            d = float(np.linalg.norm(v - tv))
            scored.append((d, c))
        scored.sort(key=lambda r: r[0])
        return [c for _, c in scored[:k]]