import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any


class GlyphStore:
    """Lightweight JSON-file metadata store."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._glyphs: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self._glyphs = json.load(f)
        else:
            self._glyphs = []

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._glyphs, f, indent=2, default=str, ensure_ascii=False)

    def add(self, entry: Dict[str, Any]) -> str:
        entry["id"] = f"glyph_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        entry["created_at"] = datetime.utcnow().isoformat()
        # Tri-state review status: "pending" (default), "accepted", "rejected".
        # Old entries used a bool `accepted`; we keep it derived for back-compat.
        entry.setdefault("status", "pending")
        entry["accepted"] = entry["status"] == "accepted"
        self._glyphs.append(entry)
        self._save()
        return entry["id"]

    def search(self, letter: Optional[str] = None,
               accepted: Optional[bool] = None,
               min_score: Optional[float] = None,
               max_score: Optional[float] = None,
               limit: int = 100) -> List[Dict[str, Any]]:
        results = self._glyphs
        if letter:
            results = [g for g in results if g.get("letter") == letter]
        if accepted is not None:
            results = [g for g in results if g.get("accepted") is accepted]
        if min_score is not None:
            results = [g for g in results if g.get("score", 1) >= min_score]
        if max_score is not None:
            results = [g for g in results if g.get("score", 1) <= max_score]
        return sorted(results, key=lambda g: g.get("score", 1))[:limit]

    def set_status(self, glyph_id: str, status: str) -> bool:
        """Set the review status to one of 'pending', 'accepted', 'rejected'."""
        if status not in ("pending", "accepted", "rejected"):
            return False
        for g in self._glyphs:
            if g["id"] == glyph_id:
                g["status"] = status
                g["accepted"] = status == "accepted"
                self._save()
                return True
        return False

    def accept(self, glyph_id: str) -> bool:
        return self.set_status(glyph_id, "accepted")

    def reject(self, glyph_id: str) -> bool:
        return self.set_status(glyph_id, "rejected")

    def unreview(self, glyph_id: str) -> bool:
        return self.set_status(glyph_id, "pending")

    def get(self, glyph_id: str) -> Optional[Dict[str, Any]]:
        for g in self._glyphs:
            if g["id"] == glyph_id:
                return g
        return None

    def all(self, limit: int = 500) -> List[Dict[str, Any]]:
        return self._glyphs[:limit]

    def delete(self, glyph_id: str) -> bool:
        for i, g in enumerate(self._glyphs):
            if g["id"] == glyph_id:
                self._glyphs.pop(i)
                self._save()
                return True
        return False

    def count(self) -> int:
        return len(self._glyphs)
