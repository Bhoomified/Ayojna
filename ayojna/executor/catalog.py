"""The catalog: which tier every extent is on right now, and since which hour.

It is the single source of truth for placement. Moving data is only "real" once
the catalog is switched (the commit point of the saga). Saved atomically as JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ayojna.executor.tierstore import TIERS, object_key
from ayojna.io import atomic_write_text


def stub_payload(volume: str, extent_id: int) -> bytes:
    """Demo stand-in for a 256 MB extent: small, but unique so checksums mean something."""
    return (f"ayojna extent {volume}/{extent_id}\n" * 32).encode()


class Catalog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: dict[str, dict] = (
            json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps(self.entries))

    def seed(self, volumes, extent_ids, store, hour: int, tier: str = "hot") -> int:
        """Register (and write) extents the catalog has never seen. Returns how many."""
        new = 0
        for v, e in zip(volumes, extent_ids):
            key = object_key(str(v), int(e))
            if key not in self.entries:
                store.put(tier, key, stub_payload(str(v), int(e)))
                self.entries[key] = {"tier": tier, "since": int(hour)}
                new += 1
        if new:
            self.save()
        return new

    def tier_of(self, key: str) -> str | None:
        e = self.entries.get(key)
        return e["tier"] if e else None

    def switch(self, key: str, tier: str, hour: int) -> None:
        before = self.entries.get(key)
        self.entries[key] = {"tier": tier, "since": int(hour)}
        try:
            self.save()
        except Exception:
            self.entries[key] = before  # not saved = not switched
            raise

    def placement(self, volumes, extent_ids) -> tuple[np.ndarray, np.ndarray]:
        """Tier index and 'since' hour per extent, in the order given."""
        rows = [self.entries[object_key(str(v), int(e))] for v, e in zip(volumes, extent_ids)]
        tiers = np.array([TIERS.index(r["tier"]) for r in rows], dtype=int)
        since = np.array([r["since"] for r in rows], dtype=float)
        return tiers, since

    def counts(self) -> dict[str, int]:
        out = {t: 0 for t in TIERS}
        for e in self.entries.values():
            out[e["tier"]] += 1
        return out