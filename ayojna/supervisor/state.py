"""Shared supervisor state on disk: leader lease (with fencing token), checkpoints, audit log.

File-based so it runs anywhere with no extra service; the same interface moves to
Redis later. Every write is atomic (write a temp file, then rename).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd


class StateStore:
    def __init__(self, root: str | Path = "data/state", lease_ttl_s: float = 10.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = lease_ttl_s

    # ---------- helpers ----------
    def _write(self, name: str, data: dict) -> None:
        tmp = self.root / f".{name}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self.root / name)

    def _read(self, name: str) -> dict | None:
        p = self.root / name
        return json.loads(p.read_text()) if p.exists() else None

    # ---------- leader lease ----------
    def acquire(self, owner: str) -> int | None:
        """Become leader if nobody holds a live lease. Returns the new fencing token, or None."""
        lease = self._read("lease.json")
        now = time.time()
        if lease and lease["owner"] != owner and lease["expires"] > now:
            return None
        token = (lease or {}).get("token", 0) + (0 if lease and lease["owner"] == owner else 1)
        self._write("lease.json", {"owner": owner, "token": token, "expires": now + self.ttl})
        return token

    def renew(self, owner: str, token: int) -> bool:
        lease = self._read("lease.json")
        if not lease or lease["owner"] != owner or lease["token"] != token:
            return False  # someone else took over: stop giving orders
        lease["expires"] = time.time() + self.ttl
        self._write("lease.json", lease)
        return True

    def is_current(self, token: int) -> bool:
        """Fencing check: only the newest leader's commands are accepted."""
        lease = self._read("lease.json")
        return bool(lease) and lease["token"] == token

    # ---------- checkpoints ----------
    def run_dir(self, run_id: str) -> Path:
        d = self.root / "runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_step(self, run_id: str, step: str, output, info: dict) -> None:
        pd.to_pickle(output, self.run_dir(run_id) / f"{step}.pkl")
        done = self._read(f"run-{run_id}.json") or {"steps": {}}
        done["steps"][step] = info
        self._write(f"run-{run_id}.json", done)

    def load_step(self, run_id: str, step: str):
        done = self._read(f"run-{run_id}.json") or {"steps": {}}
        if step not in done["steps"]:
            return None, None
        return pd.read_pickle(self.run_dir(run_id) / f"{step}.pkl"), done["steps"][step]

    def set_active_run(self, run_id: str | None) -> None:
        self._write("active.json", {"run_id": run_id})

    def active_run(self) -> str | None:
        return (self._read("active.json") or {}).get("run_id")

    # ---------- audit ----------
    def audit(self, event: dict) -> None:
        with open(self.root / "audit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **event}) + "\n")