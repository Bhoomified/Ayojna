r"""Executor: applies a MovePlan safely, one extent at a time (a saga per move).

    fence check -> idempotency check -> budget -> copy -> verify -> switch -> clean up
                                                   \______ on error: roll back ______/

- fencing:     a supervisor that lost the lease stops before the next move
- idempotency: each move has a key (run, volume, extent, target); replays are skipped
- verify:      the copy must match the source checksum before the catalog switches
- rollback:    a failed copy/verify deletes the partial copy; the catalog never changed
- budget:      never moves more GB per cycle than the migration budget
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

from ayojna.contracts import Move, MovePlan
from ayojna.executor.catalog import Catalog
from ayojna.executor.tierstore import object_key
from ayojna.planner.optimizer import plan


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Executor:
    def __init__(
        self,
        store,
        catalog: Catalog,
        ledger_path: str | Path,
        budget_gb: float,
        corrupt_first: bool = False,  # demo fault: first copy is corrupted -> rollback
        crash_after_moves: int | None = None,  # demo fault: process dies mid-plan
    ):
        self.store, self.catalog = store, catalog
        self.ledger = Path(ledger_path)
        self.budget_gb = budget_gb
        self.corrupt_first = corrupt_first
        self.crash_after_moves = crash_after_moves

    def _done_keys(self) -> set[str]:
        if not self.ledger.exists():
            return set()
        lines = self.ledger.read_text(encoding="utf-8").splitlines()
        return {json.loads(x)["key"] for x in lines if x.strip()}

    def _record(self, key: str) -> None:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "key": key}) + "\n")

    def execute(
        self,
        plan: MovePlan,
        is_current: Callable[[int], bool],
        decide: Callable[[Move], str] | None = None,
    ) -> dict:
        """decide(move) -> approved / rejected / pending (human-in-the-loop mode); None = all."""
        run_id, token = plan.envelope.run_id, plan.envelope.fencing_token
        done = self._done_keys()
        counts = {"done": 0, "skipped": 0, "rolled_back": 0, "over_budget": 0, "stale": 0}
        counts.update(pending=0, rejected=0)
        results, used_gb, fenced, applied = [], 0.0, False, 0
        for m in plan.moves:
            if not is_current(token):
                fenced = True  # a newer leader exists: stop giving orders
                break
            idem = m.idempotency_key(run_id)
            obj = object_key(m.volume, m.extent_id)
            src, dst = m.from_tier.value, m.to_tier.value
            status, note = "done", ""
            verdict = decide(m) if decide else "approved"
            if verdict != "approved":
                status = "rejected" if verdict == "rejected" else "pending"
                note = "rejected by an operator" if status == "rejected" else "awaiting approval"
            elif idem in done or self.catalog.tier_of(obj) == dst:
                status, note = "skipped", "already applied (idempotent replay)"
                if self.store.exists(dst, obj):
                    self.store.delete(src, obj)  # finish a clean-up a crash may have missed
            elif self.catalog.tier_of(obj) != src:
                status, note = "stale", f"catalog says {self.catalog.tier_of(obj)}, plan says {src}"
            elif used_gb + m.size_gb > self.budget_gb + 1e-9:
                status, note = "over_budget", "migration budget reached"
            else:
                try:
                    data = self.store.get(src, obj)
                    copy = data
                    if self.corrupt_first:
                        copy, self.corrupt_first = data[:-1] + b"X", False
                    self.store.put(dst, obj, copy)  # 1. copy
                    if _sha(self.store.get(dst, obj)) != _sha(data):  # 2. verify
                        raise OSError("checksum mismatch after copy")
                    self.catalog.switch(obj, dst, plan.hour)  # 3. switch (commit point)
                    self._record(idem)
                    self.store.delete(src, obj)  # 4. clean up the old copy
                    used_gb += m.size_gb
                    applied += 1
                except Exception as exc:  # roll back: drop the partial copy
                    if self.catalog.tier_of(obj) == src and self.store.exists(dst, obj):
                        self.store.delete(dst, obj)
                    status, note = "rolled_back", f"{type(exc).__name__}: {exc}"
            counts[status] += 1
            results.append({"key": idem, "from": src, "to": dst, "status": status, "note": note})
            if self.crash_after_moves is not None and applied >= self.crash_after_moves:
                os._exit(1)  # demo: die mid-plan; the replica resumes and skips finished moves
        return {
            "run_id": run_id,
            "mode": "executed",
            "fenced": fenced,
            "gb_moved": round(used_gb, 3),
            "budget_gb": self.budget_gb,
            **counts,
            "results": results,
        }