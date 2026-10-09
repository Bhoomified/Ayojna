"""Human decisions on recommendations, shared by the API (writes) and the executor (reads).

data/state/approvals.json = {group_key: {"decision": "approved" | "rejected", "ts", "note"}}
A group key is volume:from>to (see Move.group_key). A decision stands until it is changed,
so "approve demoting usr_0 hot>cold" keeps applying to the matching moves of later cycles.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ayojna.io import atomic_write_text

DECISIONS = ("approved", "rejected")
FILE = "approvals.json"


def load_approvals(state_dir: str | Path) -> dict[str, dict]:
    p = Path(state_dir) / FILE
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def set_decision(state_dir: str | Path, group: str, decision: str, note: str = "") -> dict:
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    data = load_approvals(state_dir)
    data[group] = {"decision": decision, "ts": time.time(), "note": note[:200]}
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    atomic_write_text(Path(state_dir) / FILE, json.dumps(data))
    return data[group]


def decision_for(approvals: dict[str, dict], group: str) -> str:
    """approved / rejected / pending."""
    return approvals.get(group, {}).get("decision", "pending")