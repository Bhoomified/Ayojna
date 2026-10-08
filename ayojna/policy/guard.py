"""Policy guard: which tiers each extent may live on. Runs before the optimizer,
so the optimizer can never choose an illegal placement (compliant by construction)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ayojna.contracts import TIER_ORDER
from ayojna.settings import CONFIG_DIR, load_config

TIER_NAMES = [t.value for t in TIER_ORDER]


def load_policy() -> dict:
    with open(Path(CONFIG_DIR) / "policy.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def allowed_tiers(volumes: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Returns a (n_extents, 4) bool mask of allowed tiers and one reason string per extent."""
    cfg, pol = load_config(), load_policy()
    mask = np.ones((len(volumes), 4), dtype=bool)
    reasons = [""] * len(volumes)
    for vol in np.unique(volumes):
        rows = np.flatnonzero(volumes == vol)
        tags = cfg.tags_for(str(vol))
        why = []
        floor = TIER_NAMES.index(pol["sla_floor"].get(tags.sla_class, "archive"))
        mask[np.ix_(rows, range(floor + 1, 4))] = False
        if floor < 3:
            why.append(f"{tags.sla_class} SLA: not below {TIER_NAMES[floor]}")
        if tags.data_class in pol["no_archive_classes"]:
            mask[rows, 3] = False
            why.append(f"{tags.data_class}: no archive")
        if tags.legal_hold and pol["legal_hold_freezes"]:
            mask[rows] = False
            mask[rows, current[rows]] = True
            why.append("legal hold: frozen")
        for r in rows:
            reasons[r] = "; ".join(why)
    return mask, reasons