"""Ayojna planner: choose the cheapest allowed tier for every extent.

Expected cost of extent i on tier t over the next H hours:
    storage + expected retrieval + expected SLA misses x penalty
    + (move cost + early-deletion fee, only if t changes)
Then: capacity limits on hot and warm (demote the extents that lose least), and a
migration budget (keep the moves with the biggest benefit).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ayojna.contracts import EXTENT_MB
from ayojna.settings import CONFIG_DIR

GB = EXTENT_MB / 1024


def load_planner_config() -> dict:
    with open(Path(CONFIG_DIR) / "planner.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class TierEconomics:
    price: np.ndarray  # $ per GB-month, per tier
    retrieval: np.ndarray  # $ per GB read, per tier
    min_hours: np.ndarray  # minimum storage time, per tier
    latency_ms: np.ndarray  # base latency, per tier
    capacity: np.ndarray  # max extents, per tier
    move_cost_per_gb: float
    hours_per_month: float


def plan(
    expected_ios: np.ndarray,  # expected I/Os per hour, per extent
    read_gb_per_io: np.ndarray,  # GB read per I/O, per extent
    sla_ms: np.ndarray,  # latency target, per extent
    current: np.ndarray,  # current tier, per extent
    held_hours: np.ndarray,  # hours spent on the current tier
    allowed: np.ndarray,  # (n, 4) from the policy guard
    confident: np.ndarray,  # False = abstain: stay put
    eco: TierEconomics,
    cfg: dict,
) -> np.ndarray:
    n, H = len(current), cfg["horizon_hours"]
    rows = np.arange(n)
    ios_h = (expected_ios * H)[:, None]
    cost = (
        GB * eco.price[None, :] * H / eco.hours_per_month
        + ios_h * read_gb_per_io[:, None] * eco.retrieval[None, :]
        + ios_h * (eco.latency_ms[None, :] > sla_ms[:, None]) * cfg["sla_penalty_per_io"]
    )
    remaining = np.clip(eco.min_hours[current] - held_hours, 0, None)
    leave_fee = GB * eco.price[current] * remaining / eco.hours_per_month
    move = (
        (GB * eco.move_cost_per_gb + leave_fee + cfg["hysteresis_usd"])
        * H
        / cfg["move_amortization_hours"]
    )
    cost = cost + move[:, None]
    cost[rows, current] -= move  # staying costs no move

    ok = allowed.copy()
    stay_ok = ok[rows, current]
    ok[~confident & stay_ok] = False  # abstain: only the current tier...
    ok[rows[~confident & stay_ok], current[~confident & stay_ok]] = True
    cost = np.where(ok, cost, np.inf)
    choice = cost.argmin(axis=1)

    for t in (0, 1):  # hot, then warm: demote the extents that lose the least
        on_t = np.flatnonzero(choice == t)
        excess = len(on_t) - eco.capacity[t]
        if excess <= 0:
            continue
        slower = cost[on_t][:, t + 1 :]
        alt = slower.min(axis=1)
        regret = alt - cost[on_t, t]
        movable = np.isfinite(alt)
        order = on_t[movable][np.argsort(regret[movable])][:excess]
        choice[order] = t + 1 + cost[order][:, t + 1 :].argmin(axis=1)

    moving = np.flatnonzero(choice != current)
    budget = int(cfg["max_gb_moved_per_hour"] / GB)
    if len(moving) > budget:
        benefit = cost[moving, current[moving]] - cost[moving, choice[moving]]
        keep = moving[np.argsort(-benefit)[:budget]]
        revert = np.setdiff1d(moving, keep)
        choice[revert] = current[revert]
    return choice