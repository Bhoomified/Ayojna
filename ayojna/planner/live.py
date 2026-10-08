"""Live planning for the supervisor: latest predictions + current placement -> MovePlan.

Same economics as the twin strategy (planner/strategy.py), but for ONE hour and with
the real placement from the catalog. Also returns a short "why" per move for the dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ayojna.contracts import TIER_ORDER, Envelope, Move, MovePlan
from ayojna.planner.optimizer import GB, TierEconomics, load_planner_config, plan
from ayojna.policy.guard import allowed_tiers
from ayojna.twin.sim import Twin


def expected_load(twin: Twin, preds: pd.DataFrame, feats: pd.DataFrame, abstain_below: float):
    """Per extent (twin order): expected I/Os per hour, GB read per I/O, confident?"""
    # learned from labelled history: average I/Os per hour of a hot / warm extent
    rate = (feats.dropna(subset=["label"]).groupby("label")["future_acc"].mean() / 24).to_dict()
    cols = ["volume", "extent_id", "hour"]
    df = preds.merge(feats[cols + ["avg_io_size_24h", "read_ratio_24h"]], on=cols)
    col = pd.DataFrame(
        {"volume": twin.volumes, "extent_id": twin.extent_ids, "col": np.arange(twin.n_extents)}
    )
    df = df.merge(col, on=["volume", "extent_id"])
    exp = np.zeros(twin.n_extents)
    rgb = np.zeros(twin.n_extents)
    conf = np.ones(twin.n_extents, dtype=bool)  # never-touched extents: surely idle
    c = df["col"].to_numpy()
    exp[c] = df["p_hot"] * rate.get("hot", 0) + df["p_warm"] * rate.get("warm", 0)
    rgb[c] = df["avg_io_size_24h"] * df["read_ratio_24h"] / 1e9
    conf[c] = df["confidence"] >= abstain_below
    why = dict(zip(zip(df["volume"], df["extent_id"]), df["reasons"].astype(str)))
    return exp, rgb, conf, why


def live_plan(
    twin: Twin,
    preds: pd.DataFrame,
    feats: pd.DataFrame,
    current: np.ndarray,
    since: np.ndarray,
    envelope: Envelope,
) -> tuple[MovePlan, dict[str, str]]:
    cfg = load_planner_config()
    hour = int(preds["hour"].max())
    exp, rgb, conf, why = expected_load(twin, preds, feats, cfg["abstain_below"])
    allowed, policy = allowed_tiers(twin.volumes, current)
    eco = TierEconomics(
        twin.price,
        twin.retrieval,
        twin.min_hours,
        twin.base_latency,
        twin.capacity_extents,
        twin.move_cost_per_gb,
        twin.hours_per_month,
    )
    choice = plan(exp, rgb, twin.sla_target_ms, current, hour - since, allowed, conf, eco, cfg)
    moves, notes = [], {}
    for i in np.flatnonzero(choice != current):
        f, t = TIER_ORDER[current[i]], TIER_ORDER[choice[i]]
        m = Move(
            volume=str(twin.volumes[i]),
            extent_id=int(twin.extent_ids[i]),
            from_tier=f,
            to_tier=t,
            size_gb=GB,
            expected_saving_per_month=round(
                GB * (twin.price[current[i]] - twin.price[choice[i]]), 6
            ),
            risk="med" if choice[i] >= 2 else "low",
        )
        moves.append(m)
        key = (m.volume, m.extent_id)
        notes[m.idempotency_key(envelope.run_id)] = "; ".join(
            x for x in (why.get(key, "never accessed in this trace"), policy[i]) if x
        )
    moves.sort(key=lambda m: -m.expected_saving_per_month)
    return MovePlan(envelope=envelope, strategy="ayojna", hour=hour, moves=moves), notes