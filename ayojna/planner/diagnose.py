"""Where does a strategy miss its SLA? Breaks missed I/Os down by volume, tier and cause.

Run:  python -m ayojna.planner.diagnose --eh data/lake/extent_hourly.parquet \
          --features data/lake/features.parquet --store models_store
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ayojna.contracts import TIER_ORDER
from ayojna.io import read_table
from ayojna.models.features import load_label_config, time_split
from ayojna.models.predictor import predict_hotness
from ayojna.planner.strategy import AyojnaStrategy
from ayojna.twin.sim import Twin


class Recorder:
    """Wraps a strategy and remembers the placement it chose every hour."""

    def __init__(self, inner):
        self.inner, self.name, self.placements = inner, inner.name, []

    def decide(self, view):
        p = np.asarray(self.inner.decide(view), dtype=int)
        self.placements.append(p)
        return p


def diagnose(eh: pd.DataFrame, feats: pd.DataFrame, store: str) -> pd.DataFrame:
    cfg = load_label_config()
    _, test = time_split(feats, cfg["split"]["test_hours"], cfg["labels"]["horizon_hours"])
    start = int(test["hour"].min())
    twin = Twin.from_extent_hourly(eh)
    preds, *_ = predict_hotness(feats, store)
    rec = Recorder(AyojnaStrategy(twin, preds, feats))
    twin.run(rec)
    seen = twin.ios.cumsum(axis=0) > 0  # extent touched at or before hour h
    rows = []
    for h in range(start, twin.n_hours):
        place, ios = rec.placements[h], twin.ios[h]
        load = np.bincount(place, weights=ios, minlength=4)
        util = np.minimum(twin.max_util, load / twin.ios_capacity)
        lat = (twin.base_latency / (1 - util))[place]
        miss = (lat > twin.sla_target_ms) & (ios > 0)
        base_ok = twin.base_latency[place] <= twin.sla_target_ms
        first_touch = ~(seen[h - 1] if h else np.zeros_like(seen[0]))
        for i in np.flatnonzero(miss):
            cause = "queueing" if base_ok[i] else ("first touch" if first_touch[i] else "slow tier")
            rows.append((twin.volumes[i], TIER_ORDER[place[i]].value, cause, ios[i]))
    df = pd.DataFrame(rows, columns=["volume", "tier", "cause", "missed_ios"])
    total = twin.ios[start:].sum()
    out = df.groupby(["volume", "tier", "cause"])["missed_ios"].sum().sort_values(ascending=False)
    print(
        f"scored hours {start}-{twin.n_hours - 1}: {int(total):,} I/Os, "
        f"{int(df['missed_ios'].sum()):,} missed ({100 * df['missed_ios'].sum() / total:.2f}%)\n"
    )
    return out.reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eh", default="data/lake/extent_hourly.parquet")
    ap.add_argument("--features", default="data/lake/features.parquet")
    ap.add_argument("--store", default="models_store")
    a = ap.parse_args()
    r = diagnose(read_table(a.eh), read_table(a.features), a.store)
    print(r.head(15).to_string(index=False) if len(r) else "no SLA misses")