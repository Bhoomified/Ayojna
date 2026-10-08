"""Turn extent_hourly into model features plus hot/warm/cold labels.

Each output row = one extent at one hour, describing its past (features) and,
for training, its next 24 hours (label). Hours with no I/O are filled with 0,
because "nothing happened" is exactly what makes data cold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ayojna.contracts import EXTENT_HOURLY_COLUMNS, Tier, validate_frame
from ayojna.settings import CONFIG_DIR

KEYS = ["volume", "extent_id", "hour"]
FEATURE_COLUMNS = [
    "acc_1h",
    "acc_6h",
    "acc_24h",
    "acc_72h",
    "hours_since_access",
    "trend_24_vs_72",
    "same_hour_yesterday",
    "read_ratio_24h",
    "avg_io_size_24h",
    "rand_ratio_24h",
    "hour_of_day",
]
# day_of_week is computed but NOT a model feature: one week of trace means the
# test day is a weekday the model never saw in training.
NEVER = 999  # hours_since_access when the extent was never touched


def load_label_config() -> dict:
    with open(Path(CONFIG_DIR) / "models.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dense_grid(eh: pd.DataFrame) -> pd.DataFrame:
    """Every active extent x every hour, zeros where nothing happened."""
    hours = np.arange(int(eh["hour"].max()) + 1)
    extents = eh[["volume", "extent_id"]].drop_duplicates()
    grid = extents.merge(pd.DataFrame({"hour": hours}), how="cross")
    df = grid.merge(eh, on=KEYS, how="left")
    fill = {c: 0 for c in EXTENT_HOURLY_COLUMNS if c not in KEYS}
    df = df.fillna(fill)
    return df.sort_values(KEYS, kind="stable").reset_index(drop=True)


def _rolling(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    """Sum of `col` over the last `window` hours, including the current hour."""
    g = df.groupby(["volume", "extent_id"], sort=False)[col]
    cs = g.cumsum()
    prev = cs.groupby([df["volume"], df["extent_id"]], sort=False).shift(window)
    return cs - prev.fillna(0)


def build_features(eh: pd.DataFrame, hot_min: int = 50, horizon: int = 24) -> pd.DataFrame:
    eh = validate_frame(eh, EXTENT_HOURLY_COLUMNS, "extent_hourly")
    df = _dense_grid(eh)
    df["acc"] = df["reads"] + df["writes"]
    df["bytes"] = df["read_bytes"] + df["write_bytes"]
    df["rand_weighted"] = df["rand_ratio"] * df["acc"]
    grp = [df["volume"], df["extent_id"]]

    df["acc_1h"] = df["acc"]
    for w in (6, 24, 72):
        df[f"acc_{w}h"] = _rolling(df, "acc", w)
    reads_24 = _rolling(df, "reads", 24)
    bytes_24 = _rolling(df, "bytes", 24)
    rand_24 = _rolling(df, "rand_weighted", 24)
    safe = df["acc_24h"].replace(0, np.nan)
    df["read_ratio_24h"] = (reads_24 / safe).fillna(0.0)
    df["avg_io_size_24h"] = (bytes_24 / safe).fillna(0.0)
    df["rand_ratio_24h"] = (rand_24 / safe).fillna(0.0)

    last = df["hour"].where(df["acc"] > 0).groupby(grp, sort=False).ffill()
    df["hours_since_access"] = (df["hour"] - last).fillna(NEVER).clip(upper=NEVER)
    df["trend_24_vs_72"] = df["acc_24h"] / (df["acc_72h"] / 3.0 + 1.0)
    df["same_hour_yesterday"] = df.groupby(grp, sort=False)["acc"].shift(24).fillna(0)
    df["hour_of_day"] = df["hour"] % 24
    df["day_of_week"] = (df["hour"] // 24) % 7

    # label: accesses in the NEXT `horizon` hours (unknown for the last hours)
    cs = df.groupby(grp, sort=False)["acc"].cumsum()
    future = cs.groupby(grp, sort=False).shift(-horizon) - cs
    df["future_acc"] = future
    df["label"] = np.select(
        [future >= hot_min, future > 0, future == 0],
        [Tier.HOT.value, Tier.WARM.value, Tier.COLD.value],
        default=None,
    )
    return df[KEYS + FEATURE_COLUMNS + ["day_of_week", "future_acc", "label"]]


def time_split(features: pd.DataFrame, test_hours: int, horizon: int = 24):
    """Train on the past, test on the future, with no overlap between label windows."""
    labelled = features[features["label"].notna()]
    last = int(labelled["hour"].max())
    test_start = last - test_hours + 1
    train = labelled[
        labelled["hour"] + horizon <= test_start - 1
    ]  # train labels end before test begins
    test = labelled[labelled["hour"] >= test_start]
    if train.empty or test.empty:
        raise ValueError("not enough hours for a train/test split: replay more days of trace")
    return train, test
