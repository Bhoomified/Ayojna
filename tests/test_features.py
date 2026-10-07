import pandas as pd
import pytest

from ayojna.models.baseline import evaluate, predict_rule
from ayojna.models.features import FEATURE_COLUMNS, NEVER, build_features, time_split


def _eh(rows):
    base = {"read_bytes": 0, "write_bytes": 0, "avg_io_size": 4096.0, "rand_ratio": 1.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_dense_grid_rolling_and_recency():
    eh = _eh([
        {"volume": "v_0", "extent_id": 0, "hour": 0, "reads": 10, "writes": 0},
        {"volume": "v_0", "extent_id": 0, "hour": 5, "reads": 0, "writes": 4},
        {"volume": "v_0", "extent_id": 1, "hour": 7, "reads": 1, "writes": 0},
    ])
    f = build_features(eh, hot_min=5, horizon=2).set_index(["extent_id", "hour"])
    assert set(FEATURE_COLUMNS) <= set(f.columns)
    assert f.loc[(0, 5), "acc_6h"] == 14          # hours 0..5
    assert f.loc[(0, 7), "hours_since_access"] == 2
    assert f.loc[(1, 3), "hours_since_access"] == NEVER
    assert f.loc[(0, 3), "label"] == "warm"       # next 2 h: hours 4-5 -> 4 I/Os
    assert f.loc[(0, 5), "label"] == "cold"       # hours 6-7: nothing
    assert pd.isna(f.loc[(0, 6), "label"])        # window runs past the last hour (7)


def test_time_split_has_no_overlap():
    rows = [{"volume": "v_0", "extent_id": 0, "hour": h, "reads": h % 3, "writes": 0} for h in range(120)]
    f = build_features(_eh(rows), hot_min=5, horizon=24)
    train, test = time_split(f, test_hours=24, horizon=24)
    assert train["hour"].max() + 24 < test["hour"].min()


def test_rule_baseline_probabilities_and_scores():
    rows = [{"volume": "v_0", "extent_id": e, "hour": h, "reads": 30 if e == 0 else (1 if e == 1 and h % 12 == 0 else 0), "writes": 0}
            for e in range(3) for h in range(100)]
    f = build_features(_eh(rows), hot_min=50, horizon=24)
    test = f[f["label"].notna()]
    pred = predict_rule(test)
    assert ((pred[["p_hot", "p_warm", "p_cold"]].sum(axis=1) - 1).abs() < 1e-9).all()
    scores = evaluate(test["label"], pred["pred"])
    assert scores["macro_f1"] > 0.8
