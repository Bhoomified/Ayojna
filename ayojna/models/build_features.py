"""Build features + labels and score the rule baseline.

Run:  python -m ayojna.models.build_features --inp data/lake/extent_hourly.parquet --out data/lake/features.parquet
"""
from __future__ import annotations

import argparse

from ayojna.io import read_table, write_table
from ayojna.models.baseline import evaluate, predict_rule
from ayojna.models.features import build_features, load_label_config, time_split


def main(inp: str, out: str) -> dict:
    cfg = load_label_config()
    hot_min, horizon = cfg["labels"]["hot_min_accesses"], cfg["labels"]["horizon_hours"]
    feats = build_features(read_table(inp), hot_min=hot_min, horizon=horizon)
    write_table(feats, out)
    print(f"features: {len(feats):,} rows -> {out}")
    print("label mix:", feats["label"].value_counts(dropna=False).to_dict())
    train, test = time_split(feats, cfg["split"]["test_hours"], horizon)
    print(f"train rows {len(train):,} | test rows {len(test):,} (test = last {cfg['split']['test_hours']} labelled hours)")
    scores = evaluate(test["label"], predict_rule(test, hot_min)["pred"])
    print(f"rule baseline on test: macro-F1 {scores['macro_f1']:.3f}, hot recall {scores['hot_recall']:.3f}")
    return scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/lake/extent_hourly.parquet")
    ap.add_argument("--out", default="data/lake/features.parquet")
    a = ap.parse_args()
    main(a.inp, a.out)
