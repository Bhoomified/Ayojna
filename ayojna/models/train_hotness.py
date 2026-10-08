"""Train the hotness model, compare it with the rule baseline, promote it if it wins.

Run:
    python -m ayojna.models.train_hotness --features data/lake/features.parquet --store models_store
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ayojna.io import read_table
from ayojna.models.baseline import evaluate, predict_rule
from ayojna.models.features import load_label_config, time_split
from ayojna.models.hotness import train
from ayojna.models.predictor import LATEST


def main(features_path: str, store: str) -> dict:
    cfg = load_label_config()
    hot_min, horizon = cfg["labels"]["hot_min_accesses"], cfg["labels"]["horizon_hours"]
    feats = read_table(features_path)
    tr, te = time_split(feats, cfg["split"]["test_hours"], horizon)
    print(
        f"train {len(tr):,} rows (hours {tr['hour'].min()}-{tr['hour'].max()}) | "
        f"test {len(te):,} rows (hours {te['hour'].min()}-{te['hour'].max()})"
    )

    model = train(tr)
    pred = model.predict(te, with_reasons=False)
    ml = evaluate(te["label"], pred["pred"])
    rule = evaluate(te["label"], predict_rule(te, hot_min)["pred"])
    model.metrics = {"ml": ml, "rule": rule, "abstain_rate": float(pred["abstain"].mean())}

    print(f"\n{'':<14}{'macro-F1':>10}{'hot recall':>12}")
    print(f"{'rule baseline':<14}{rule['macro_f1']:>10.3f}{rule['hot_recall']:>12.3f}")
    print(f"{'ML model':<14}{ml['macro_f1']:>10.3f}{ml['hot_recall']:>12.3f}")
    print(f"abstain rate (confidence < 0.6): {model.metrics['abstain_rate']:.1%}")

    path = model.save(store)
    if ml["macro_f1"] > rule["macro_f1"]:
        (Path(store) / LATEST).write_text(json.dumps({"file": path.name, "version": model.version}))
        print(f"\nPROMOTED {path.name}: beats the rule baseline")
    else:
        print(f"\nNOT promoted {path.name}: does not beat the rule baseline")
        print("the supervisor keeps using the rule fallback")

    first_of_each = pred.drop_duplicates("pred").index  # one hot, one warm, one cold
    print("\nexample explanations:")
    for _, r in model.predict(te.loc[first_of_each]).iterrows():
        print(
            f"  {r['volume']} extent {r['extent_id']} hour {r['hour']}: {r['pred']} "
            f"({r['confidence']:.0%}) because {r['reasons']}"
        )
    return model.metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/lake/features.parquet")
    ap.add_argument("--store", default="models_store")
    a = ap.parse_args()
    main(a.features, a.store)
