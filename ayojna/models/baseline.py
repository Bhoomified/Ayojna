"""Rule-based hotness classifier.

Two jobs: (1) the baseline every ML model must beat, (2) the fallback the
supervisor uses when the ML model fails. It must never crash.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from ayojna.contracts import Tier

LABELS = [Tier.HOT.value, Tier.WARM.value, Tier.COLD.value]


def predict_rule(features: pd.DataFrame, hot_min: int = 50) -> pd.DataFrame:
    """Recent and busy -> hot; touched in the last 3 days -> warm; else cold."""
    busy_recent = (features["acc_24h"] >= hot_min) & (features["hours_since_access"] <= 24)
    seen = features["acc_72h"] > 0
    label = np.select([busy_recent, seen], [Tier.HOT.value, Tier.WARM.value], default=Tier.COLD.value)
    probs = {
        Tier.HOT.value: (0.80, 0.15, 0.05),
        Tier.WARM.value: (0.15, 0.70, 0.15),
        Tier.COLD.value: (0.05, 0.15, 0.80),
    }
    p = np.array([probs[x] for x in label]) if len(label) else np.zeros((0, 3))
    return pd.DataFrame({"p_hot": p[:, 0], "p_warm": p[:, 1], "p_cold": p[:, 2], "pred": label},
                        index=features.index)


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "hot_recall": report[Tier.HOT.value]["recall"],
        "support": {k: int(report[k]["support"]) for k in LABELS},
    }
