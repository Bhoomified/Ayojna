"""What the supervisor calls to get hotness predictions.

Fallback chain (from the design doc):
  1. latest promoted ML model             -> source "primary", status "ok"
  2. if it is missing, broken or insane   -> rule baseline, source "fallback", status "degraded"
It never raises for a model problem: the run always gets an answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ayojna.contracts import Source, Status
from ayojna.models.baseline import predict_rule
from ayojna.models.features import KEYS
from ayojna.models.hotness import ABSTAIN_BELOW, HotnessModel

LATEST = "hotness-latest.json"  # pointer file written when a model is promoted


def latest_model_path(store: str | Path) -> Path | None:
    pointer = Path(store) / LATEST
    if not pointer.exists():
        return None
    return Path(store) / json.loads(pointer.read_text())["file"]


def _sanity_check(pred: pd.DataFrame) -> None:
    probs = pred[["p_hot", "p_warm", "p_cold"]].to_numpy()
    if not np.isfinite(probs).all():
        raise ValueError("non-finite probabilities")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("probabilities do not sum to 1")
    if len(pred) >= 200 and pred["pred"].nunique() == 1:
        raise ValueError("model predicts a single class for everything")


def predict_hotness(
    features: pd.DataFrame, store: str | Path = "models_store", hot_min: int = 50
) -> tuple[pd.DataFrame, Source, Status, str]:
    """Returns (predictions, source, status, note)."""
    try:
        path = latest_model_path(store)
        if path is None:
            raise FileNotFoundError("no promoted model yet")
        model = HotnessModel.load(path)
        pred = model.predict(features)
        _sanity_check(pred)
        return pred, Source.PRIMARY, Status.OK, f"model {model.version}"
    except Exception as exc:  # any model failure -> rule fallback
        rule = predict_rule(features, hot_min)
        pred = features[KEYS].copy()
        pred[["p_hot", "p_warm", "p_cold", "pred"]] = rule[["p_hot", "p_warm", "p_cold", "pred"]]
        pred["confidence"] = rule[["p_hot", "p_warm", "p_cold"]].max(axis=1)
        pred["abstain"] = pred["confidence"] < ABSTAIN_BELOW
        pred["reasons"] = "rule fallback: recency and activity thresholds"
        return pred, Source.FALLBACK, Status.DEGRADED, f"fallback ({type(exc).__name__}: {exc})"
