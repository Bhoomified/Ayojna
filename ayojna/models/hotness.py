"""ML hotness model: predicts hot / warm / cold for the next 24 hours.

Model: scikit-learn HistGradientBoostingClassifier, gradient-boosted trees
built the same way as LightGBM (histogram-based splits), with balanced class
weights. Reasons: for each prediction, which features pushed it there, found by
replacing one feature at a time with its typical value and measuring how much
the predicted probability drops.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ayojna.contracts import Tier
from ayojna.models.features import FEATURE_COLUMNS, KEYS

CLASSES = [Tier.HOT.value, Tier.WARM.value, Tier.COLD.value]
ABSTAIN_BELOW = 0.6  # below this confidence the planner must not move the extent

READABLE = {
    "acc_1h": "I/Os in the last hour",
    "acc_6h": "I/Os in the last 6 h",
    "acc_24h": "I/Os in the last 24 h",
    "acc_72h": "I/Os in the last 72 h",
    "hours_since_access": "hours since last access",
    "trend_24_vs_72": "activity trend (24 h vs 72 h)",
    "same_hour_yesterday": "I/Os at this hour yesterday",
    "read_ratio_24h": "read share (24 h)",
    "avg_io_size_24h": "average I/O size (24 h)",
    "rand_ratio_24h": "random-access share (24 h)",
    "hour_of_day": "hour of day",
}


@dataclass
class HotnessModel:
    estimator: HistGradientBoostingClassifier
    features: list[str]
    typical: dict  # median of each feature in training data (used for reasons)
    spread: dict  # spread (IQR) of each feature in training data (used for reasons)
    version: str
    metrics: dict = field(default_factory=dict)

    # ---------- prediction ----------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilities in CLASSES order: hot, warm, cold."""
        x = df[self.features].to_numpy(dtype=float)
        raw = self.estimator.predict_proba(x)
        order = [list(self.estimator.classes_).index(c) for c in CLASSES]
        return raw[:, order]

    def predict(self, df: pd.DataFrame, with_reasons: bool = True) -> pd.DataFrame:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"features missing: {missing}")
        p = self.predict_proba(df)
        idx = p.argmax(axis=1)
        out = df[KEYS].copy()
        out["p_hot"], out["p_warm"], out["p_cold"] = p[:, 0], p[:, 1], p[:, 2]
        out["pred"] = np.array(CLASSES)[idx]
        out["confidence"] = p.max(axis=1)
        out["abstain"] = out["confidence"] < ABSTAIN_BELOW
        out["reasons"] = self.reasons(df, p, idx) if with_reasons else ""
        return out

    def reasons(
        self, df: pd.DataFrame, p: np.ndarray, idx: np.ndarray, top_k: int = 3
    ) -> list[str]:
        """Top features that most support each prediction (feature ablation)."""
        x = df[self.features].to_numpy(dtype=float)
        rows = np.arange(len(x))
        base = p[rows, idx]
        drops = np.zeros((len(x), len(self.features)))
        for j, name in enumerate(self.features):
            xj = x.copy()
            xj[:, j] = self.typical[name]
            pj = self.estimator.predict_proba(xj)
            order = [list(self.estimator.classes_).index(c) for c in CLASSES]
            drops[:, j] = base - pj[:, order][rows, idx]
        # how unusual each value is versus training data (used when no single feature dominates)
        typical = np.array([self.typical[f] for f in self.features])
        spread = np.array([self.spread[f] for f in self.features])
        unusual = np.abs(x - typical) / spread
        out = []
        for i in rows:
            best = [j for j in np.argsort(-drops[i])[:top_k] if drops[i, j] > 0.01]
            if not best:  # several features agree, none decisive alone: name the most unusual ones
                best = [j for j in np.argsort(-unusual[i])[:2] if unusual[i, j] > 0]
            out.append(
                "; ".join(
                    f"{READABLE.get(self.features[j], self.features[j])} = {x[i, j]:g}"
                    for j in best
                )
            )
        return out

    # ---------- persistence ----------
    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"hotness-{self.version}.joblib"
        joblib.dump(self, path)
        (folder / f"hotness-{self.version}.json").write_text(
            json.dumps(
                {"version": self.version, "features": self.features, "metrics": self.metrics},
                indent=2,
            )
        )
        return path

    @staticmethod
    def load(path: str | Path) -> "HotnessModel":
        model = joblib.load(Path(path))
        if not isinstance(model, HotnessModel):
            raise TypeError(f"{path} is not a HotnessModel")
        return model


def train(train_df: pd.DataFrame, seed: int = 7) -> HotnessModel:
    data = train_df[train_df["label"].notna()]
    if data["label"].nunique() < 2:
        raise ValueError("training data has fewer than 2 classes: replay more trace hours")
    est = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )
    est.fit(data[FEATURE_COLUMNS].to_numpy(dtype=float), data["label"].to_numpy())
    # every class must be present, otherwise CLASSES order cannot be built
    for c in CLASSES:
        if c not in est.classes_:
            raise ValueError(f"class {c!r} missing from training labels")
    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    typical = {c: float(data[c].median()) for c in FEATURE_COLUMNS}
    iqr = data[FEATURE_COLUMNS].quantile(0.75) - data[FEATURE_COLUMNS].quantile(0.25)
    std = data[FEATURE_COLUMNS].std().fillna(0)
    spread = {
        c: float(iqr[c]) if iqr[c] > 0 else (float(std[c]) if std[c] > 0 else 1.0)
        for c in FEATURE_COLUMNS
    }
    return HotnessModel(est, list(FEATURE_COLUMNS), typical, spread, version)
