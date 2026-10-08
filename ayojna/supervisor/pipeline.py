"""The hourly decision cycle as supervised steps: load -> features -> hotness -> plan.

The plan step holds (no moves) for now; Step 5 plugs in the policy guard + planner.
"""

from __future__ import annotations

import time
from pathlib import Path

from ayojna.contracts import Envelope, MovePlan, Source, Status
from ayojna.io import read_table
from ayojna.models.features import build_features, load_label_config
from ayojna.models.predictor import predict_hotness
from ayojna.supervisor.runner import Degraded, Step


def _inject(fn, fault: str | None):
    """Fault injection for demos and tests: 'fail' raises, 'slow' sleeps past the timeout."""
    if fault is None:
        return fn

    def broken(ctx):
        if fault == "fail":
            raise RuntimeError("injected fault")
        time.sleep(60)
        return fn(ctx)

    return broken


def build_steps(
    eh_path: str,
    features_path: str,
    model_store: str,
    faults: dict | None = None,
    timeout_s: float = 30.0,
) -> list[Step]:
    faults = faults or {}
    labels = load_label_config()["labels"]

    def load(ctx):
        return read_table(eh_path)

    def features(ctx):
        return build_features(ctx["load"], labels["hot_min_accesses"], labels["horizon_hours"])

    def features_fallback(ctx):  # last good features table on disk
        return read_table(features_path)

    def hotness(ctx):
        f = ctx["features"]
        latest = f[f["hour"] == f["hour"].max()]
        pred, source, status, note = predict_hotness(
            latest, model_store, labels["hot_min_accesses"]
        )
        return Degraded(pred, note) if source == Source.FALLBACK else pred

    def plan(ctx):
        hour = int(ctx["hotness"]["hour"].max())
        env = Envelope(
            run_id=ctx["run_id"],
            data_version=Path(eh_path).name,
            fencing_token=ctx["token"],
            status=Status.OK,
        )
        return MovePlan(envelope=env, strategy="hold", hour=hour, moves=[])

    def wrap(name, fn):
        return _inject(fn, faults.get(name))

    return [
        Step("load", wrap("load", load), timeout_s),
        Step("features", wrap("features", features), timeout_s, fallback=features_fallback),
        Step("hotness", wrap("hotness", hotness), timeout_s),
        Step("plan", wrap("plan", plan), timeout_s),
    ]