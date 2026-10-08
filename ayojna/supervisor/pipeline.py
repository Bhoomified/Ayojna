"""The hourly decision cycle as supervised steps:

    load -> features -> hotness -> plan (policy guard + optimizer) -> execute (saga)

Each step has a fallback so the cycle degrades instead of breaking:
features -> last good file, hotness -> rule model, plan -> hold (no moves),
execute -> keep the plan as a recommendation only.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ayojna.contracts import Envelope, MovePlan, Source, Status
from ayojna.executor.catalog import Catalog
from ayojna.executor.saga import Executor
from ayojna.executor.tierstore import make_store
from ayojna.io import atomic_write_text, read_table
from ayojna.models.features import build_features, load_label_config
from ayojna.models.predictor import predict_hotness
from ayojna.planner.live import live_plan
from ayojna.planner.optimizer import load_planner_config
from ayojna.settings import CONFIG_DIR
from ayojna.supervisor.runner import Degraded, Step
from ayojna.supervisor.state import StateStore
from ayojna.twin.sim import Twin


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


def _write_json(path: Path, data: dict) -> None:
    atomic_write_text(path, json.dumps(data, default=str))


def policy_version() -> str:
    return hashlib.sha256((Path(CONFIG_DIR) / "policy.yaml").read_bytes()).hexdigest()[:8]


def build_steps(
    eh_path: str,
    features_path: str,
    model_store: str,
    faults: dict | None = None,
    timeout_s: float = 30.0,
    state_dir: str = "data/state",
    tiers_root: str = "data/tiers",
    store_kind: str = "fs",
    recommend_only: bool = False,
    corrupt_first_move: bool = False,
    crash_after_moves: int | None = None,
) -> list[Step]:
    faults = faults or {}
    labels = load_label_config()["labels"]
    state = StateStore(state_dir)
    catalog_path = state.root / "catalog.json"

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
        pred.attrs["model_version"] = note
        return Degraded(pred, note) if source == Source.FALLBACK else pred

    def envelope(ctx, status=Status.OK, source=Source.PRIMARY):
        return Envelope(
            run_id=ctx["run_id"],
            data_version=Path(eh_path).name,
            model_version=ctx["hotness"].attrs.get("model_version", "unknown"),
            policy_version=policy_version(),
            fencing_token=ctx["token"],
            status=status,
            source=source,
        )

    def plan(ctx):
        preds = ctx["hotness"]
        hour = int(preds["hour"].max())
        twin = Twin.from_extent_hourly(ctx["load"])
        catalog = Catalog(catalog_path)
        catalog.seed(twin.volumes, twin.extent_ids, make_store(store_kind, tiers_root), hour)
        current, since = catalog.placement(twin.volumes, twin.extent_ids)
        mp, why = live_plan(twin, preds, ctx["features"], current, since, envelope(ctx))
        _write_json(state.root / "last_plan.json", {"plan": mp.model_dump(mode="json"), "why": why})
        return mp

    def plan_fallback(ctx):  # hold: no moves is always safe
        env = envelope(ctx, Status.DEGRADED, Source.FALLBACK)
        mp = MovePlan(envelope=env, strategy="hold", hour=int(ctx["hotness"]["hour"].max()))
        _write_json(state.root / "last_plan.json", {"plan": mp.model_dump(mode="json"), "why": {}})
        return mp

    def execute(ctx):
        mp: MovePlan = ctx["plan"]
        # a replica resuming this run re-stamps the plan with ITS token (fencing)
        env = mp.envelope.model_copy(update={"fencing_token": ctx["token"]})
        mp = mp.model_copy(update={"envelope": env})
        if recommend_only:
            report = {"run_id": ctx["run_id"], "mode": "recommend-only", "results": []}
        else:
            ex = Executor(
                make_store(store_kind, tiers_root),
                Catalog(catalog_path),
                state.root / "ledger.jsonl",
                load_planner_config()["max_gb_moved_per_hour"],
                corrupt_first=corrupt_first_move,
                crash_after_moves=crash_after_moves,
            )
            report = ex.execute(mp, state.is_current)
        _write_json(state.root / "last_exec.json", report)
        return report

    def execute_fallback(ctx):  # storage unreachable: keep the plan as a recommendation
        report = {"run_id": ctx["run_id"], "mode": "recommend-only", "results": []}
        _write_json(state.root / "last_exec.json", report)
        return Degraded(report, "executor unavailable: plan kept as recommendation")

    def wrap(name, fn):
        return _inject(fn, faults.get(name))

    return [
        Step("load", wrap("load", load), timeout_s),
        Step("features", wrap("features", features), timeout_s, fallback=features_fallback),
        Step("hotness", wrap("hotness", hotness), timeout_s),
        Step("plan", wrap("plan", plan), timeout_s, fallback=plan_fallback),
        Step("execute", wrap("execute", execute), timeout_s, fallback=execute_fallback),
    ]