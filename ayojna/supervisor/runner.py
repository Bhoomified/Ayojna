"""Runs a cycle of steps with timeout, retry, fallback, checkpoint and a degradation level.

Levels: L0 normal · L1 some fallbacks used · L3 hold (a critical step failed: no moves).
(L2 recommend-only and L4 safe mode are set by the pipeline and the CLI.)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from ayojna.supervisor.state import StateStore


@dataclass
class Degraded:
    """Return this from a step that answered with its own internal fallback."""

    output: Any
    note: str


@dataclass
class Step:
    name: str
    fn: Callable[[dict], Any]  # receives the outputs of earlier steps
    timeout_s: float = 30.0
    retries: int = 1
    fallback: Callable[[dict], Any] | None = None
    critical: bool = True  # if it fails with no fallback, the cycle holds


def _call(fn, ctx, timeout_s):
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn, ctx).result(timeout=timeout_s)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def run_cycle(run_id: str, steps: list[Step], store: StateStore, token: int) -> dict:
    ctx: dict = {"run_id": run_id, "token": token}
    report = {"run_id": run_id, "level": "L0", "steps": {}}
    store.set_active_run(run_id)
    for step in steps:
        if not store.is_current(token):
            raise PermissionError("lost the leader lease: another supervisor took over")
        saved, info = store.load_step(run_id, step.name)
        if info is not None:  # finished before a crash: resume, do not redo
            ctx[step.name] = saved
            report["steps"][step.name] = {**info, "resumed": True}
            if info["source"] == "fallback" and report["level"] == "L0":
                report["level"] = "L1"
            continue
        source, note, out = "primary", "", None
        for attempt in range(step.retries + 1):
            try:
                out = _call(step.fn, ctx, step.timeout_s)
                break
            except Exception as exc:  # includes TimeoutError
                note = f"{type(exc).__name__}: {exc}"
                if attempt < step.retries:
                    time.sleep(0.2 * 2**attempt)
        else:
            if step.fallback is not None:
                try:
                    out, source = _call(step.fallback, ctx, step.timeout_s), "fallback"
                except Exception as exc:
                    source, note = "failed", f"fallback failed: {exc}"
            else:
                source = "failed"
        if isinstance(out, Degraded):
            out, source, note = out.output, "fallback", out.note
        info = {"source": source, "note": note}
        store.audit({"run_id": run_id, "token": token, "step": step.name, **info})
        report["steps"][step.name] = info
        if source == "failed":
            if step.critical:
                report["level"] = "L3"
                break
            continue
        if source == "fallback" and report["level"] == "L0":
            report["level"] = "L1"
        ctx[step.name] = out
        store.save_step(run_id, step.name, out, info)
    store.set_active_run(None)
    report["outputs"] = ctx
    return report