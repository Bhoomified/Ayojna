import time

from ayojna.ingest.build import build
from ayojna.ingest.synth import generate
from ayojna.supervisor.pipeline import build_steps
from ayojna.supervisor.runner import Degraded, Step, run_cycle
from ayojna.supervisor.state import StateStore


def test_lease_fencing_and_takeover(tmp_path):
    store = StateStore(tmp_path, lease_ttl_s=0.3)
    t1 = store.acquire("primary")
    assert t1 == 1 and store.acquire("replica") is None  # lease is held
    time.sleep(0.4)  # primary stops renewing
    t2 = store.acquire("replica")
    assert t2 == 2
    assert not store.is_current(t1) and not store.renew("primary", t1)  # old leader is fenced off


def _store(tmp_path):
    s = StateStore(tmp_path)
    return s, s.acquire("primary")


def test_retry_then_success(tmp_path):
    store, token = _store(tmp_path)
    calls = []

    def flaky(ctx):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("blip")
        return 42

    r = run_cycle("r1", [Step("a", flaky, retries=1)], store, token)
    assert r["level"] == "L0" and r["outputs"]["a"] == 42 and len(calls) == 2


def test_timeout_uses_fallback(tmp_path):
    store, token = _store(tmp_path)
    slow = Step(
        "a", lambda ctx: time.sleep(2), timeout_s=0.2, retries=0, fallback=lambda ctx: "old"
    )
    r = run_cycle("r1", [slow], store, token)
    assert r["level"] == "L1" and r["outputs"]["a"] == "old"


def test_degraded_output_counts_as_fallback(tmp_path):
    store, token = _store(tmp_path)
    r = run_cycle("r1", [Step("a", lambda ctx: Degraded(1, "rule"))], store, token)
    assert r["level"] == "L1" and r["steps"]["a"]["source"] == "fallback"


def test_critical_failure_holds(tmp_path):
    store, token = _store(tmp_path)

    def boom(ctx):
        raise RuntimeError("down")

    later = []
    steps = [Step("a", boom, retries=0), Step("b", lambda ctx: later.append(1))]
    r = run_cycle("r1", steps, store, token)
    assert r["level"] == "L3" and later == []  # nothing after it runs


def test_resume_skips_finished_steps(tmp_path):
    store, token = _store(tmp_path)
    ran = []
    steps = [
        Step("a", lambda ctx: ran.append("a") or 1),
        Step("b", lambda ctx: ran.append("b") or 2),
    ]
    store.save_step("r1", "a", 1, {"source": "primary", "note": ""})  # 'a' finished before a crash
    r = run_cycle("r1", steps, store, token)
    assert ran == ["b"] and r["steps"]["a"]["resumed"]


def test_stale_leader_cannot_run(tmp_path):
    store = StateStore(tmp_path, lease_ttl_s=0.1)
    old = store.acquire("primary")
    time.sleep(0.2)
    store.acquire("replica")
    try:
        run_cycle("r1", [Step("a", lambda ctx: 1)], store, old)
        raise AssertionError("stale leader ran a step")
    except PermissionError:
        pass


def test_real_pipeline_falls_back_without_model(tmp_path):
    generate(tmp_path / "raw", days=3, seed=4)
    build(tmp_path / "raw", tmp_path / "eh.csv")
    store, token = _store(tmp_path / "state")
    steps = build_steps(str(tmp_path / "eh.csv"), str(tmp_path / "f.csv"), str(tmp_path / "none"))
    r = run_cycle("r1", steps, store, token)
    assert r["level"] == "L1"  # no trained model: rule fallback
    assert r["steps"]["hotness"]["source"] == "fallback"
    assert r["outputs"]["plan"].envelope.fencing_token == token