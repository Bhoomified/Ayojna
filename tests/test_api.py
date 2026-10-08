import json

from fastapi.testclient import TestClient

from ayojna.api.app import create_app
from ayojna.api.snapshot import snapshot
from ayojna.ingest.build import build
from ayojna.ingest.synth import generate
from ayojna.models.features import build_features
from ayojna.io import write_table
from ayojna.supervisor.pipeline import build_steps
from ayojna.supervisor.runner import run_cycle
from ayojna.supervisor.state import StateStore


def test_empty_dirs_do_not_crash(tmp_path):
    c = TestClient(create_app(tmp_path / "state", tmp_path / "lake"))
    for url in ["/api/status", "/api/placement", "/api/plan", "/api/execution", "/api/scoreboard"]:
        r = c.get(url)
        assert r.status_code == 200 and r.json()["available"] is False
    assert c.get("/api/audit").json() == []
    assert c.get("/api/kpis").json()["level"] is None


def test_full_loop_through_the_api(tmp_path):
    generate(tmp_path / "raw", days=5, seed=4)
    eh = build(tmp_path / "raw", tmp_path / "eh.csv")
    write_table(build_features(eh), tmp_path / "f.csv")
    lake, state = tmp_path / "lake", tmp_path / "state"
    board = snapshot(
        str(tmp_path / "eh.csv"),
        str(tmp_path / "f.csv"),
        str(tmp_path / "none"),
        str(lake / "scoreboard.json"),
    )
    assert {r["strategy"] for r in board["summary"]} >= {"all_hot", "ayojna"}

    store = StateStore(state, lease_ttl_s=60)
    token = store.acquire("primary")
    steps = build_steps(
        str(tmp_path / "eh.csv"),
        str(tmp_path / "f.csv"),
        str(tmp_path / "none"),
        state_dir=str(state),
        tiers_root=str(tmp_path / "tiers"),
    )
    r = run_cycle("r1", steps, store, token)
    ex = r["outputs"]["execute"]
    store.audit(
        {
            "event": "cycle",
            "run_id": "r1",
            "token": token,
            "owner": "primary",
            "level": r["level"],
            "moves_planned": len(r["outputs"]["plan"].moves),
            "moves_done": ex["done"],
            "rolled_back": 0,
            "gb_moved": ex["gb_moved"],
        }
    )

    c = TestClient(create_app(state, lake))
    k = c.get("/api/kpis").json()
    assert k["level"] == "L1" and k["moves_last_cycle"] == ex["done"] > 0
    assert k["twin_saving_vs_all_hot_pct"] > 0 and k["live_saving_pct"] > 0
    s = c.get("/api/status").json()
    assert s["leader"] == "primary" and s["fencing_token"] == token and s["leader_alive"]
    p = c.get("/api/plan?limit=5").json()
    assert p["n_moves"] == ex["done"] and len(p["moves"]) == 5 and all(m["why"] for m in p["moves"])
    place = c.get("/api/placement").json()
    assert sum(place["extents"].values()) == len(json.loads((state / "catalog.json").read_text()))
    e = c.get("/api/explain/src1_2/0").json()
    assert e["allowed_tiers"] == [e["tier"]] and "legal hold" in e["policy"]
    assert c.get("/api/audit?limit=1").json()[0]["event"] == "cycle"  # newest first
    assert c.get("/").status_code == 200