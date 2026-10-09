import json

from fastapi.testclient import TestClient

from ayojna.api.app import create_app
from ayojna.ingest.build import build
from ayojna.ingest.synth import generate
from ayojna.models.features import build_features
from ayojna.io import write_table
from ayojna.supervisor.pipeline import build_steps
from ayojna.supervisor.runner import run_cycle
from ayojna.supervisor.state import StateStore


def _cycle(tmp_path, run_id, **kw):
    state = tmp_path / "state"
    store = StateStore(state, lease_ttl_s=60)
    token = store.acquire("primary")
    steps = build_steps(
        str(tmp_path / "eh.csv"),
        str(tmp_path / "f.csv"),
        str(tmp_path / "none"),
        state_dir=str(state),
        tiers_root=str(tmp_path / "tiers"),
        **kw,
    )
    return run_cycle(run_id, steps, store, token)


def _data(tmp_path):
    generate(tmp_path / "raw", days=3, seed=4)
    eh = build(tmp_path / "raw", tmp_path / "eh.csv")
    write_table(build_features(eh), tmp_path / "f.csv")


def test_every_move_has_a_four_stage_decision_card(tmp_path):
    _data(tmp_path)
    _cycle(tmp_path, "r1", recommend_only=True)
    saved = json.loads((tmp_path / "state" / "last_plan.json").read_text())
    assert saved["plan"]["moves"] and len(saved["cards"]) == len(saved["plan"]["moves"])
    card = next(iter(saved["cards"].values()))
    assert {"prediction", "policy", "costs_24h", "decision"} <= set(card)
    assert set(card["costs_24h"]) == {"hot", "warm", "cold", "archive"}
    assert card["decision"]["to"] in card["policy"]["allowed"]


def test_recommendations_group_and_only_approved_groups_run(tmp_path):
    _data(tmp_path)
    _cycle(tmp_path, "r1", approval_mode=True)  # nothing approved yet: nothing moves
    c = TestClient(create_app(tmp_path / "state", tmp_path / "lake"))
    rec = c.get("/api/recommendations").json()
    assert rec["summary"]["moves"] > 0 and rec["summary"]["pending"] == len(rec["items"])
    first = rec["items"][0]
    assert (
        first["action"].split()[0] in ("Promote", "Demote", "Archive")
        and first["drivers"] is not None
    )
    detail = c.get("/api/recommendations/detail", params={"group": first["id"]}).json()
    assert detail["available"] and detail["items"][0]["decision"]["to"] == first["to"]
    ok = c.post("/api/recommendations/decide", json={"group": first["id"], "decision": "approved"})
    assert ok.status_code == 200
    bad = c.post("/api/recommendations/decide", json={"group": first["id"], "decision": "maybe"})
    assert bad.status_code == 422
    r = _cycle(tmp_path, "r2", approval_mode=True)
    ex = r["outputs"]["execute"]
    assert ex["mode"] == "approval" and ex["done"] == first["extents"]
    assert ex["pending"] == rec["summary"]["moves"] - first["extents"]