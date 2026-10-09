import json

from ayojna.contracts import Envelope, Move, MovePlan, Tier
from ayojna.executor.catalog import Catalog
from ayojna.executor.saga import Executor
from ayojna.executor.tierstore import FsTierStore, object_key


def _setup(tmp_path, budget_gb=10.0, **kw):
    store = FsTierStore(tmp_path / "tiers")
    cat = Catalog(tmp_path / "catalog.json")
    cat.seed(["web_0", "web_0"], [1, 2], store, hour=0)
    ex = Executor(store, cat, tmp_path / "ledger.jsonl", budget_gb, **kw)
    return store, cat, ex


def _plan(token=1):
    env = Envelope(run_id="r1", data_version="t", fencing_token=token)
    moves = [
        Move(
            volume="web_0",
            extent_id=e,
            from_tier=Tier.HOT,
            to_tier=Tier.WARM,
            size_gb=0.25,
            expected_saving_per_month=0.01,
        )
        for e in (1, 2)
    ]
    return MovePlan(envelope=env, strategy="ayojna", hour=5, moves=moves)


def test_moves_data_and_switches_catalog(tmp_path):
    store, cat, ex = _setup(tmp_path)
    r = ex.execute(_plan(), lambda t: True)
    assert r["done"] == 2 and r["gb_moved"] == 0.5
    k = object_key("web_0", 1)
    assert store.exists("warm", k) and not store.exists("hot", k)  # moved, not copied twice
    assert Catalog(tmp_path / "catalog.json").tier_of(k) == "warm"  # persisted


def test_replay_is_idempotent(tmp_path):
    _, _, ex = _setup(tmp_path)
    ex.execute(_plan(), lambda t: True)
    r = ex.execute(_plan(), lambda t: True)  # e.g. a replica resumes the same run
    assert r["done"] == 0 and r["skipped"] == 2


def test_bad_copy_rolls_back(tmp_path):
    store, cat, ex = _setup(tmp_path, corrupt_first=True)
    r = ex.execute(_plan(), lambda t: True)
    assert r["rolled_back"] == 1 and r["done"] == 1
    k = object_key("web_0", 1)
    assert cat.tier_of(k) == "hot" and store.exists("hot", k) and not store.exists("warm", k)


def test_fenced_leader_moves_nothing(tmp_path):
    store, cat, ex = _setup(tmp_path)
    r = ex.execute(_plan(), lambda t: False)
    assert r["fenced"] and r["done"] == 0 and cat.counts()["hot"] == 2


def test_budget_caps_moves(tmp_path):
    _, _, ex = _setup(tmp_path, budget_gb=0.25)
    r = ex.execute(_plan(), lambda t: True)
    assert r["done"] == 1 and r["over_budget"] == 1


def test_ledger_has_idempotency_keys(tmp_path):
    _, _, ex = _setup(tmp_path)
    ex.execute(_plan(), lambda t: True)
    keys = [json.loads(x)["key"] for x in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert keys == ["r1:web_0:1:warm", "r1:web_0:2:warm"]



def test_only_approved_moves_run_in_approval_mode(tmp_path):
    store, cat, ex = _setup(tmp_path)
    verdicts = {1: "approved", 2: "rejected"}
    r = ex.execute(_plan(), lambda t: True, decide=lambda m: verdicts[m.extent_id])
    assert r["done"] == 1 and r["rejected"] == 1 and cat.tier_of(object_key("web_0", 2)) == "hot"
    r = ex.execute(_plan(), lambda t: True, decide=lambda m: "pending")
    assert r["pending"] == 2