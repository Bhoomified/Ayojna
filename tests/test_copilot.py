import json

from fastapi.testclient import TestClient

from ayojna.api.app import create_app
from ayojna.api.service import Service
from ayojna.copilot.copilot import ask, route
from ayojna.copilot.llm import LLMUnavailable


def _state(tmp_path):
    st, lake = tmp_path / "state", tmp_path / "lake"
    st.mkdir()
    lake.mkdir()
    (st / "catalog.json").write_text(
        json.dumps(
            {
                "web_0/000003": {"tier": "hot", "since": 100},
                "src1_2/000000": {"tier": "hot", "since": 0},
            }
        )
    )
    move = {
        "volume": "web_0",
        "extent_id": 3,
        "from_tier": "hot",
        "to_tier": "warm",
        "size_gb": 0.25,
        "expected_saving_per_month": 0.0138,
        "risk": "low",
    }
    env = {"run_id": "r9", "model_version": "model m1", "policy_version": "abc12345"}
    (st / "last_plan.json").write_text(
        json.dumps(
            {
                "plan": {"envelope": env, "strategy": "ayojna", "hour": 167, "moves": [move]},
                "why": {"r9:web_0:3:warm": "I/Os in the last 72 h = 0"},
            }
        )
    )
    (lake / "scoreboard.json").write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "strategy": "all_hot",
                        "monthly_cost": 13.825,
                        "saving_vs_all_hot_pct": 0.0,
                        "sla_met_pct": 100.0,
                        "compliance_pct": 100.0,
                    },
                    {
                        "strategy": "age_rule",
                        "monthly_cost": 6.501,
                        "saving_vs_all_hot_pct": 52.97,
                        "sla_met_pct": 99.99,
                        "compliance_pct": 88.97,
                    },
                    {
                        "strategy": "ayojna",
                        "monthly_cost": 4.351,
                        "saving_vs_all_hot_pct": 68.53,
                        "sla_met_pct": 100.0,
                        "compliance_pct": 100.0,
                    },
                ]
            }
        )
    )
    return Service(st, lake), st, lake


def _no_llm(system, user):
    raise LLMUnavailable("no LLM configured")


def test_routing():
    assert route("why is web_0/3 on warm?") == ("extent", {"volume": "web_0", "extent_id": 3})
    assert route("explain usr_0 extent 12")[1] == {"volume": "usr_0", "extent_id": 12}
    assert route("any rollbacks last cycle?")[0] == "execution"
    assert route("who is the leader?")[0] == "status"
    assert route("policy for src1_2")[0] == "policy"
    assert route("how much do we save?")[0] == "savings"
    assert route("hello")[0] == "overview"


def test_template_answer_without_llm(tmp_path):
    svc, _, _ = _state(tmp_path)
    r = ask("why is web_0/3 moving?", svc, llm=_no_llm)
    assert r["source"] == "template" and "hot -> warm" in r["answer"]
    assert "I/Os in the last 72 h = 0" in r["answer"]
    r = ask("legal hold on src1_2?", svc, llm=_no_llm)
    assert "legal hold: yes" in r["answer"] and "Allowed tiers: hot." in r["answer"]


def test_grounded_llm_answer_is_used(tmp_path):
    svc, _, _ = _state(tmp_path)
    r = ask(
        "savings?", svc, llm=lambda s, u: "Ayojna saves 68.5% vs all-hot and 33.1% vs age_rule."
    )
    assert r["source"] == "llm"


def test_invented_numbers_are_rejected(tmp_path):
    svc, _, _ = _state(tmp_path)
    r = ask("savings?", svc, llm=lambda s, u: "Ayojna saves 91.2% every month.")
    assert r["source"] == "template" and "91.2" in r["note"] and "68.53%" in r["answer"]


def test_llm_sees_only_facts(tmp_path):
    svc, _, _ = _state(tmp_path)
    seen = {}

    def spy(system, user):
        seen["system"], seen["user"] = system, user
        return "ok"

    ask("why is web_0/3 on hot?", svc, llm=spy)
    assert "ONLY" in seen["system"] and '"tier": "hot"' in seen["user"]


def test_empty_state_is_explained(tmp_path):
    svc = Service(tmp_path / "s", tmp_path / "l")
    r = ask("show the plan", svc, llm=_no_llm)
    assert "Nothing to report yet" in r["answer"]


def test_ask_endpoint(tmp_path):
    _, st, lake = _state(tmp_path)
    c = TestClient(create_app(st, lake))
    r = c.post("/api/ask", json={"question": "why is web_0/3 on hot?"})
    assert r.status_code == 200 and r.json()["intent"] == "extent"
    assert c.post("/api/ask", json={"question": ""}).status_code == 422
    assert "provider" in c.get("/api/copilot").json()