import json
import os

import httpx

from ayojna.copilot.guard import numbers_in, ungrounded_numbers
from ayojna.copilot.llm import (
    LLMConfig,
    LLMUnavailable,
    complete,
    config_from_env,
    load_dotenv,
)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_no_provider_means_unavailable():
    try:
        complete("s", "q", LLMConfig(provider="none"))
        raise AssertionError("should not answer")
    except LLMUnavailable:
        pass


def test_anthropic_request_and_parse():
    seen = {}

    def handler(req):
        seen["url"], seen["key"] = str(req.url), req.headers["x-api-key"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": " hi "}]})

    cfg = LLMConfig("anthropic", "m1", "https://api.anthropic.com", "k123")
    assert complete("SYS", "Q", cfg, _client(handler)) == "hi"
    assert seen["url"] == "https://api.anthropic.com/v1/messages" and seen["key"] == "k123"
    assert seen["body"]["system"] == "SYS" and seen["body"]["temperature"] == 0
    assert seen["body"]["messages"] == [{"role": "user", "content": "Q"}]


def test_ollama_parse():
    def handler(req):
        assert req.url.path == "/api/chat" and json.loads(req.content)["stream"] is False
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})

    cfg = LLMConfig("ollama", "llama3.2", "http://localhost:11434")
    assert complete("s", "q", cfg, _client(handler)) == "ok"


def test_http_errors_become_unavailable():
    for handler in (
        lambda req: httpx.Response(500, json={"error": "boom"}),
        lambda req: httpx.Response(200, json={"unexpected": True}),
    ):
        try:
            complete("s", "q", LLMConfig("ollama", "m", "http://x"), _client(handler))
            raise AssertionError("should fail")
        except LLMUnavailable:
            pass


def test_timeout_becomes_unavailable():
    def handler(req):
        raise httpx.ReadTimeout("slow", request=req)

    try:
        complete("s", "q", LLMConfig("ollama", "m", "http://x"), _client(handler))
        raise AssertionError("should fail")
    except LLMUnavailable as exc:
        assert "ReadTimeout" in str(exc)


def test_dotenv_does_not_override(tmp_path):
    (tmp_path / ".env").write_text("AYOJNA_TEST_A=from_file\nAYOJNA_TEST_B='x'\n# c=1\n")
    os.environ["AYOJNA_TEST_A"] = "from_env"
    os.environ.pop("AYOJNA_TEST_B", None)
    try:
        load_dotenv(tmp_path / ".env")
        assert os.environ["AYOJNA_TEST_A"] == "from_env" and os.environ["AYOJNA_TEST_B"] == "x"
    finally:
        os.environ.pop("AYOJNA_TEST_A", None)
        os.environ.pop("AYOJNA_TEST_B", None)


def test_guard_accepts_grounded_and_rounded():
    facts = {"saving_pct": 68.53, "volume": "web_0", "level": "L3", "moves": [{"gb": 49.75}]}
    ans = "web_0 saves 68.5% at level L3; 49.75 GB moved in 2 steps."
    assert ungrounded_numbers(ans, facts) == []


def test_guard_flags_invented_numbers():
    facts = {"saving_pct": 68.53}
    assert ungrounded_numbers("It saves 93.7% and $1,200 a month.", facts) == ["93.7", "1200"]


def test_numbers_in_reads_strings():
    assert numbers_in({"why": "I/Os in the last 72 h = 107"}) >= {72.0, 107.0}


def test_empty_env_values_mean_defaults():
    keys = ("AYOJNA_LLM", "AYOJNA_LLM_MODEL", "AYOJNA_LLM_URL", "ANTHROPIC_API_KEY")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        os.environ.update({"AYOJNA_LLM": "ollama", "AYOJNA_LLM_MODEL": "", "AYOJNA_LLM_URL": ""})
        os.environ["ANTHROPIC_API_KEY"] = ""
        cfg = config_from_env()
        assert cfg.model == "llama3.2" and cfg.base_url == "http://localhost:11434"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v