import json
import os

import httpx
import ayojna.copilot.llm as llm
from ayojna.copilot.guard import numbers_in, ungrounded_numbers
from ayojna.copilot.llm import (
    LLMConfig,
    LLMUnavailable,
    complete,
    configs_from_env,
    load_dotenv,
)

ENV_KEYS = (
    "AYOJNA_LLM",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "ANTHROPIC_API_KEY",
    "AYOJNA_OLLAMA_MODEL",
    "AYOJNA_OLLAMA_URL",
)


class _env:
    """Temporarily set environment variables (restored afterwards)."""

    def __init__(self, **values):
        self.values, self.saved = values, {}

    def __enter__(self):
        for k in ENV_KEYS:
            self.saved[k] = os.environ.pop(k, None)
        os.environ.update(self.values)

    def __exit__(self, *exc):
        for k in ENV_KEYS:
            os.environ.pop(k, None)
            if self.saved[k] is not None:
                os.environ[k] = self.saved[k]

llm.RETRY_WAIT_S = (0, 0)  # no real waiting in tests
llm.load_dotenv = lambda *args, **kwargs: None  # tests never read your real .env

def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_no_provider_means_unavailable():
    with _env(AYOJNA_LLM="none"):
        try:
            complete("s", "q")
            raise AssertionError("should not answer")
        except LLMUnavailable:
            pass


def test_gemini_request_and_parse():
    seen = {}

    def handler(req):
        seen["url"], seen["key"] = str(req.url), req.headers["x-goog-api-key"]
        seen["body"] = json.loads(req.content)
        parts = [{"text": "thinking...", "thought": True}, {"text": " Saves 68.5%. "}]
        return httpx.Response(200, json={"candidates": [{"content": {"parts": parts}}]})

    cfg = LLMConfig("gemini", "gemini-flash-latest", "https://g.test", "gk")
    ans = complete("SYS", "Q", cfg, _client(handler))
    assert ans == "Saves 68.5%." and ans.provider == "gemini"  # thought parts dropped
    assert seen["url"] == "https://g.test/v1beta/models/gemini-flash-latest:generateContent"
    assert seen["key"] == "gk" and seen["body"]["systemInstruction"]["parts"][0]["text"] == "SYS"
    assert seen["body"]["generationConfig"]["temperature"] == 0


def test_groq_request_and_parse():
    def handler(req):
        assert req.url.path.endswith("/chat/completions")
        assert req.headers["authorization"] == "Bearer qk"
        body = json.loads(req.content)
        assert body["messages"][0] == {"role": "system", "content": "S"}
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    cfg = LLMConfig("groq", "llama-3.3-70b-versatile", "https://q.test/openai/v1", "qk")
    assert complete("S", "Q", cfg, _client(handler)).provider == "groq"


def test_anthropic_and_ollama_parse():
    def handler(req):
        if req.url.path == "/v1/messages":
            assert req.headers["x-api-key"] == "ak"
            return httpx.Response(200, json={"content": [{"type": "text", "text": "a"}]})
        return httpx.Response(200, json={"message": {"content": "o"}})

    c = _client(handler)
    assert complete("s", "q", LLMConfig("anthropic", "m", "https://a.test", "ak"), c) == "a"
    assert complete("s", "q", LLMConfig("ollama", "m", "http://o.test"), c) == "o"


def test_chain_falls_through_to_next_provider():
    def handler(req):
        if "generateContent" in req.url.path:
            return httpx.Response(429, json={"error": {"message": "quota exceeded"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "from groq"}}]})

    with _env(AYOJNA_LLM="gemini,groq", GEMINI_API_KEY="g", GROQ_API_KEY="q"):
        ans = complete("s", "q", client=_client(handler))
    assert ans == "from groq" and ans.provider == "groq"


def test_all_failing_is_unavailable_with_reasons():
    def handler(req):
        raise httpx.ConnectError("offline", request=req)

    with _env(AYOJNA_LLM="gemini,ollama", GEMINI_API_KEY="g"):
        try:
            complete("s", "q", client=_client(handler))
            raise AssertionError("should fail")
        except LLMUnavailable as exc:
            assert "gemini: ConnectError" in str(exc) and "ollama: ConnectError" in str(exc)


def test_missing_key_is_reported_not_sent():
    with _env(AYOJNA_LLM="gemini"):
        try:
            complete("s", "q", client=_client(lambda req: httpx.Response(200)))
            raise AssertionError("should fail")
        except LLMUnavailable as exc:
            assert "GEMINI_API_KEY is empty" in str(exc)


def test_auto_chain_uses_providers_with_keys():
    with _env(GROQ_API_KEY="q", GEMINI_API_KEY="g"):
        assert [c.provider for c in configs_from_env()] == ["gemini", "groq"]
    with _env(AYOJNA_LLM="ollama", AYOJNA_OLLAMA_MODEL="", AYOJNA_OLLAMA_URL=""):
        (c,) = configs_from_env()
        assert c.model == "llama3.2" and c.base_url == "http://localhost:11434"


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

def test_busy_model_is_retried_then_succeeds():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": {"message": "high demand"}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    cfg = LLMConfig("gemini", "gemini-flash-latest", "https://g.test", "gk")
    assert complete("s", "q", cfg, _client(handler)) == "ok" and len(calls) == 2


def test_second_model_used_when_first_is_missing_or_busy():
    def handler(req):
        if "model-a" in req.url.path:
            return httpx.Response(404, json={"error": {"message": "not found"}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "b"}]}}]})

    cfg = LLMConfig("gemini", "model-a, model-b", "https://g.test", "gk")
    assert complete("s", "q", cfg, _client(handler)) == "b"


def test_groq_gpt_oss_asks_for_low_reasoning():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    cfg = LLMConfig("groq", "openai/gpt-oss-120b", "https://q.test/openai/v1", "qk")
    complete("s", "q", cfg, _client(handler))
    assert seen["body"]["reasoning_effort"] == "low" and seen["body"]["max_tokens"] == 1024