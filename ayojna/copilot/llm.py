"""Optional LLM behind the copilot. Plain HTTPS via httpx, no SDKs. Providers:

  gemini    - Google Gemini API, FREE tier (key from aistudio.google.com) -> GEMINI_API_KEY
  groq      - Groq cloud, FREE tier, very fast open models               -> GROQ_API_KEY
  ollama    - local model, free and offline (`ollama serve`)             -> no key
  anthropic - Claude API (paid)                                          -> ANTHROPIC_API_KEY
  none      - no LLM: the copilot answers from deterministic templates

AYOJNA_LLM picks one provider or a fallback CHAIN, e.g. AYOJNA_LLM=gemini,groq,ollama
(tried in order; a timeout, quota error or outage just moves to the next one).
Default: every provider that has a key, in the order gemini, groq, anthropic.
Any total failure raises LLMUnavailable, so the copilot falls back to templates.
The LLM only ever writes text: it is never in the control loop and cannot move data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx

from ayojna.settings import REPO_ROOT

PROVIDERS = {  # name: (key env var, default model, default base url)
    "gemini": (
        "GEMINI_API_KEY",
        "gemini-flash-latest",
        "https://generativelanguage.googleapis.com",
    ),
    "groq": ("GROQ_API_KEY", "llama-3.3-70b-versatile", "https://api.groq.com/openai/v1"),
    "ollama": ("", "llama3.2", "http://localhost:11434"),
    "anthropic": ("ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001", "https://api.anthropic.com"),
}
AUTO_ORDER = ["gemini", "groq", "anthropic"]  # used when AYOJNA_LLM is not set


class LLMUnavailable(RuntimeError):
    """No LLM configured, or every provider failed. The copilot falls back to templates."""


class Answer(str):
    """The answer text, plus which provider wrote it."""

    provider: str = ""


def load_dotenv(path: str | Path = REPO_ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment (never overrides real env vars)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():  # -sig: Notepad BOM
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class LLMConfig:
    provider: str = "none"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    timeout_s: float = 20.0


def _env(name: str, default: str = "") -> str:
    return os.getenv(name) or default  # empty value = not set


def configs_from_env() -> list[LLMConfig]:
    """The provider chain, in the order it will be tried. Empty list = no LLM."""
    load_dotenv()
    chosen = _env("AYOJNA_LLM")
    if chosen:
        names = [n.strip().lower() for n in chosen.split(",") if n.strip()]
    else:
        names = [n for n in AUTO_ORDER if _env(PROVIDERS[n][0])]
    out = []
    for n in names:
        if n not in PROVIDERS:
            continue  # "none" or a typo: skipped
        key_var, model, url = PROVIDERS[n]
        out.append(
            LLMConfig(
                provider=n,
                model=_env(f"AYOJNA_{n.upper()}_MODEL", model),
                base_url=_env(f"AYOJNA_{n.upper()}_URL", url).rstrip("/"),
                api_key=_env(key_var) if key_var else "",
                timeout_s=float(_env("AYOJNA_LLM_TIMEOUT", "20")),
            )
        )
    return out


def config_from_env() -> LLMConfig:
    """The first provider in the chain (shown on the dashboard)."""
    chain = configs_from_env()
    return chain[0] if chain else LLMConfig()


def _call(cfg: LLMConfig, system: str, user: str, client: httpx.Client) -> str:
    key_var = PROVIDERS[cfg.provider][0]
    if key_var and not cfg.api_key:
        raise LLMUnavailable(f"{cfg.provider}: {key_var} is empty")
    if cfg.provider == "gemini":
        r = client.post(
            f"{cfg.base_url}/v1beta/models/{cfg.model}:generateContent",
            headers={"x-goog-api-key": cfg.api_key},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0},
            },
        )
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if cfg.provider == "groq":
        r = client.post(
            f"{cfg.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            json={
                "model": cfg.model,
                "temperature": 0,
                "max_tokens": 400,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    if cfg.provider == "anthropic":
        r = client.post(
            f"{cfg.base_url}/v1/messages",
            headers={"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": cfg.model,
                "max_tokens": 400,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"] if b["type"] == "text")
    r = client.post(  # ollama
        f"{cfg.base_url}/api/chat",
        json={
            "model": cfg.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def complete(
    system: str, user: str, cfg: LLMConfig | None = None, client: httpx.Client | None = None
) -> Answer:
    """Try each provider in the chain (or just `cfg`). Temperature 0. Raises LLMUnavailable."""
    chain = [cfg] if cfg is not None else configs_from_env()
    chain = [c for c in chain if c.provider in PROVIDERS]
    if not chain:
        raise LLMUnavailable("no LLM configured (set GEMINI_API_KEY, GROQ_API_KEY or AYOJNA_LLM)")
    errors = []
    for c in chain:
        own = client is None
        http = client or httpx.Client(timeout=c.timeout_s)
        try:
            text = _call(c, system, user, http).strip()
            if not text:
                raise LLMUnavailable(f"{c.provider}: empty answer")
            ans = Answer(text)
            ans.provider = c.provider
            return ans
        except LLMUnavailable as exc:
            errors.append(str(exc))
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            detail = ""
            if isinstance(exc, httpx.HTTPStatusError):
                detail = f" {exc.response.status_code} {exc.response.text[:120]}"
            errors.append(f"{c.provider}: {type(exc).__name__}{detail or ': ' + str(exc)[:120]}")
        finally:
            if own:
                http.close()
    raise LLMUnavailable("all LLM providers failed -> " + " | ".join(errors))