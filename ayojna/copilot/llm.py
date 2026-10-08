"""Optional LLM behind the copilot. Two providers, no SDKs (plain HTTPS via httpx):

  anthropic - Claude API, needs ANTHROPIC_API_KEY
  ollama    - a local model (free, works offline), needs `ollama serve`
  none      - no LLM: the copilot answers from deterministic templates

Pick one with AYOJNA_LLM (default: anthropic if a key is set, else none).
Any failure raises LLMUnavailable, so the caller can fall back. The LLM only ever
writes text: it is never in the control loop and cannot move data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx

from ayojna.settings import REPO_ROOT

DEFAULT_MODELS = {"anthropic": "claude-haiku-4-5-20251001", "ollama": "llama3.2"}
DEFAULT_URLS = {"anthropic": "https://api.anthropic.com", "ollama": "http://localhost:11434"}


class LLMUnavailable(RuntimeError):
    """No LLM configured, or the call failed. The copilot falls back to templates."""


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


def config_from_env() -> LLMConfig:
    load_dotenv()
    env = lambda name, default="": os.getenv(name) or default  # noqa: E731 (empty = unset)
    key = env("ANTHROPIC_API_KEY")
    provider = env("AYOJNA_LLM", "anthropic" if key else "none").lower()
    if provider not in ("anthropic", "ollama", "none"):
        provider = "none"
    return LLMConfig(
        provider=provider,
        model=env("AYOJNA_LLM_MODEL", DEFAULT_MODELS.get(provider, "")),
        base_url=env("AYOJNA_LLM_URL", DEFAULT_URLS.get(provider, "")).rstrip("/"),
        api_key=key,
        timeout_s=float(env("AYOJNA_LLM_TIMEOUT", "20")),
    )


def complete(
    system: str, user: str, cfg: LLMConfig | None = None, client: httpx.Client | None = None
) -> str:
    """One question in, one answer out (temperature 0). Raises LLMUnavailable on any problem."""
    cfg = cfg or config_from_env()
    if cfg.provider == "none":
        raise LLMUnavailable("no LLM configured (set ANTHROPIC_API_KEY or AYOJNA_LLM=ollama)")
    if cfg.provider == "anthropic" and not cfg.api_key:
        raise LLMUnavailable("AYOJNA_LLM=anthropic but ANTHROPIC_API_KEY is empty")
    own = client is None
    client = client or httpx.Client(timeout=cfg.timeout_s)
    try:
        if cfg.provider == "anthropic":
            r = client.post(
                f"{cfg.base_url}/v1/messages",
                headers={
                    "x-api-key": cfg.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": cfg.model,
                    "max_tokens": 400,
                    "temperature": 0,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
            r.raise_for_status()
            text = "".join(b.get("text", "") for b in r.json()["content"] if b["type"] == "text")
        else:
            r = client.post(
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
            text = r.json()["message"]["content"]
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        raise LLMUnavailable(f"{cfg.provider} call failed: {type(exc).__name__}: {exc}") from exc
    finally:
        if own:
            client.close()
    if not text.strip():
        raise LLMUnavailable(f"{cfg.provider} returned an empty answer")
    return text.strip()