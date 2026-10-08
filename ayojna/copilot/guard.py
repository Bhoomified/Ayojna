"""Grounding guard: an LLM answer may only use numbers that exist in the facts it was given.

If the model invents a figure ("saves 93%"), the answer is rejected and the copilot
falls back to its template. Identifiers like web_0 or L3 are not treated as numbers.
"""

from __future__ import annotations

import re

# a number that is not part of a word (skips web_0, L3, run-2026..., 256MB)
NUMBER = re.compile(r"(?<![\w.\-])\d+(?:\.\d+)?(?![\w])")
SMALL = 10  # small whole numbers ("2 moves", "top 3") are always allowed


def numbers_in(obj) -> set[float]:
    """Every number inside a nested dict/list/str (strings are scanned for numbers too)."""
    out: set[float] = set()
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.add(abs(float(obj)))
    elif isinstance(obj, str):
        out.update(float(x) for x in NUMBER.findall(obj.replace(",", "")))
    elif isinstance(obj, dict):
        for v in obj.values():
            out |= numbers_in(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            out |= numbers_in(v)
    return out


def _close(v: float, allowed: set[float]) -> bool:
    return any(abs(v - a) <= max(0.051, 0.006 * a) for a in allowed)  # rounding is fine


def ungrounded_numbers(answer: str, facts, question: str = "") -> list[str]:
    """Numbers in the answer that appear nowhere in the facts or the question."""
    allowed = numbers_in(facts) | numbers_in(question)
    bad = []
    for tok in NUMBER.findall(answer.replace(",", "")):
        v = float(tok)
        if (v == int(v) and v <= SMALL) or _close(v, allowed):
            continue
        bad.append(tok)
    return bad