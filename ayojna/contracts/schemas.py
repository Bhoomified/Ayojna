"""Shared data contracts for every Ayojna module.

Rule: a module may only send or receive the shapes defined here. If you need a
new field, change this file in its own pull request and tell the other person.

Two kinds of contracts live here:
1. Pydantic models for single messages (an envelope, a move, a plan).
2. Column specs for bulk tables (DataFrames saved to the data lake).
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

# One extent = a 256 MB slice of a volume. It is the unit Ayojna tiers.
EXTENT_MB = 256
EXTENT_BYTES = EXTENT_MB * 1024 * 1024


class Tier(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    ARCHIVE = "archive"


# Fastest to slowest. Use this order whenever tiers are compared.
TIER_ORDER: list[Tier] = [Tier.HOT, Tier.WARM, Tier.COLD, Tier.ARCHIVE]


class Status(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"


class Source(str, Enum):
    PRIMARY = "primary"
    FALLBACK = "fallback"
    CACHE = "cache"


class Envelope(BaseModel):
    """Travels with every message between modules (see the design doc)."""

    run_id: str
    data_version: str
    model_version: str = "none"
    policy_version: str = "none"
    fencing_token: int = Field(default=0, ge=0)
    status: Status = Status.OK
    source: Source = Source.PRIMARY
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HotnessPrediction(BaseModel):
    volume: str
    extent_id: int = Field(ge=0)
    hour: int = Field(ge=0)
    p_hot: float = Field(ge=0, le=1)
    p_warm: float = Field(ge=0, le=1)
    p_cold: float = Field(ge=0, le=1)
    top_reasons: list[str] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def _probabilities_sum_to_one(self) -> "HotnessPrediction":
        total = self.p_hot + self.p_warm + self.p_cold
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"probabilities must sum to 1, got {total:.4f}")
        return self

    @property
    def label(self) -> Tier:
        best = max((self.p_hot, Tier.HOT), (self.p_warm, Tier.WARM), (self.p_cold, Tier.COLD))
        return best[1]

    @property
    def confidence(self) -> float:
        return max(self.p_hot, self.p_warm, self.p_cold)


class AllowedTiers(BaseModel):
    volume: str
    extent_id: int = Field(ge=0)
    tiers: list[Tier]  # may be empty: then the extent must stay where it is
    denials: list[str] = Field(default_factory=list)


class Move(BaseModel):
    volume: str
    extent_id: int = Field(ge=0)
    from_tier: Tier
    to_tier: Tier
    size_gb: float = Field(gt=0)
    expected_saving_per_month: float
    risk: Literal["low", "med", "high"] = "low"

    @model_validator(mode="after")
    def _must_change_tier(self) -> "Move":
        if self.from_tier == self.to_tier:
            raise ValueError("a move must change tier")
        return self

    def idempotency_key(self, run_id: str) -> str:
        return f"{run_id}:{self.volume}:{self.extent_id}:{self.to_tier.value}"
    
    def group_key(self) -> str:
        """Recommendations are grouped (and approved) per volume and direction."""
        return f"{self.volume}:{self.from_tier.value}>{self.to_tier.value}"


class MovePlan(BaseModel):
    envelope: Envelope
    strategy: str
    hour: int = Field(ge=0)
    moves: list[Move] = Field(default_factory=list)

    @property
    def total_gb(self) -> float:
        return sum(m.size_gb for m in self.moves)


class SimMetrics(BaseModel):
    strategy: str
    hour: int = Field(ge=0)
    cost: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    sla_met_pct: float = Field(ge=0, le=100)
    gb_moved: float = Field(ge=0)
    compliance_pct: float = Field(ge=0, le=100)

    @field_validator("strategy")
    @classmethod
    def _strategy_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("strategy name is empty")
        return v


# --------------------------------------------------------------------------
# Bulk tables (DataFrames). Column name -> pandas dtype.
# --------------------------------------------------------------------------

EXTENT_HOURLY_COLUMNS: dict[str, str] = {
    "volume": "string",
    "extent_id": "int64",
    "hour": "int64",          # hours since the start of the trace
    "reads": "int64",
    "writes": "int64",
    "read_bytes": "int64",
    "write_bytes": "int64",
    "avg_io_size": "float64",  # bytes per I/O in that hour
    "rand_ratio": "float64",   # share of I/Os that were not sequential, 0..1
}


class ContractError(ValueError):
    """Raised when a DataFrame does not match its contract."""


def validate_frame(df: pd.DataFrame, columns: dict[str, str], name: str) -> pd.DataFrame:
    """Check columns, cast dtypes and reject impossible values. Returns a clean copy."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ContractError(f"{name}: missing columns {missing}")
    out = df[list(columns)].copy()
    try:
        out = out.astype(columns)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name}: cannot cast columns: {exc}") from exc
    numeric = [c for c, t in columns.items() if t != "string"]
    if out[numeric].isna().any().any():
        raise ContractError(f"{name}: null values in numeric columns")
    if (out[numeric] < 0).any().any():
        raise ContractError(f"{name}: negative values")
    if "rand_ratio" in out and (out["rand_ratio"] > 1).any():
        raise ContractError(f"{name}: rand_ratio above 1")
    return out
