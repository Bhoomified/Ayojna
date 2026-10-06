"""Loads and validates the YAML files in config/. Import `load_config()` anywhere."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from ayojna.contracts import Tier

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"


class TierSpec(BaseModel):
    base_latency_ms: float = Field(gt=0)
    price_gb_month: float = Field(ge=0)
    retrieval_per_gb: float = Field(ge=0)
    min_storage_days: int = Field(ge=0)
    capacity_share: float = Field(gt=0, le=1)


class TiersConfig(BaseModel):
    currency: str
    tiers: dict[Tier, TierSpec]
    move_cost_per_gb: float = Field(ge=0)

    @model_validator(mode="after")
    def _all_tiers_present_and_ordered(self) -> "TiersConfig":
        missing = set(Tier) - set(self.tiers)
        if missing:
            raise ValueError(f"tiers.yaml is missing {sorted(t.value for t in missing)}")
        prices = [self.tiers[t].price_gb_month for t in (Tier.HOT, Tier.WARM, Tier.COLD, Tier.ARCHIVE)]
        if prices != sorted(prices, reverse=True):
            raise ValueError("tier prices must fall from hot to archive")
        return self


class SlaClass(BaseModel):
    p95_latency_ms: float = Field(gt=0)


class VolumeTags(BaseModel):
    data_class: str
    sla_class: str
    retention_days: int = Field(ge=0)
    residency: str
    legal_hold: bool
    encryption_required: bool


class Config(BaseModel):
    tiers: TiersConfig
    sla: dict[str, SlaClass]
    volumes: dict[str, VolumeTags]
    default_volume: VolumeTags

    def tags_for(self, volume: str) -> VolumeTags:
        """Tags for a volume; unknown volumes get the defaults."""
        return self.volumes.get(volume, self.default_volume)

    @model_validator(mode="after")
    def _sla_classes_exist(self) -> "Config":
        for name, tags in {**self.volumes, "defaults": self.default_volume}.items():
            if tags.sla_class not in self.sla:
                raise ValueError(f"volume {name} uses unknown sla_class {tags.sla_class!r}")
        return self


def _read(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_config() -> Config:
    tiers = _read("tiers.yaml")
    sla = _read("sla.yaml")["classes"]
    comp = _read("compliance.yaml")
    defaults = comp["defaults"]
    volumes = {name: VolumeTags(**{**defaults, **(tags or {})}) for name, tags in comp["volumes"].items()}
    return Config(tiers=TiersConfig(**tiers), sla=sla, volumes=volumes, default_volume=VolumeTags(**defaults))
