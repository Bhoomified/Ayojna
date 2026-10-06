from ayojna.contracts import Tier
from ayojna.settings import load_config


def test_config_loads_and_is_consistent():
    cfg = load_config()
    assert set(cfg.tiers.tiers) == set(Tier)
    assert cfg.tiers.tiers[Tier.HOT].price_gb_month > cfg.tiers.tiers[Tier.ARCHIVE].price_gb_month
    assert cfg.tags_for("src1_2").legal_hold is True
    assert cfg.tags_for("unknown_volume").sla_class == "general"
