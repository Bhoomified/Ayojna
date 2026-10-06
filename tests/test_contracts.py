import pandas as pd
import pytest
from pydantic import ValidationError

from ayojna.contracts import (EXTENT_HOURLY_COLUMNS, ContractError, Envelope, HotnessPrediction,
                              Move, MovePlan, Tier, validate_frame)


def test_prediction_label_and_confidence():
    p = HotnessPrediction(volume="web_0", extent_id=3, hour=10, p_hot=0.7, p_warm=0.2, p_cold=0.1)
    assert p.label == Tier.HOT
    assert p.confidence == pytest.approx(0.7)


def test_prediction_rejects_bad_probabilities():
    with pytest.raises(ValidationError):
        HotnessPrediction(volume="web_0", extent_id=3, hour=10, p_hot=0.7, p_warm=0.7, p_cold=0.1)


def test_move_must_change_tier():
    with pytest.raises(ValidationError):
        Move(volume="web_0", extent_id=1, from_tier=Tier.HOT, to_tier=Tier.HOT,
             size_gb=0.25, expected_saving_per_month=1.0)


def test_plan_total_and_idempotency_key():
    m = Move(volume="web_0", extent_id=1, from_tier=Tier.HOT, to_tier=Tier.COLD,
             size_gb=0.25, expected_saving_per_month=0.02)
    plan = MovePlan(envelope=Envelope(run_id="r1", data_version="d1"), strategy="optimizer", hour=5, moves=[m, m])
    assert plan.total_gb == pytest.approx(0.5)
    assert m.idempotency_key("r1") == "r1:web_0:1:cold"


def _good_rows():
    return pd.DataFrame({"volume": ["web_0"], "extent_id": [1], "hour": [0], "reads": [5], "writes": [1],
                         "read_bytes": [40960], "write_bytes": [8192], "avg_io_size": [8192.0], "rand_ratio": [0.5]})


def test_validate_frame_accepts_good_rows():
    out = validate_frame(_good_rows(), EXTENT_HOURLY_COLUMNS, "extent_hourly")
    assert list(out.columns) == list(EXTENT_HOURLY_COLUMNS)


@pytest.mark.parametrize("col,value", [("reads", -1), ("rand_ratio", 1.5)])
def test_validate_frame_rejects_bad_values(col, value):
    df = _good_rows()
    df[col] = value
    with pytest.raises(ContractError):
        validate_frame(df, EXTENT_HOURLY_COLUMNS, "extent_hourly")


def test_validate_frame_rejects_missing_column():
    with pytest.raises(ContractError):
        validate_frame(_good_rows().drop(columns=["hour"]), EXTENT_HOURLY_COLUMNS, "extent_hourly")
