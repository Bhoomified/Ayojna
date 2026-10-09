
import json

import numpy as np
import pytest

from ayojna.contracts import HotnessPrediction, Source, Status
from ayojna.ingest.build import build
from ayojna.ingest.synth import generate
from ayojna.models.features import build_features, time_split
from ayojna.models.hotness import HotnessModel, train
from ayojna.models.predictor import LATEST, predict_hotness


@pytest.fixture
def split(tmp_path):
    generate(tmp_path / "raw", days=5, seed=11)
    eh = build(tmp_path / "raw", tmp_path / "eh.csv")
    feats = build_features(eh, hot_min=50, horizon=24)
    return time_split(feats, test_hours=24, horizon=24)


def test_model_outputs_valid_predictions_with_reasons(split):
    tr, te = split
    model = train(tr)
    pred = model.predict(te.head(300))
    assert np.allclose(pred[["p_hot", "p_warm", "p_cold"]].sum(axis=1), 1.0)
    r = pred.iloc[0]
    HotnessPrediction(
        volume=r.volume,
        extent_id=int(r.extent_id),
        hour=int(r.hour),
        p_hot=r.p_hot,
        p_warm=r.p_warm,
        p_cold=r.p_cold,
        top_reasons=[x for x in r.reasons.split("; ") if x][:3],
    )  # fits the contract
    assert (pred["reasons"].str.count(";") <= 2).all()
    assert (pred["reasons"].str.len() > 0).mean() > 0.95  # nearly every prediction is explained


def test_save_load_roundtrip(split, tmp_path):
    tr, te = split
    model = train(tr)
    path = model.save(tmp_path / "store")
    again = HotnessModel.load(path)
    assert np.allclose(model.predict_proba(te.head(50)), again.predict_proba(te.head(50)))


def test_predictor_falls_back_without_a_model(split, tmp_path):
    _, te = split
    pred, source, status, note = predict_hotness(te.head(100), store=tmp_path / "empty")
    assert source == Source.FALLBACK and status == Status.DEGRADED
    assert pred["pred"].isin(["hot", "warm", "cold"]).all()


def test_predictor_falls_back_on_corrupt_model(split, tmp_path):
    _, te = split
    store = tmp_path / "store"
    store.mkdir()
    (store / "bad.joblib").write_text("not a model")
    (store / LATEST).write_text(json.dumps({"file": "bad.joblib"}))
    _, source, status, _ = predict_hotness(te.head(50), store=store)
    assert source == Source.FALLBACK and status == Status.DEGRADED


def test_predictor_uses_promoted_model(split, tmp_path):
    tr, te = split
    store = tmp_path / "store"
    path = train(tr).save(store)
    (store / LATEST).write_text(json.dumps({"file": path.name}))
    _, source, status, note = predict_hotness(te.head(300), store=store)
    assert source == Source.PRIMARY and status == Status.OK and note.startswith("model")



def test_predictions_carry_signed_feature_drivers(split):
    import json

    train_df, test_df = split
    model = train(train_df)
    pred = model.predict(test_df.head(50))
    drivers = [json.loads(d) for d in pred["drivers"]]
    assert any(drivers) and all(len(d) <= 5 for d in drivers)
    assert all({"feature", "value", "effect"} <= set(x) for d in drivers for x in d)
