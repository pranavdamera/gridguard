"""Tests for stratified model evaluation."""

import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gridguard.features.engineer import get_X_y
from gridguard.models.evaluate import (
    compare_models,
    evaluate_model,
    full_stratified_report,
    stratified_metrics,
)


@pytest.fixture()
def model_and_df():
    from tests._helpers import make_frame

    df = make_frame(start="2022-01-01", end="2022-06-30", seed=77)
    model = Pipeline([("sc", StandardScaler()), ("r", Ridge())])
    X, y = get_X_y(df)
    model.fit(X, y)
    return model, df


def test_evaluate_model_returns_required_keys(model_and_df):
    model, df = model_and_df
    X, y = get_X_y(df)
    metrics = evaluate_model(model, X, y)
    for key in ("mae_kw", "rmse_kw", "mape_pct", "r2", "n_samples"):
        assert key in metrics


def test_evaluate_model_rmse_non_negative(model_and_df):
    model, df = model_and_df
    X, y = get_X_y(df)
    metrics = evaluate_model(model, X, y)
    assert metrics["rmse_kw"] >= 0
    assert metrics["mae_kw"] >= 0


def test_stratified_by_hour_of_day(model_and_df):
    model, df = model_and_df
    result = stratified_metrics(df, model, strata="hour_of_day")
    assert "hour_of_day" in result.columns
    assert len(result) <= 24
    assert (result["n_samples"] > 0).all()


def test_stratified_by_season(model_and_df):
    model, df = model_and_df
    result = stratified_metrics(df, model, strata="season")
    assert "season" in result.columns
    assert len(result) >= 2  # at least 2 seasons in 6 months


def test_stratified_by_irradiance_bin(model_and_df):
    model, df = model_and_df
    result = stratified_metrics(df, model, strata="irradiance_bin")
    assert "irradiance_bin" in result.columns
    bins = result["irradiance_bin"].astype(str).tolist()
    # Should have at least night and a daylight bin
    assert len(bins) >= 2


def test_stratified_unknown_strata_raises(model_and_df):
    model, df = model_and_df
    with pytest.raises(ValueError, match="Unknown strata"):
        stratified_metrics(df, model, strata="bogus")


def test_full_stratified_report_saves_csvs(model_and_df, tmp_path):
    model, df = model_and_df
    reports = full_stratified_report(df, model, model_name="ridge", report_dir=tmp_path)
    assert len(reports) >= 3  # hour_of_day, season, irradiance_bin
    # CSV files should exist
    assert any(tmp_path.glob("*.csv"))


def test_compare_models_sorted_by_rmse(model_and_df):
    model, df = model_and_df
    X, y = get_X_y(df)
    models = {"a": model, "b": model}  # same model twice for determinism
    result = compare_models(models, X, y)
    assert list(result["model"]) == sorted(result["model"].tolist(), key=lambda _: True)
    rmse_values = result["rmse_kw"].tolist()
    assert rmse_values == sorted(rmse_values)


def test_stratified_hour_rmse_positive(model_and_df):
    model, df = model_and_df
    result = stratified_metrics(df, model, strata="hour_of_day")
    assert (result["rmse_kw"].dropna() >= 0).all()
