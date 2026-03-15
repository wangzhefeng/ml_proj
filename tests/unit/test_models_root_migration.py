from mlproj.legacy_models import (
    run_featuretools_legacy_demo,
    run_optuna_legacy_demo,
    run_pipeline_legacy_demo,
    run_pls_legacy_demo,
    run_quadratic_legacy_demo,
)


def test_featuretools_legacy_bridge():
    out = run_featuretools_legacy_demo()
    assert out["rows"] > 0
    assert out["cols"] > 0


def test_pipeline_legacy_bridge():
    out = run_pipeline_legacy_demo()
    assert out["train_rows"] > 0
    assert out["train_cols"] > 0


def test_pls_legacy_bridge():
    pred = run_pls_legacy_demo()
    assert pred.shape[0] == 4


def test_quadratic_legacy_bridge():
    out = run_quadratic_legacy_demo()
    assert "best_value" in out


def test_optuna_legacy_bridge():
    out = run_optuna_legacy_demo()
    assert "best_score" in out
