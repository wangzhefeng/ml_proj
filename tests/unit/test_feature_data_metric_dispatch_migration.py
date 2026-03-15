from mlproj.legacy_models import (
    run_data_provider_legacy_demo,
    run_feature_engine_legacy_demo,
    run_metric_legacy_demo,
)


def test_feature_engine_dispatch_bridge():
    out = run_feature_engine_legacy_demo("feature_engine/FeatureBinning.py")
    assert out["method"] in {"binarization", "feature_engine_bridge"}


def test_data_provider_dispatch_bridge():
    out = run_data_provider_legacy_demo("data_provider/config_loader.py")
    assert out["method"] == "load_yaml"


def test_metric_dispatch_bridge():
    out = run_metric_legacy_demo("metrics/metric_report.py")
    assert out["method"] == "metric_report"
