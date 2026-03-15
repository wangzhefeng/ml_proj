from mlproj.legacy_models.subdir_models import run_subdir_legacy_demo
from mlproj.legacy_models.supervised_models import run_supervised_legacy_demo
from mlproj.legacy_models.unsupervised_models import run_unsupervised_legacy_demo


def test_supervised_wrapper_bridge():
    out = run_subdir_legacy_demo("models/supervised/logisticregression.py")
    assert out["task"] == "classification"
    assert out["model_name"] in {"logistic_regression", "random_forest"}


def test_supervised_regression_wrapper_bridge():
    out = run_subdir_legacy_demo("models/supervised/svm/svm_regression.py")
    assert out["task"] == "regression"
    assert out["model_name"] == "random_forest"


def test_unsupervised_wrapper_bridge():
    out = run_subdir_legacy_demo("models/unsupervised/clustering/kmeans.py")
    assert out["task"] == "clustering"
    assert out["model_name"] == "kmeans"


def test_direct_supervised_dispatch():
    out = run_supervised_legacy_demo("models/supervised/nb/gaussian_nb.py")
    assert out["task"] == "classification"


def test_direct_unsupervised_dispatch():
    out = run_unsupervised_legacy_demo("models/unsupervised/clustering/mini_batch_kmeans.py")
    assert out["model_name"] == "kmeans_small"
