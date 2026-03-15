from .data_provider_models import run_data_provider_legacy_demo
from .deploy_models import run_deploy_legacy_demo
from .feature_engine_models import run_feature_engine_legacy_demo
from .fusion_models import run_fusion_legacy_demo
from .metric_models import run_metric_legacy_demo
from .root_models import (
    run_featuretools_legacy_demo,
    run_lgb_clf_legacy_demo,
    run_optuna_legacy_demo,
    run_pipeline_legacy_demo,
    run_pls_legacy_demo,
    run_quadratic_legacy_demo,
    run_sklearn_legacy_demo,
)
from .subdir_models import run_subdir_legacy_demo
from .supervised_models import run_supervised_legacy_demo
from .unsupervised_models import run_unsupervised_legacy_demo

__all__ = [
    "run_lgb_clf_legacy_demo",
    "run_featuretools_legacy_demo",
    "run_sklearn_legacy_demo",
    "run_optuna_legacy_demo",
    "run_pipeline_legacy_demo",
    "run_pls_legacy_demo",
    "run_quadratic_legacy_demo",
    "run_subdir_legacy_demo",
    "run_supervised_legacy_demo",
    "run_unsupervised_legacy_demo",
    "run_fusion_legacy_demo",
    "run_deploy_legacy_demo",
    "run_feature_engine_legacy_demo",
    "run_data_provider_legacy_demo",
    "run_metric_legacy_demo",
]
