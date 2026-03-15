# 旧脚本迁移记录

本目录用于记录原仓库脚本化示例的迁移进度。

## 迁移策略

- 新框架入口统一在 `src/mlproj`。
- 旧目录先保留，保证可追溯与可回滚。
- 新需求优先在统一接口与 CLI 中实现。

## 当前已迁移

- `data_analysis/*.py` -> `mlproj.analysis.*`
- `metrics/*.py` -> `mlproj.analysis.metrics` / `mlproj.evaluation.legacy_metric`
- `model_select/*.py` -> `mlproj.selection.search`
- `model_fusion/*.py` -> `mlproj.fusion.ensemble` / `mlproj.legacy_models.fusion_models`
- `model_deploy/*.py` -> `mlproj.deploy.runtime` / `mlproj.legacy_models.deploy_models`
- `feature_engine/*.py` -> `mlproj.features.legacy_engine`
- `data_provider/config_loader.py` -> `mlproj.data.legacy_provider`
- `data_provider/json_loader.py` -> `mlproj.data.legacy_provider`
- `data_provider/yaml_loader.py` -> `mlproj.data.legacy_provider`
- `data_provider/data_loader_lgb.py` -> `mlproj.data.legacy_provider`
- `data_provider/data_loader_xgb.py` -> `mlproj.data.legacy_provider`
- `metric/metric_report.py` -> `metrics/metric_report.py`（兼容目录）
- `models/*.py` -> `mlproj.legacy_models.root_models`
- `models/supervised/**/*.py` -> `mlproj.legacy_models.subdir_models`
- `models/unsupervised/**/*.py` -> `mlproj.legacy_models.subdir_models`
