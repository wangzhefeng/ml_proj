# ml-proj

`ml-proj` 是一个面向工程落地的机器学习建模框架，覆盖：
- 数据读取与切分
- 预处理与特征工程
- 模型训练、调参与评估
- 模型融合（Voting / Stacking / Bagging / Boosting / Averaging）
- 模型部署（FastAPI / Flask / ONNX 导出）
- 训练产物注册、离线预测与在线推理

## 当前状态

- 框架主干已可用：`train / tune / evaluate / predict / serve`。
- 四类任务模板已提供：`classification / regression / clustering / timeseries`。
- 旧代码采用增量迁移策略：旧脚本保留为兼容层，新逻辑统一收口到 `src/mlproj`。

## 项目结构

```text
src/mlproj/
  data/          数据集读取与切分
  preprocess/    预处理接口与实现
  features/      特征流水线
  models/        模型工厂与适配
  selection/     超参数搜索（Grid/Random/Halving/NestedCV）
  fusion/        模型融合能力
  deploy/        模型部署能力（FastAPI/Flask/ONNX）
  training/      训练流程
  evaluation/    评估指标
  registry/      训练产物管理
  inference/     离线预测与在线服务
  analysis/      数据分析接口
  legacy_models/ 旧脚本兼容桥接
```

## 快速开始

```bash
uv sync --all-groups
uv run mlproj train --config configs/classification/train.yaml
uv run mlproj tune --config configs/classification/search.yaml
uv run mlproj evaluate --config configs/regression/train.yaml
```

## 训练产物

```text
artifacts/{task}/{model}/{run_id}/
  model.joblib
  metrics.json
  feature_schema.json
  params.json
  summary.json
```

## 旧脚本迁移（已完成）

- `model_select/*.py` -> `mlproj.selection.search`
- `model_fusion/*.py` -> `mlproj.fusion.ensemble` / `mlproj.legacy_models.fusion_models`
- `model_deploy/*.py` -> `mlproj.deploy.runtime` / `mlproj.legacy_models.deploy_models`
- `feature_engine/*.py` -> `mlproj.features.legacy_engine`
- `data_provider/config_loader.py` -> `mlproj.data.legacy_provider`
- `data_provider/json_loader.py` -> `mlproj.data.legacy_provider`
- `data_provider/yaml_loader.py` -> `mlproj.data.legacy_provider`
- `data_provider/data_loader_lgb.py` -> `mlproj.data.legacy_provider`
- `data_provider/data_loader_xgb.py` -> `mlproj.data.legacy_provider`
- `metrics/metric_report.py` -> `mlproj.evaluation.legacy_metric`
- `metric/metric_report.py` -> `metrics/metric_report.py`（兼容目录）
- `models/*.py` -> `mlproj.legacy_models.root_models`
- `models/supervised/**/*.py` -> `mlproj.legacy_models.subdir_models`
- `models/unsupervised/**/*.py` -> `mlproj.legacy_models.subdir_models`

## 质量校验

```bash
uv run ruff check . --exclude "*.ipynb"
uv run black --check src tests model_select model_fusion model_deploy feature_engine data_provider metrics metric models
uv run pytest -q
```
