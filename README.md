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
uv run mlproj evaluate --model-uri artifacts/classification/logistic_regression/<run_id>/model.joblib --input dataset/classification/test.csv --target-col target --task classification
```

## evaluate 命令（P1 改造）

`evaluate` 已改为基于“已训练模型”评估，不再触发重新训练：

```bash
uv run mlproj evaluate \
  --model-uri <model.joblib> \
  --input <csv_or_parquet> \
  --target-col <label_column> \
  --task classification \
  --output-metrics outputs/eval_metrics.json
```

也支持配置文件方式（配置内需提供 `model_uri/input`）：

```bash
uv run mlproj evaluate --config configs/classification/evaluate.yaml
```

## 数据加载方式（P1 改造）

1. 单文件 + 自动切分

```yaml
source:
  type: csv
  path: dataset/classification/train.csv
  target: target
split:
  strategy: random
  valid_size: 0.2
  test_size: 0.2
```

2. 显式 train/valid/test（不再二次随机切分）

```yaml
source:
  type: csv
  train_path: dataset/classification/train.csv
  valid_path: dataset/classification/valid.csv
  test_path: dataset/classification/test.csv
  target: target
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

说明：`artifacts/` 与 `outputs/` 已加入 `.gitignore`，默认不再作为开发产物提交。

## 质量校验

```bash
uv run ruff check src tests configs main.py
uv run black --check src tests configs main.py
uv run pytest -q
```

## 2026-03-15 Update
- DatasetLoader now auto-routes to boosting loaders when `model.name` is `lightgbm/lgbm` or `xgboost/xgb` and `source` provides `train_path` + `test_path` (TSV with label in first column).
- Backend objects (LightGBM `Dataset` / XGBoost `DMatrix`) are stored in `DatasetBundle.metadata` for downstream reuse.
