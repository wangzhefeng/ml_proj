# ml-proj

`ml-proj` 是一个面向工程落地的机器学习建模框架，覆盖：数据读取、预处理、特征工程、模型训练、调参、评估、融合与推理。

## 当前架构

```text
src/mlproj/
  data/          数据读取与切分
  preprocess/    预处理模块
  features/      特征工程模块
  models/        模型工厂与适配器
  selection/     搜索与调参
  fusion/        模型融合
  training/      训练编排
  evaluation/    模型评估
  inference/     离线预测与在线服务
  registry/      产物管理
  utils/         通用工具
```

## 快速开始

```bash
uv sync --all-groups
uv run mlproj train --config configs/classification/train.yaml
uv run mlproj tune --config configs/classification/search.yaml
```

## 核心命令

- 训练：`uv run mlproj train --config <train.yaml>`
- 调参：`uv run mlproj tune --config <search.yaml>`
- 评估：`uv run mlproj evaluate --config <evaluate.yaml>`
- 预测：`uv run mlproj predict --model-uri <model.joblib> --input <input.csv> --output <pred.csv>`
- 服务：`uv run mlproj serve --model-uri <model.joblib> --host 127.0.0.1 --port 8000`

## 数据配置（CSV）

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

2. 显式 train/valid/test

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

## 质量检查

```bash
uv run ruff check src tests
uv run pytest -q
```

## 说明

- 项目仅支持用户自定义数据文件加载（不再使用 sklearn 内置数据源）。
- 当 `model.name` 为 `lightgbm/lgbm` 或 `xgboost/xgb` 且 `source` 提供 `train_path + test_path`（TSV）时，`DatasetLoader` 会自动走对应后端加载分支。