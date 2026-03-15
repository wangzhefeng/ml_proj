# ml-proj

`ml-proj` 是一个面向工程落地的机器学习建模框架，采用“配置驱动 + 分层解耦”设计。

当前任务范围：
- `classification`
- `regression`
- `clustering`
- `pca_reduction`
- `anomaly_detection`
- `topic_modeling`

## 架构分层

```text
src/mlproj/
  data/          数据读取与切分
  preprocess/    预处理模块
  features/      特征工程模块
  models/        模型工厂、后端注册中心、后端适配
  selection/     搜索与调参
  fusion/        模型融合
  training/      训练模块（薄包装）
  evaluation/    评估模块
  inference/     离线预测与在线服务
  pipeline/      统一 PipelineRunner 编排内核
  registry/      产物管理
  utils/         通用工具
```

## 断裂升级说明（V2）

- 旧 CLI 参数路径不再兼容（如 `--config`）。
- 新入口统一为 `run`，`train/tune/evaluate/predict/serve` 为薄包装。
- 配置支持双轨：
  - YAML：`--config-yaml`
  - Python Config Class：`--config-module + --config-class`

## 后端机制（可插拔）

内置后端：`sklearn`、`lightgbm`、`xgboost`、`catboost`。

创建模型由 `model.backend + model.name` 决定；支持 `backend_provider` 动态加载外部后端。

## 快速开始

```bash
uv sync --all-groups
uv run mlproj run --action train --config-yaml configs/classification/train.yaml
uv run mlproj run --action tune --config-yaml configs/regression/search.yaml
uv run mlproj run --action evaluate --config-yaml configs/classification/evaluate.yaml
uv run mlproj run --action predict --config-yaml configs/classification/predict.yaml --override model_uri=artifacts/classification/sklearn_logistic_regression/<run_id>/model.joblib
```

## 核心命令

- 统一入口：
  - `uv run mlproj run --action <train|tune|evaluate|predict|serve> --config-yaml <file.yaml>`
  - `uv run mlproj run --action train --config-module mypkg.my_cfg --config-class TrainConfig`
- 薄包装子命令：
  - `uv run mlproj train --config-yaml <train.yaml>`
  - `uv run mlproj tune --config-yaml <search.yaml>`
  - `uv run mlproj evaluate --config-yaml <evaluate.yaml>`
  - `uv run mlproj predict --config-yaml <predict.yaml>`
  - `uv run mlproj serve --config-yaml <serve.yaml>`
- 覆盖参数：
  - `--override key=value`
  - 示例：`--override model.params.n_estimators=300 --override random_state=7`

## 配置规范

训练/调参配置必填：
- `task`
- `source`
- `model.backend`
- `model.name`

不保留旧字段自动映射。

## 训练产物

```text
artifacts/{task}/{backend_model}/{run_id}/
  model.joblib
  metrics.json
  feature_schema.json
  params.json
  summary.json
  run_spec.json
  stage_trace.json
```

## 质量检查

```bash
uv run ruff check src tests
uv run pytest -q
```
