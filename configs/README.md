# configs

本目录按任务拆分配置，统一包含：
- `train.yaml`
- `search.yaml`
- `predict.yaml`
- `evaluate.yaml`

支持任务：
- `classification`
- `regression`
- `clustering`
- `pca_reduction`
- `anomaly_detection`
- `topic_modeling`

## 断裂升级后的关键规则

- CLI 使用 `--config-yaml`（不再支持 `--config`）。
- 训练/调参配置必须包含：
  - `task`
  - `model.backend`
  - `model.name`

## 推荐命令

- 统一入口：
  - `uv run mlproj run --action train --config-yaml configs/classification/train.yaml`
  - `uv run mlproj run --action tune --config-yaml configs/regression/search.yaml`
  - `uv run mlproj run --action evaluate --config-yaml configs/classification/evaluate.yaml`

- 薄包装：
  - `uv run mlproj train --config-yaml configs/classification/train.yaml`
  - `uv run mlproj tune --config-yaml configs/anomaly_detection/search.yaml`
  - `uv run mlproj evaluate --config-yaml configs/topic_modeling/evaluate.yaml`
  - `uv run mlproj predict --config-yaml configs/classification/predict.yaml --override model_uri=artifacts/.../model.joblib`
