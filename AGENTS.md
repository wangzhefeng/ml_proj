# AGENTS.md

本文件是 `ml-proj` 的项目级 Codex 协作规范（忽略目录：`todo/`）。

## 1. 项目定位

- 项目名：`ml-proj`
- Python：`>=3.10`
- 包管理：`uv`
- CLI 入口：`mlproj`
- 源码目录：`src/mlproj`
- 测试目录：`tests`
- 当前任务范围：`classification`、`regression`、`clustering`、`pca_reduction`、`anomaly_detection`、`topic_modeling`

## 2. 架构分层

- `data`：数据读取与切分
- `preprocess`：数据预处理
- `features`：特征工程流水线
- `models`：模型工厂、后端注册中心、后端适配器
- `selection`：搜索与调参
- `fusion`：模型融合（Voting/Stacking/Bagging/Boosting）
- `training`：训练编排
- `evaluation`：评估器与指标
- `inference`：离线预测与在线服务
- `registry`：训练产物与运行输出
- `utils`：日志与公共工具

## 3. 模型后端约束

- 配置必须显式声明：`model.backend` + `model.name`
- 内置后端：`sklearn`、`lightgbm`、`xgboost`、`catboost`
- 可通过 `backend_provider` 动态加载第三方后端模块
- 不保留旧配置字段自动映射（断裂升级）

## 4. 工作约束

- 忽略目录：`todo/`
- 优先修改 `src/mlproj` 内模块，不做临时脚本式改造
- 保持 `train/evaluate/predict/tune/serve` 的 CLI 行为一致
- 修改接口时，必须同步更新测试
- 新增模型与测试默认使用确定性 `random_state`

## 5. 质量门禁

- `uv run ruff check src tests`
- `uv run pytest -q`

## 6. 常用命令

- 环境同步：`uv sync --all-groups`
- 训练：`uv run mlproj train --config configs/classification/train.yaml`
- 调参：`uv run mlproj tune --config configs/classification/search.yaml`
- 评估：`uv run mlproj evaluate --config configs/classification/evaluate.yaml`
- 预测：`uv run mlproj predict --model-uri <model.joblib> --input <input.csv> --output <pred.csv>`
- 服务：`uv run mlproj serve --model-uri <model.joblib> --host 127.0.0.1 --port 8000`
- 测试：`uv run pytest -q`

## 7. 修改原则

- 小步修改、可回滚、可验证
- 尽量复用现有模块与测试资产
- 避免无关重构与破坏性变更
- 对外行为变更需在 README/配置中可追踪
