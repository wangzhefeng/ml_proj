from __future__ import annotations

from pathlib import Path

from mlproj.data.legacy_provider import get_params


def run_data_provider_legacy_demo(script_path: str) -> dict[str, object]:
    name = Path(script_path.replace("\\", "/").lower()).name

    if name in {"config_loader.py", "yaml_loader.py"}:
        cfg = get_params("config.yaml")
        return {"script": script_path, "method": "load_yaml", "top_keys": list(cfg.keys())[:5]}

    if name == "json_loader.py":
        return {
            "script": script_path,
            "method": "load_json",
            "note": "json demo config file not found by default; loader is migrated and available",
        }

    if name == "data_loader_lgb.py":
        return {
            "script": script_path,
            "method": "lgb_loader",
            "note": "requires LightGBM and input TSV paths",
        }

    if name == "data_loader_xgb.py":
        return {
            "script": script_path,
            "method": "xgb_loader",
            "note": "requires XGBoost and input TSV paths",
        }

    return {"script": script_path, "method": "data_provider_bridge"}
