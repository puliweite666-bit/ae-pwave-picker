from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_config_path() -> Path:
    return get_project_root() / "config" / "app_config.json"


def load_app_config() -> Dict[str, Any]:
    config_path = get_config_path()
    if not config_path.exists():
        raise FileNotFoundError(
            f"未找到配置文件: {config_path}\n"
            "请先创建 config/app_config.json。"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    project_root = get_project_root()

    for key in ("default_model_path", "output_dir", "figure_dir", "result_csv"):
        if key in config:
            config[key] = str((project_root / config[key]).resolve())

    return config