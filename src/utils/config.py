# src/utils/config.py

import yaml
from pathlib import Path

def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def load_experiment_config(experiment_name: str) -> dict:
    root = Path("configs")

    exp = _load_yaml(root / "experiment" / f"{experiment_name}.yaml")
    if not exp:
        raise ValueError(f"Missing experiment: {experiment_name}")

    dataset_cfg = _load_yaml(root / "dataset" / f"{exp['dataset']}.yaml")
    model_cfg = _load_yaml(root / "model" / f"{exp['model']}.yaml")

    cfg = {
        "name": exp["name"],
        "model": exp["model"],
        "dataset": exp["dataset"],
        **dataset_cfg,
    }

    cfg = _deep_merge(cfg, exp.get("overrides", {}))
    cfg["model_cfg"] = model_cfg

    cfg["results_dir"] = f"results/{exp['dataset']}/{exp['model']}"

    return cfg