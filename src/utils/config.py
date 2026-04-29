from pathlib import Path
import yaml

def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _deep_merge(base: dict, override: dict) -> dict:
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
        raise ValueError(f"Missing experiment config: {experiment_name}")

    # 1. Start from training defaults
    cfg = _load_yaml(root / "base" / "training.yaml")

    # 2. Merge dataset config
    cfg = _deep_merge(cfg, _load_yaml(root / "dataset" / f"{exp['dataset']}.yaml"))

    # 3. Merge model config
    cfg = _deep_merge(cfg, _load_yaml(root / "model" / f"{exp['model']}.yaml"))

    # 4. Apply experiment-level overrides
    cfg = _deep_merge(cfg, exp.get("overrides", {}))

    # 5. Attach metadata
    cfg["name"] = exp["name"]
    cfg["_model_type"] = exp["model"]
    cfg["_dataset_type"] = exp["dataset"]
    cfg.setdefault("checkpointing", {})
    cfg["results_dir"] = f"results/{exp['dataset']}/{exp['name']}"

    return cfg