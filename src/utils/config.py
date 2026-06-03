"""Config loader with defaults and validation."""
import yaml
from pathlib import Path


_DEFAULTS = {
    "data": {
        "dataset": "ml-100k",
        "data_dir": "data/",
        "models_dir": "models/",
        "output_dir": "outputs/",
        "test_ratio": 0.2,
        "random_seed": 42,
    },
    "ncf": {
        "embedding_dim": 64,
        "hidden_layers": [256, 128, 64],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 30,
        "early_stopping_patience": 5,
        "early_stopping_delta": 0.001,
    },
    "evaluation": {"k_values": [5, 10, 20], "relevance_threshold": 4.0},
    "api": {"host": "0.0.0.0", "port": 8000, "top_n_default": 10},
    "logging": {"level": "INFO", "log_dir": "logs/", "max_bytes": 10485760, "backup_count": 5},
}


def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return _DEFAULTS

    with open(p) as f:
        user_cfg = yaml.safe_load(f) or {}

    # Deep merge user over defaults
    cfg = _deep_merge(_DEFAULTS, user_cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
