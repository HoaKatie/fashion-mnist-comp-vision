"""Read and combine the YAML training settings.

The training script loads its model's YAML file from configs/. This module
starts with the shared values and applies the model settings and --set options.
"""

from pathlib import Path
import yaml

def _parse_value(raw):
    # Convert text from the command line into the number, string, or null it represents.
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        # YAML reads "1e-4" as a string and we want it as a float
        try:
            return float(value)
        except ValueError:
            return value
    return value

def load_config(path, overrides=None):
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    # An optional base file is relative to this YAML file, usually configs/shared.yaml.
    base = cfg.pop("base", None)
    if base is not None:
        merged = load_config(path.parent / base)
        # Model values win when the same setting appears in both files.
        merged.update(cfg)
        cfg = merged

    # Apply command-line changes last and check that each key names an existing setting.
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        if key not in cfg:
            raise KeyError(f"Unknown config key {key!r}. Known keys: {sorted(cfg)}")
        cfg[key] = _parse_value(raw)
    return cfg
