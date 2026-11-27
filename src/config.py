"""Read YAML settings for training. Start with the shared file and add any changes for this run"""

from pathlib import Path
import yaml

def _parse_value(raw):
    # Turn command line text into the number or other value it means
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        # Treat scientific notation like 1e-4 as a float too
        try:
            return float(value)
        except ValueError:
            return value
    return value

def load_config(path, overrides=None):
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    # Look for the base YAML next to this settings file
    base = cfg.pop("base", None)
    if base is not None:
        merged = load_config(path.parent / base)
        # Use the model's value if both files have the same key
        merged.update(cfg)
        cfg = merged

    # Apply --set last and catch typos in setting names
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        if key not in cfg:
            raise KeyError(f"Unknown config key {key!r}. Known keys: {sorted(cfg)}")
        cfg[key] = _parse_value(raw)
    return cfg
