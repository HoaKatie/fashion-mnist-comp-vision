"""Helpers for picking a device, keeping runs repeatable and saving training outputs"""

import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.utils import save_image

def set_seed(seed):
    # Set the seeds so runs can be repeated
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # PyTorch also seeds the GPU generators

def get_device(preferred="auto"):
    # If auto, try CUDA first, then Apple MPS, then CPU
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def run_name(cfg):
    """Use the training settings to name the run folder."""
    if cfg["model"] == "classifier":
        return f"classifier_{cfg['dataset']}"
    # Put the latent size and seed in the name, plus beta for VAE runs
    name = f"{cfg['model']}_{cfg['dataset']}_z{cfg['latent_dim']}"
    if cfg["model"] == "vae":
        name += f"_beta{float(cfg['beta']):g}"
    return f"{name}_seed{cfg['seed']}"

def run_dir_for(cfg):
    return Path(cfg["output_dir"]) / run_name(cfg)

def prepare_run_dir(cfg):
    """Make a clean output folder for this run, replacing an earlier run with the same name"""
    run_dir = run_dir_for(cfg)
    # Clear an earlier run with the same name so old logs don't mix in
    if run_dir.exists():
        print(f"Note: replacing previous results in {run_dir}")
        shutil.rmtree(run_dir)
    (run_dir / "samples").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    # Save the settings with any changes made using --set
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    return run_dir

def fixed_noise(n, latent_dim, seed=1234):
    """Reuse the same noise at every epoch to compare the sample grids."""
    # Use a separate seed so the previews get the same noise every time
    return torch.randn(n, latent_dim, generator=torch.Generator().manual_seed(seed))


def to_unit_range(x):
    """Scale model pixels from [-1, 1] to [0, 1] for saving."""
    return (x.clamp(-1, 1) + 1) / 2


def save_grid(images, path, nrow=8):
    # Move the grid to CPU and put pixels in the range the image writer needs
    save_image(to_unit_range(images.detach().cpu()), path, nrow=nrow, padding=2)


def append_csv(path, row):
    path = Path(path)
    is_new = not path.exists()
    # Write the header once and append a row each time
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _to_builtin(obj):
    # Convert arrays and number types to values JSON understands
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    raise TypeError(f"Cannot serialise {type(obj)}")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)
