"""Build the models and save their weights during training.

compare.py uses this file to load the checkpoints.
"""

import torch

from src.data import dataset_info
from src.models.classifier import EvalClassifier
from src.models.gan import GAN
from src.models.vae import VAE

__all__ = [
    "VAE",
    "GAN",
    "EvalClassifier",
    "build_model",
    "save_checkpoint",
    "load_checkpoint",
    "load_classifier",
]


def build_model(cfg):
    # Use the settings for the input channels and network sizes
    channels = dataset_info(cfg["dataset"])["channels"]
    if cfg["model"] == "vae":
        return VAE(cfg["latent_dim"], channels, cfg["base_channels"])
    if cfg["model"] == "gan":
        return GAN(cfg["latent_dim"], channels, cfg["base_channels"])
    raise ValueError(f"Unknown generative model {cfg['model']!r}")


def save_checkpoint(path, model, cfg, epoch, train_time_sec):
    # state_dict has weights and other values like batch norm stats
    # Save the settings and epoch too so we can rebuild this model later
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "train_time_sec": train_time_sec,
        },
        path,
    )


def load_checkpoint(path, device):
    # Read the checkpoint and load its tensors onto the chosen device
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    # Build the same layers from the saved settings before loading weights
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    # Eval mode uses the saved batch norm stats for new images
    model.eval()
    return model, cfg, ckpt["epoch"]


def load_classifier(path, device):
    # The classifier file also has its accuracy on the test set
    ckpt = torch.load(path, map_location=device, weights_only=True)
    info = dataset_info("fashion_mnist")
    # Build the same one-channel CNN with ten outputs and set it to eval
    clf = EvalClassifier(info["channels"], info["num_classes"]).to(device)
    clf.load_state_dict(ckpt["state_dict"])
    clf.eval()
    return clf, ckpt["test_accuracy"]
