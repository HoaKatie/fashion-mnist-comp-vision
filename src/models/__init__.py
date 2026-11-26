"""Build models and handle their checkpoints.

The training scripts use build_model and save_checkpoint for the generators.
compare.py uses load_checkpoint and load_classifier to restore the saved models.
"""

import torch

from src.data import dataset_info
from src.models.classifier import EvalClassifier
from src.models.gan import GAN
from src.models.vae import VAE

__all__ = ["VAE", "GAN", "EvalClassifier", "build_model", "save_checkpoint", "load_checkpoint", "load_classifier"]

def build_model(cfg):
    # Read the image channel count and use the settings to choose the network's size.
    channels = dataset_info(cfg["dataset"])["channels"]
    if cfg["model"] == "vae":
        return VAE(cfg["latent_dim"], channels, cfg["base_channels"])
    if cfg["model"] == "gan":
        return GAN(cfg["latent_dim"], channels, cfg["base_channels"])
    raise ValueError(f"Unknown generative model {cfg['model']!r}")

def save_checkpoint(path, model, cfg, epoch, train_time_sec):
    # state_dict contains the weights and buffers such as batch-normalization statistics.
    # Keep the settings and epoch beside it so the loader can rebuild the same network.
    torch.save(
        {"state_dict": model.state_dict(), "config": cfg, "epoch": epoch, "train_time_sec": train_time_sec},
        path,
    )

def load_checkpoint(path, device):
    # 1. Read the stored settings and move the saved tensors to the chosen device.
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    # 2. Recreate the model with those settings, then fill it with the saved weights.
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    # 3. Use the saved batch-normalization statistics when generating images.
    model.eval()
    return model, cfg, ckpt["epoch"]

def load_classifier(path, device):
    # This checkpoint contains the clothing classifier and its test accuracy.
    ckpt = torch.load(path, map_location=device, weights_only=True)
    info = dataset_info("fashion_mnist")
    # Rebuild the one-channel CNN with ten class outputs and switch it to evaluation mode.
    clf = EvalClassifier(info["channels"], info["num_classes"]).to(device)
    clf.load_state_dict(ckpt["state_dict"])
    clf.eval()
    return clf, ckpt["test_accuracy"]
