"""Train the VAE on Fashion-MNIST. Encode each image, sample a code and try to rebuild the image.

Run from the project folder:
    python -m src.train_vae

Uses configs/vae.yaml and saves losses, pictures and checkpoints under runs/.
The saved VAE in trained_models/ is ready for compare.py.
"""

import argparse
import time

import torch

from src.config import load_config
from src.data import get_loader
from src.losses import vae_loss
from src.models import build_model, save_checkpoint
from src.models.networks import count_parameters, init_dcgan_weights
from src.utils import (
    append_csv,
    fixed_noise,
    get_device,
    prepare_run_dir,
    save_grid,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE")
    # Pick the YAML with --config and change individual values with --set
    parser.add_argument("--config", default="configs/vae.yaml")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override config values",
    )
    return parser.parse_args()


def train(cfg):
    # Set the seed and get batches of real clothes
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    loader = get_loader(
        cfg["dataset"],
        "train",
        cfg["batch_size"],
        cfg["num_workers"],
        cfg["data_root"],
        cfg["subset"],
        seed=cfg["seed"],
    )

    # Build the encoder and decoder and use one Adam optimizer for both
    model = build_model(cfg).to(device)
    model.apply(init_dcgan_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Make the folder for weights, sample images and logs
    run_dir = prepare_run_dir(cfg)
    # I keep the preview noise fixed so the grids show what training changed
    z_fixed = fixed_noise(64, cfg["latent_dim"]).to(device)
    print(
        f"[vae] {run_dir.name} | params={count_parameters(model):,} | device={device}"
    )

    start = time.time()
    # Each epoch is one pass through the training batches
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        sums = torch.zeros(3, device=device)
        # Use each image as its own target so we don't need class labels
        for x, _ in loader:
            x = x.to(device)
            # Encode the image, sample a code and decode it
            x_recon, mu, logvar = model(x)
            # Add the pixel error and the KL term scaled by beta
            loss, recon, kl = vae_loss(x, x_recon, mu, logvar, beta=cfg["beta"])

            # Clear the old gradients and update both parts using this batch
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # Keep just the loss numbers so logging doesn't hold onto the gradient graph
            sums += torch.stack([loss.detach(), recon.detach(), kl.detach()])

        # Log the average losses and the time since training started
        loss, recon, kl = (sums / len(loader)).tolist()
        elapsed = time.time() - start
        append_csv(
            run_dir / "train_log.csv",
            {
                "epoch": epoch,
                "loss": loss,
                "recon": recon,
                "kl": kl,
                "elapsed_sec": round(elapsed, 1),
            },
        )
        print(
            f"[vae] epoch {epoch:3d}/{cfg['epochs']}  loss={loss:.3f}  recon={recon:.3f}  "
            f"kl={kl:.3f}  ({elapsed:.0f}s)"
        )

        # Use the fixed noise and eval mode for the preview grid
        model.eval()
        with torch.no_grad():
            save_grid(
                model.decode(z_fixed), run_dir / "samples" / f"epoch_{epoch:03d}.png"
            )
        # Save weights at the chosen interval and at the last epoch
        if epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
            save_checkpoint(
                run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                model,
                cfg,
                epoch,
                elapsed,
            )
    return run_dir


def main():
    args = parse_args()
    # Load the shared and VAE settings, then apply any --set values
    cfg = load_config(args.config, args.set)
    if cfg["model"] != "vae":
        raise SystemExit(f"{args.config} is a '{cfg['model']}' config, expected 'vae'")
    train(cfg)


if __name__ == "__main__":
    main()
