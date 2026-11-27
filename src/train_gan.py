"""Train the DCGAN on Fashion-MNIST. Update the discriminator first, then the generator for each batch.

Run from the project folder:
    python -m src.train_gan

Uses configs/gan.yaml and saves losses, pictures and checkpoints under runs/.
The saved DCGAN in trained_models/ is ready for compare.py.
"""

import argparse
import time

import torch

from src.config import load_config
from src.data import get_loader
from src.losses import discriminator_loss, generator_loss
from src.models.build import build_model, save_checkpoint
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
    parser = argparse.ArgumentParser(description="Train a DCGAN")
    # Pick the YAML with --config and change individual values with --set
    parser.add_argument("--config", default="configs/gan.yaml")
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

    # Each part of the GAN gets its own Adam optimizer
    model = build_model(cfg).to(device)
    model.apply(init_dcgan_weights)
    G, D = model.generator, model.discriminator
    betas = (cfg["adam_beta1"], 0.999)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["lr"], betas=betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["lr"], betas=betas)

    # Make the folder for weights, sample images and logs
    run_dir = prepare_run_dir(cfg)
    # We use the same noise for each preview so we can see what training changed
    z_fixed = fixed_noise(64, cfg["latent_dim"]).to(device)
    print(
        f"[gan] {run_dir.name} | params={count_parameters(model):,} | device={device}"
    )

    start = time.time()
    # Update D and then G on each training batch
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        sums = torch.zeros(4, device=device)
        for x, _ in loader:
            x = x.to(device)
            # One noise vector per real image so the fake batch is the same size
            z = torch.randn(x.size(0), cfg["latent_dim"], device=device)
            fake = G(z)

            # Train D to give real images a 1 and fakes a 0
            # Detach the fakes so this update doesn't train G
            real_logits = D(x)
            fake_logits = D(fake.detach())
            loss_d = discriminator_loss(real_logits, fake_logits)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # Run the fakes through the updated D and keep the gradient path back to G
            # Train G to get a real score on the fake images
            loss_g = generator_loss(D(fake))
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            # Turn D's earlier scores into probabilities for the log
            sums += torch.stack(
                [
                    loss_d.detach(),
                    loss_g.detach(),
                    torch.sigmoid(real_logits).mean().detach(),
                    torch.sigmoid(fake_logits).mean().detach(),
                ]
            )

        # Average the batch values into one log row per epoch
        loss_d, loss_g, d_real, d_fake = (sums / len(loader)).tolist()
        elapsed = time.time() - start
        append_csv(
            run_dir / "train_log.csv",
            {
                "epoch": epoch,
                "loss_d": loss_d,
                "loss_g": loss_g,
                "d_real": d_real,
                "d_fake": d_fake,
                "elapsed_sec": round(elapsed, 1),
            },
        )
        print(
            f"[gan] epoch {epoch:3d}/{cfg['epochs']}  loss_d={loss_d:.3f}  loss_g={loss_g:.3f}  "
            f"D(x)={d_real:.2f}  D(G(z))={d_fake:.2f}  ({elapsed:.0f}s)"
        )

        # Use the fixed noise and eval mode for the preview grid
        model.eval()
        with torch.no_grad():
            save_grid(
                model.decode(z_fixed), run_dir / "samples" / f"epoch_{epoch:03d}.png"
            )
        # Save G and D at the chosen interval and at the last epoch
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
    # Load the shared and GAN settings, then apply any --set values
    cfg = load_config(args.config, args.set)
    if cfg["model"] != "gan":
        raise SystemExit(f"{args.config} is a '{cfg['model']}' config, expected 'gan'")
    train(cfg)

if __name__ == "__main__":
    main()
