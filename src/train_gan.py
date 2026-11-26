"""Train the Fashion-MNIST deep convolutional GAN (DCGAN).

For each batch, train the discriminator to spot fake clothes and then train the
generator to make more convincing clothes. Run from the project folder.

    python -m src.train_gan

The defaults come from configs/gan.yaml. The run saves losses, sample images,
and checkpoints under runs/. compare.py can use the DCGAN already bundled.
"""

import argparse
import time
import torch
from src.config import load_config
from src.data import get_loader
from src.losses import discriminator_loss, generator_loss
from src.models import build_model, save_checkpoint
from src.models.networks import count_parameters, init_dcgan_weights
from src.utils import append_csv, fixed_noise, get_device, prepare_run_dir, save_grid, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train a DCGAN")
    # --config selects a YAML file. --set can change values for this one run.
    parser.add_argument("--config", default="configs/gan.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    return parser.parse_args()

def train(cfg):
    # 1. Set the random seed and load batches of real clothing images.
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    loader = get_loader(cfg["dataset"], "train", cfg["batch_size"], cfg["num_workers"],
                        cfg["data_root"], cfg["subset"], seed=cfg["seed"])

    # 2. Set up the generator and discriminator with a separate Adam optimizer for each.
    model = build_model(cfg).to(device)
    model.apply(init_dcgan_weights)
    G, D = model.generator, model.discriminator
    betas = (cfg["adam_beta1"], 0.999)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["lr"], betas=betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["lr"], betas=betas)

    # 3. Create the run folder before saving training outputs.
    run_dir = prepare_run_dir(cfg)
    # Keep this noise the same for the picture we save after each epoch.
    z_fixed = fixed_noise(64, cfg["latent_dim"]).to(device)
    print(f"[gan] {run_dir.name} | params={count_parameters(model):,} | device={device}")

    start = time.time()
    # 4. During each epoch, alternate both updates for every batch.
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        sums = torch.zeros(4, device=device)
        for x, _ in loader:
            x = x.to(device)
            # Draw one noise vector per real image and use G to make a matching batch of fakes.
            z = torch.randn(x.size(0), cfg["latent_dim"], device=device)
            fake = G(z)

            # Train D to score real images as 1 and fake images as 0.
            # detach() keeps this loss from sending gradients back to G.
            real_logits = D(x)
            fake_logits = D(fake.detach())
            loss_d = discriminator_loss(real_logits, fake_logits)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # Score the same fakes with the updated D. This time keep the connection to G.
            # The generator loss treats these fakes as real targets, so G learns to fool D.
            loss_g = generator_loss(D(fake))
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            # Convert the earlier D scores to probabilities and keep them for the log.
            sums += torch.stack([loss_d.detach(), loss_g.detach(),
                                 torch.sigmoid(real_logits).mean().detach(),
                                 torch.sigmoid(fake_logits).mean().detach()])

        # 5. Average the batch values to get one row in the training log for this epoch.
        loss_d, loss_g, d_real, d_fake = (sums / len(loader)).tolist()
        elapsed = time.time() - start
        append_csv(run_dir / "train_log.csv",
                   {"epoch": epoch, "loss_d": loss_d, "loss_g": loss_g, "d_real": d_real, "d_fake": d_fake,
                    "elapsed_sec": round(elapsed, 1)})
        print(f"[gan] epoch {epoch:3d}/{cfg['epochs']}  loss_d={loss_d:.3f}  loss_g={loss_g:.3f}  "
              f"D(x)={d_real:.2f}  D(G(z))={d_fake:.2f}  ({elapsed:.0f}s)")

        # 6. Generate a progress picture with fixed noise and the saved batch statistics.
        model.eval()
        with torch.no_grad():
            save_grid(model.decode(z_fixed), run_dir / "samples" / f"epoch_{epoch:03d}.png")
        # Save both parts of the GAN at each configured interval and at the final epoch.
        if epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
            save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, cfg, epoch, elapsed)
    return run_dir

def main():
    args = parse_args()
    # Combine the shared settings, GAN settings, and any command-line overrides.
    cfg = load_config(args.config, args.set)
    if cfg["model"] != "gan":
        raise SystemExit(f"{args.config} is a '{cfg['model']}' config, expected 'gan'")
    train(cfg)

if __name__ == "__main__":
    main()
