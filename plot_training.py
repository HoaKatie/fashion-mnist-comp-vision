"""Plot how the saved VAE and DCGAN runs changed over training.

Read the archived seed-0 training logs and checkpoint scores in results/training/.
Run python plot_training.py from the project folder to write results/training_curves.png.
"""

import csv
from pathlib import Path

import matplotlib

# Save a PNG without opening a plot window.
matplotlib.use("Agg")
from matplotlib import pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"
ARCHIVE = RESULTS / "training"


def read_rows(filename):
    with (ARCHIVE / filename).open(newline="") as file:
        return list(csv.DictReader(file))


def plot_line(axis, rows, column, label, **style):
    # The CSV stores text. Convert the epoch and metric values before plotting.
    epochs = [int(row["epoch"]) for row in rows]
    values = [float(row[column]) for row in rows]
    axis.plot(epochs, values, label=label, **style)


def main():
    vae_log = read_rows("vae_train_log.csv")
    gan_log = read_rows("gan_train_log.csv")
    scores = read_rows("checkpoint_metrics.csv")

    figure, ((vae_loss, gan_loss), (distance, recall)) = plt.subplots(
        2, 2, figsize=(8, 5.5), layout="constrained",
    )

    # Each model gets its own loss axis because it is trained with a different objective.
    plot_line(vae_loss, vae_log, "loss", "Total")
    plot_line(vae_loss, vae_log, "recon", "Reconstruction", linestyle="--")
    plot_line(vae_loss, vae_log, "kl", "KL penalty", linestyle=":")
    vae_loss.set(title="VAE training loss", ylabel="Loss per image", ylim=(0, None))

    plot_line(gan_loss, gan_log, "loss_g", "Generator")
    plot_line(gan_loss, gan_log, "loss_d", "Discriminator", linestyle="--")
    gan_loss.set(title="DCGAN training losses", ylabel="Binary cross-entropy", ylim=(0, None))

    # Use the same evaluation metrics for both models at every saved checkpoint.
    for model, label, color, marker in (
        ("vae", "VAE", "tab:orange", "o"),
        ("gan", "DCGAN", "tab:green", "s"),
    ):
        checkpoints = [row for row in scores if row["model"] == model]
        plot_line(distance, checkpoints, "fid_classifier", label, color=color, marker=marker)
        plot_line(recall, checkpoints, "recall_classifier", label, color=color, marker=marker)
    distance.set(title="Fréchet distance (lower is better)", ylabel="FID-clf", ylim=(0, None))
    recall.set(title="Feature recall (higher is better)", ylabel="Recall", ylim=(0, 1))

    for axis in (vae_loss, gan_loss, distance, recall):
        axis.set(xlabel="Epoch", xticks=range(5, 31, 5), xlim=(0, 31))
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)

    output = RESULTS / "training_curves.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
