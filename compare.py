"""Compare clothes made by the saved VAE and DCGAN. Save the scores and pictures.

From the project folder:
    python compare.py                 # full CPU run, saves in results/
    python compare.py --samples 1000  # quick run, saves in results/custom-compare/

Use --output results/my-comparison to save a run in its own folder.
Use --data to reuse a folder that already has FashionMNIST/.
"""

import argparse
import csv
from pathlib import Path

import torch
from src.data import CLASS_NAMES, load_split
from src.metrics import (
    class_kl,
    class_shares,
    frechet_distance,
    laplacian_variance,
    path_smoothness,
    precision_recall,
    slerp,
)
from src.models import load_checkpoint, load_classifier
from src.plots import save_class_balance, save_grids
from src.utils import get_device

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TRAINED = ROOT / "trained_models"
DEFAULT_SAMPLES = 10000
DEFAULT_DEVICE = "cpu"
DEFAULT_CHECKPOINTS = {
    "vae": TRAINED / "vae_fashion_mnist_z32_beta1_seed0/checkpoints/epoch_030.pt",
    "gan": TRAINED / "gan_fashion_mnist_z32_seed0/checkpoints/epoch_030.pt",
    "classifier": TRAINED / "classifier_fashion_mnist/classifier.pt",
}
DATASET = "fashion_mnist"
EVAL_SEED = 0
N_PATHS = 100
N_STEPS = 10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # Pick how many images to compare, where to find them and where to save the scores
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="images per group (16 to 10000)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data",
        help="folder containing FashionMNIST/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output folder under results/, relative to the project folder",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=("auto", "cpu", "cuda", "mps"),
        help="default: cpu",
    )
    for name, checkpoint in DEFAULT_CHECKPOINTS.items():
        parser.add_argument(
            f"--{name}",
            type=Path,
            default=checkpoint,
            help=f"path to the {name} checkpoint",
        )
    args = parser.parse_args()
    # Need at least 16 for the grids and the test set only has 10,000
    if not 16 <= args.samples <= DEFAULT_SAMPLES:
        parser.error("--samples must be between 16 and 10000")
    for path in (args.vae, args.gan, args.classifier):
        if not path.is_file():
            parser.error(f"missing checkpoint {path}")

    # The full CPU run with the saved models goes straight in results/
    # I keep quick runs and other settings in a separate folder
    custom = (
        args.samples != DEFAULT_SAMPLES
        or args.device != DEFAULT_DEVICE
        or any(
            getattr(args, name).resolve() != path.resolve()
            for name, path in DEFAULT_CHECKPOINTS.items()
        )
    )
    if args.output is None:
        args.output = RESULTS / "custom-compare" if custom else RESULTS

    # Read relative paths from the project folder and keep outputs under results/
    args.output = (ROOT / args.output).resolve()
    results = RESULTS.resolve()
    if args.output != results and results not in args.output.parents:
        parser.error(
            "--output must be inside results/, for example results/custom-compare"
        )
    if custom and args.output == results:
        parser.error("choose a subfolder of results/ for a custom comparison")
    return args


def in_batches(fn, inputs, device):
    # Send up to 500 at a time so the whole set doesn't have to fit on the device
    # fn can make images from noise or pull features from images
    return torch.cat([fn(batch.to(device)).cpu() for batch in inputs.split(500)])


def compare_images(images, classifier, test_features, test_shares, device):
    # Get the 128 features before the classifier picks a label
    features = in_batches(classifier.embed, images, device)
    # Keep the most likely class and its probability
    logits = classifier.fc(features.to(device)).cpu()
    confidence, labels = logits.softmax(dim=1).max(dim=1)
    shares = class_shares(labels, len(CLASS_NAMES))
    # Features tell us how real and varied the images are, labels give us the class mix
    precision, recall = precision_recall(test_features, features)
    scores = {
        "fid_classifier": frechet_distance(test_features, features),
        "precision_classifier": precision,
        "recall_classifier": recall,
        "laplacian_var": laplacian_variance(images).mean().item(),
        "class_kl": class_kl(shares, test_shares),
    }
    return scores, shares, confidence


def interpolate(model, classifier, device):
    # Pick 100 pairs of noise vectors and make 10 frames per path
    random = torch.Generator().manual_seed(EVAL_SEED + 1)
    start = torch.randn(N_PATHS, model.latent_dim, generator=random)
    end = torch.randn(N_PATHS, model.latent_dim, generator=random)
    steps = slerp(start, end, torch.linspace(0, 1, N_STEPS))
    # Put all the codes in one list so we can decode them in batches
    images = in_batches(model.decode, steps.flatten(0, 1), device)
    features = in_batches(classifier.embed, images, device)
    # Put the features back into paths for the score and plot the first path
    return path_smoothness(features.reshape(N_PATHS, N_STEPS, -1)), images[:N_STEPS]


# Only running the models here, so no need to track gradients
@torch.inference_mode()
def compare(args):
    # Load the classifier and use the same real test set for both models
    device = get_device(args.device)
    classifier, accuracy = load_classifier(args.classifier, device)
    print(
        f"Comparing {args.samples:,} images per group on {device}. Classifier test accuracy {accuracy:.1%}.",
        flush=True,
    )

    test_images, test_labels = load_split(
        DATASET, "test", args.data, count=args.samples
    )
    test_features = in_batches(classifier.embed, test_images, device)
    # Get the real class mix from the true labels
    test_shares = class_shares(test_labels, len(CLASS_NAMES))

    # Pick random training images to score against the test set too
    # This gives me a baseline for how two groups of real images score
    real_images, _ = load_split(
        DATASET, "train", args.data, count=args.samples, shuffle_seed=EVAL_SEED
    )
    scores, _, _ = compare_images(
        real_images, classifier, test_features, test_shares, device
    )
    run_info = {
        "n_samples": args.samples,
        "eval_seed": EVAL_SEED,
        "eval_classifier_accuracy": accuracy,
    }
    rows = [
        {
            "model": "Real train",
            "seed": None,
            "epoch": None,
            "latent_dim": None,
            "beta": None,
            **run_info,
            **scores,
            "interp_step_cv": None,
        }
    ]
    # Start the score table and figures with the real images
    samples = {"Real test": test_images[:16]}
    class_balance = {"Real test": test_shares}
    confidence_examples = {}
    path_examples = {}

    # Read each model's latent size from its checkpoint
    for name, kind, checkpoint in (
        ("VAE", "vae", args.vae),
        ("DCGAN", "gan", args.gan),
    ):
        print(f"Evaluating {name} ...", flush=True)
        model, cfg, epoch = load_checkpoint(checkpoint, device)
        if cfg["model"] != kind:
            raise SystemExit(
                f"Expected a {name} checkpoint, got {cfg['model']} at {checkpoint}"
            )

        # Reset the seed so both bundled models get the same noise
        # Both use 32 numbers per code so the inputs match
        random = torch.Generator().manual_seed(EVAL_SEED)
        z = torch.randn(args.samples, model.latent_dim, generator=random)
        images = in_batches(model.decode, z, device)
        scores, shares, confidence = compare_images(
            images, classifier, test_features, test_shares, device
        )
        smoothness, path = interpolate(model, classifier, device)
        rows.append(
            {
                "model": name,
                "seed": cfg["seed"],
                "epoch": epoch,
                "latent_dim": model.latent_dim,
                "beta": cfg.get("beta"),
                **run_info,
                **scores,
                "interp_step_cv": smoothness,
            }
        )

        # Show the first samples and the ones the classifier is most and least sure about
        samples[name] = images[:16]
        class_balance[name] = shares
        order = confidence.argsort()
        confidence_examples[f"{name} — highest confidence"] = images[order[-8:].flip(0)]
        confidence_examples[f"{name} — lowest confidence"] = images[order[:8]]
        path_examples[name] = path

    # Save the scores to CSV and make the PNGs from the samples we kept
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    save_grids(args.output / "samples.png", samples, nrow=4, columns=3, size=(9, 3))
    save_class_balance(args.output / "class_balance.png", CLASS_NAMES, class_balance)
    save_grids(
        args.output / "successes_and_failures.png",
        confidence_examples,
        nrow=4,
        columns=2,
        size=(7, 4.3),
    )
    save_grids(
        args.output / "interpolation.png",
        path_examples,
        nrow=N_STEPS,
        columns=1,
        size=(9, 2.2),
    )

    # Print a few scores here and leave the full table and settings in the CSV
    for row in rows:
        print(
            f"{row['model']:10}  FID-clf {row['fid_classifier']:6.2f}  "
            f"precision {row['precision_classifier']:.3f}  recall {row['recall_classifier']:.3f}  "
            f"sharpness {row['laplacian_var']:.3f}"
        )
    print(f"Saved metrics and plots to {args.output}")


if __name__ == "__main__":
    compare(parse_args())
