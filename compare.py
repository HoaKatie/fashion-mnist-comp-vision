"""Compare the VAE and DCGAN on the same Fashion-MNIST test images.

Load the saved models, generate clothing images, and write a CSV and four PNGs
to comparison/. Run these commands from the folder that contains compare.py.

    python compare.py                  # full comparison with 10,000 images per group
    python compare.py --samples 1000   # quick demo with the bundled checkpoints

Use --data to point to a folder that already contains FashionMNIST/.
"""

import argparse
import csv
from pathlib import Path

import torch

from src.data import CLASS_NAMES, load_split
from src.metrics import class_kl, class_shares, frechet_distance, laplacian_variance
from src.metrics import path_smoothness, precision_recall, slerp
from src.models import load_checkpoint, load_classifier
from src.plots import save_class_balance, save_grids
from src.utils import get_device

ROOT = Path(__file__).resolve().parent
TRAINED = ROOT / "trained_models"
DATASET = "fashion_mnist"
EVAL_SEED = 0
N_PATHS = 100
N_STEPS = 10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # The options control the number of images, where to find inputs, and where to save outputs.
    parser.add_argument("--samples", type=int, default=10000, help="images per group (16 to 10000)")
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="folder containing FashionMNIST/")
    parser.add_argument("--output", type=Path, default=ROOT / "comparison", help="where to save scores and plots")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--vae", type=Path, default=TRAINED / "vae_fashion_mnist_z32_beta1_seed0/checkpoints/epoch_030.pt")
    parser.add_argument("--gan", type=Path, default=TRAINED / "gan_fashion_mnist_z32_seed0/checkpoints/epoch_030.pt")
    parser.add_argument("--classifier", type=Path, default=TRAINED / "classifier_fashion_mnist/classifier.pt")
    args = parser.parse_args()
    # We need 16 samples for the image grids. The test split contains 10,000 images.
    if not 16 <= args.samples <= 10000:
        parser.error("--samples must be between 16 and 10000")
    for path in (args.vae, args.gan, args.classifier):
        if not path.is_file():
            parser.error(f"missing checkpoint {path}")
    return args


def in_batches(fn, inputs, device):
    # Send up to 500 inputs to the device at a time, then join the outputs on the CPU.
    # fn is either the model's decoder or the classifier's feature extractor.
    return torch.cat([fn(batch.to(device)).cpu() for batch in inputs.split(500)])


def compare_images(images, classifier, test_features, test_shares, device):
    # First get the 128 features the classifier sees just before it picks a class.
    features = in_batches(classifier.embed, images, device)
    # The final layer scores the ten classes. Keep the top probability and its label.
    logits = classifier.fc(features.to(device)).cpu()
    confidence, labels = logits.softmax(dim=1).max(dim=1)
    shares = class_shares(labels, len(CLASS_NAMES))
    # Use features for fidelity and coverage, and class shares for the mix of clothes.
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
    # Draw 100 pairs of endpoints and make 10 frames along each path, including both endpoints.
    random = torch.Generator().manual_seed(EVAL_SEED + 1)
    start = torch.randn(N_PATHS, model.latent_dim, generator=random)
    end = torch.randn(N_PATHS, model.latent_dim, generator=random)
    steps = slerp(start, end, torch.linspace(0, 1, N_STEPS))
    # Flatten the paths into one list of codes so the decoder can process a batch at a time.
    images = in_batches(model.decode, steps.flatten(0, 1), device)
    features = in_batches(classifier.embed, images, device)
    # Group features back by path for the score. Show just the first path in the plot.
    return path_smoothness(features.reshape(N_PATHS, N_STEPS, -1)), images[:N_STEPS]


# The comparison only needs forward passes. Inference mode skips gradient tracking.
@torch.inference_mode()
def compare(args):
    # 1. Load the classifier and prepare the shared reference from the real test split.
    device = get_device(args.device)
    classifier, accuracy = load_classifier(args.classifier, device)
    print(f"Comparing {args.samples:,} images per group on {device}. Classifier test accuracy {accuracy:.1%}.", flush=True)

    test_images, test_labels = load_split(DATASET, "test", args.data, count=args.samples)
    test_features = in_batches(classifier.embed, test_images, device)
    # The reference class shares come from the dataset's true labels.
    test_shares = class_shares(test_labels, len(CLASS_NAMES))

    # 2. Score a random sample from the real training split against the same test images.
    # This gives us an example of what the scores look like when both sets are real.
    real_images, _ = load_split(DATASET, "train", args.data, count=args.samples, shuffle_seed=EVAL_SEED)
    scores, _, _ = compare_images(real_images, classifier, test_features, test_shares, device)
    run_info = {"n_samples": args.samples, "eval_seed": EVAL_SEED, "eval_classifier_accuracy": accuracy}
    rows = [{
        "model": "Real train", "seed": None, "epoch": None, "latent_dim": None, "beta": None,
        **run_info, **scores, "interp_step_cv": None,
    }]
    # Collect the scores and plot inputs as we go. These start with the real examples.
    samples = {"Real test": test_images[:16]}
    class_balance = {"Real test": test_shares}
    confidence_examples = {}
    path_examples = {}

    # 3. Load and score each generator. The checkpoint tells us the latent size it was trained with.
    for name, kind, checkpoint in (("VAE", "vae", args.vae), ("DCGAN", "gan", args.gan)):
        print(f"Evaluating {name} ...", flush=True)
        model, cfg, epoch = load_checkpoint(checkpoint, device)
        if cfg["model"] != kind:
            raise SystemExit(f"Expected a {name} checkpoint, got {cfg['model']} at {checkpoint}")

        # Reset the noise seed for each model. Both bundled models use 32 values per vector,
        # so each gets the same vectors in the same order.
        random = torch.Generator().manual_seed(EVAL_SEED)
        z = torch.randn(args.samples, model.latent_dim, generator=random)
        images = in_batches(model.decode, z, device)
        scores, shares, confidence = compare_images(images, classifier, test_features, test_shares, device)
        smoothness, path = interpolate(model, classifier, device)
        rows.append({
            "model": name, "seed": cfg["seed"], "epoch": epoch,
            "latent_dim": model.latent_dim, "beta": cfg.get("beta"),
            **run_info, **scores, "interp_step_cv": smoothness,
        })

        # Show the first random samples and the classifier's highest and lowest confidence ones.
        samples[name] = images[:16]
        class_balance[name] = shares
        order = confidence.argsort()
        confidence_examples[f"{name} — highest confidence"] = images[order[-8:].flip(0)]
        confidence_examples[f"{name} — lowest confidence"] = images[order[:8]]
        path_examples[name] = path

    # 4. Write the score table, then turn the collected images and class shares into PNGs.
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    save_grids(args.output / "samples.png", samples, nrow=4, columns=3, size=(9, 3))
    save_class_balance(args.output / "class_balance.png", CLASS_NAMES, class_balance)
    save_grids(args.output / "successes_and_failures.png", confidence_examples, nrow=4, columns=2, size=(7, 4.3))
    save_grids(args.output / "interpolation.png", path_examples, nrow=N_STEPS, columns=1, size=(9, 2.2))

    # Print the main scores for a quick look. The CSV has the rest of the scores and settings.
    for row in rows:
        print(f"{row['model']:10}  FID-clf {row['fid_classifier']:6.2f}  "
              f"precision {row['precision_classifier']:.3f}  recall {row['recall_classifier']:.3f}  "
              f"sharpness {row['laplacian_var']:.3f}")
    print(f"Saved metrics and plots to {args.output}")


if __name__ == "__main__":
    compare(parse_args())
