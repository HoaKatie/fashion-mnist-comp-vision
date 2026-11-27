"""Train the CNN that gives us clothing labels and features for the comparison.

Run from the project folder:
    python -m src.train_classifier

Uses configs/classifier.yaml and saves classifier.pt under runs/.
For compare.py we use the classifier already in trained_models/.
"""

import argparse

import torch
import torch.nn.functional as F

from src.config import load_config
from src.data import dataset_info, get_loader, load_split
from src.models.classifier import EvalClassifier
from src.utils import get_device, run_dir_for, save_json, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train the evaluation classifier")
    # Pick the settings file and use --set for one-off changes
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override config values",
    )
    return parser.parse_args()

@torch.no_grad()
def accuracy(model, images, labels, device, batch=1000):
    # Eval mode turns off dropout and uses the saved batch norm stats
    model.eval()
    correct = 0
    for x, y in zip(images.split(batch), labels.split(batch)):
        # Pick the highest-scoring class and count how many labels match
        correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    return correct / len(labels)

def main():
    # Load the settings and set the seed so the run can be repeated
    args = parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    info = dataset_info(cfg["dataset"])

    # Train on the train split and check accuracy on test images the classifier didn't train on
    loader = get_loader(
        cfg["dataset"],
        "train",
        cfg["batch_size"],
        cfg["num_workers"],
        cfg["data_root"],
        cfg["subset"],
        seed=cfg["seed"],
    )
    test_x, test_y = load_split(cfg["dataset"], "test", cfg["data_root"])

    # Make the CNN and give Adam its weights to update
    model = EvalClassifier(info["channels"], info["num_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Train on each batch, then check accuracy once per epoch
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        # Use the raw class scores and the true labels for cross-entropy
        for x, y in loader:
            loss = F.cross_entropy(model(x.to(device)), y.to(device))
            # Clear the last gradients so Adam only uses this batch's update
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        acc = accuracy(model, test_x, test_y, device)
        print(f"[classifier] epoch {epoch}/{cfg['epochs']}  test accuracy = {acc:.4f}")

    # Save the final weights and test accuracy for compare.py
    out_dir = run_dir_for(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "test_accuracy": acc},
        out_dir / "classifier.pt",
    )
    save_json(
        {"dataset": cfg["dataset"], "epochs": cfg["epochs"], "test_accuracy": acc},
        out_dir / "metrics.json",
    )
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()
