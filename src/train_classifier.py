"""Train the clothing classifier used to evaluate the two generators.

The classifier learns the ten class labels from real Fashion-MNIST images.
Its predictions and 128 image features are used by compare.py. Run from the
project folder.

    python -m src.train_classifier

The defaults come from configs/classifier.yaml. This saves classifier.pt
under runs/. compare.py can use the classifier already bundled.
"""

import argparse

import torch
import torch.nn.functional as F

from src.config import load_config
from src.data import dataset_info, get_loader, load_split
from src.models import EvalClassifier
from src.utils import get_device, run_dir_for, save_json, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train the evaluation classifier")
    # Read the classifier's settings and any key=value overrides for this run.
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    return parser.parse_args()

@torch.no_grad()
def accuracy(model, images, labels, device, batch=1000):
    # Evaluation mode turns off dropout and uses the learned batch statistics.
    model.eval()
    correct = 0
    for x, y in zip(images.split(batch), labels.split(batch)):
        # Pick the class with the largest score for each image and count matches.
        correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    return correct / len(labels)

def main():
    # 1. Read the settings, set the seed, and choose where to run the calculations.
    args = parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    info = dataset_info(cfg["dataset"])

    # 2. Learn from the training split and keep the test split for checking accuracy.
    loader = get_loader(cfg["dataset"], "train", cfg["batch_size"], cfg["num_workers"],
                        cfg["data_root"], cfg["subset"], seed=cfg["seed"])
    test_x, test_y = load_split(cfg["dataset"], "test", cfg["data_root"])

    # 3. Build the CNN and the optimizer that will update its weights.
    model = EvalClassifier(info["channels"], info["num_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # 4. Update the classifier on each batch, then check test accuracy after the epoch.
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        # Cross-entropy compares the ten class scores with the true class of each image.
        for x, y in loader:
            loss = F.cross_entropy(model(x.to(device)), y.to(device))
            # Clear the last batch's gradients, compute new ones, and take an Adam step.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        acc = accuracy(model, test_x, test_y, device)
        print(f"[classifier] epoch {epoch}/{cfg['epochs']}  test accuracy = {acc:.4f}")

    # 5. Save the final epoch's weights and test accuracy for the comparison.
    out_dir = run_dir_for(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "test_accuracy": acc}, out_dir / "classifier.pt")
    save_json({"dataset": cfg["dataset"], "epochs": cfg["epochs"], "test_accuracy": acc}, out_dir / "metrics.json")
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()
