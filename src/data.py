"""Load Fashion-MNIST.

Training uses batches from get_loader() and compare.py reads a split with load_split()
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGE_SIZE = 32
CLASS_NAMES = datasets.FashionMNIST.classes

# Add a black border to get 32 by 32 without resizing the clothes
# Scale pixels to [-1, 1] to match what the generator outputs
TRANSFORM = transforms.Compose(
    [
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def dataset_info(name):
    # All three models take one grayscale channel at the same size
    if name != "fashion_mnist":
        raise ValueError(f"Unknown dataset {name!r}")
    return {"channels": 1, "num_classes": len(CLASS_NAMES), "image_size": IMAGE_SIZE}

def get_dataset(name, split, root="data"):
    dataset_info(name)
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    # Look under root and download the data if it's missing
    return datasets.FashionMNIST(
        root, train=split == "train", download=True, transform=TRANSFORM
    )

def get_loader(
    name,
    split,
    batch_size,
    num_workers=2,
    root="data",
    subset=None,
    seed=0,
    shuffle=None,
):
    dataset = get_dataset(name, split, root)
    # Use a fixed seed so a smaller run can pick the same subset again
    if subset:
        order = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(seed)
        )
        dataset = Subset(dataset, order[:subset].tolist())
    if shuffle is None:
        shuffle = split == "train"
    # When shuffling, skip the short last batch so training batches stay the same size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

def load_split(name, split, root="data", count=None, shuffle_seed=None):
    """Load one split into image and label tensors. Use count to take fewer images."""
    dataset = get_dataset(name, split, root)
    # Keep the dataset order or pick a fixed random sample if given a seed
    if shuffle_seed is None:
        order = slice(count)
    else:
        random = torch.Generator().manual_seed(shuffle_seed)
        order = torch.randperm(len(dataset), generator=random)[:count]
    # Add the grayscale channel and pad and scale the same way as training
    images = dataset.data[order].unsqueeze(1).float() / 255.0
    images = F.pad(images, (2, 2, 2, 2))
    # Use the same indices so each image keeps its label
    return images * 2 - 1, dataset.targets[order]
