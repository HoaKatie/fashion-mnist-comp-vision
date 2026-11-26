"""Load Fashion-MNIST and prepare its pixels for the models.

The training scripts use get_loader() to read mini-batches. compare.py and
the classifier's accuracy check use load_split() to read images into tensors.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGE_SIZE = 32
CLASS_NAMES = datasets.FashionMNIST.classes

# Pad each 28 by 28 image with two black pixels per side to keep the original edges.
# ToTensor scales pixels to [0, 1]. Normalize maps them to [-1, 1] for the models.
TRANSFORM = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def dataset_info(name):
    # All three models use the same image size and number of grayscale channels.
    if name != "fashion_mnist":
        raise ValueError(f"Unknown dataset {name!r}")
    return {"channels": 1, "num_classes": len(CLASS_NAMES), "image_size": IMAGE_SIZE}

def get_dataset(name, split, root="data"):
    dataset_info(name)
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    # Torchvision reads the data from root and downloads it the first time if needed.
    return datasets.FashionMNIST(root, train=split == "train", download=True, transform=TRANSFORM)

def get_loader(name, split, batch_size, num_workers=2, root="data", subset=None, seed=0, shuffle=None):
    dataset = get_dataset(name, split, root)
    # A subset uses a seeded random order so short experiments can reuse the same images.
    if subset:
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
        dataset = Subset(dataset, order[:subset].tolist())
    if shuffle is None:
        shuffle = split == "train"
    # Shuffled batches skip the final partial batch. Workers prepare the images in parallel.
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
    """Read images and labels into tensors. The optional count is for smaller comparisons."""
    dataset = get_dataset(name, split, root)
    # Keep the original image order or shuffle with a fixed seed before taking count images.
    if shuffle_seed is None:
        order = slice(count)
    else:
        random = torch.Generator().manual_seed(shuffle_seed)
        order = torch.randperm(len(dataset), generator=random)[:count]
    # Add the grayscale channel for the model, then do the same padding and scaling as above.
    images = dataset.data[order].unsqueeze(1).float() / 255.0
    images = F.pad(images, (2, 2, 2, 2))
    # Use the same indices for images and labels so they stay paired.
    return images * 2 - 1, dataset.targets[order]
