"""Save the image grids and class chart. Run python compare.py to make the PNGs."""

import matplotlib

# Use a file-only plotting backend before loading pyplot
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

def save_grids(path, groups, nrow, columns, size):
    # Give each group of images its own spot in the figure
    rows = len(groups) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=size, squeeze=False, layout="constrained"
    )
    for ax, (name, images) in zip(axes.flat, groups.items()):
        # Change [-1, 1] pixels to [0, 1] so they display properly
        images = ((images + 1) / 2).clamp(0, 1)
        grid = make_grid(images, nrow=nrow, pad_value=1)
        # Show the single grayscale channel without smoothing the pixels
        ax.imshow(grid[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
    figure.savefig(path, dpi=160)
    plt.close(figure)

def save_class_balance(path, classes, groups):
    figure, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    positions = range(len(classes))
    width = 0.26
    # Put the bars side by side for each clothing class and show percentages
    for offset, (name, shares) in zip((-width, 0, width), groups.items()):
        ax.bar([x + offset for x in positions], shares * 100, width=width, label=name)
    ax.set_xticks(positions, classes, rotation=30, ha="right")
    ax.set_ylabel("Images in each clothing class (%)")
    ax.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)
