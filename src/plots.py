"""Save the image grids and class balance chart.

compare.py passes in the images and class shares. Run python compare.py from
the project folder to generate the PNGs.
"""

import matplotlib

# Draw image files without opening windows. Set the backend before importing pyplot.
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torchvision.utils import make_grid


def save_grids(path, groups, nrow, columns, size):
    # Make one subplot per named group, such as the VAE samples or the real test images.
    rows = len(groups) // columns
    figure, axes = plt.subplots(rows, columns, figsize=size, squeeze=False, layout="constrained")
    for ax, (name, images) in zip(axes.flat, groups.items()):
        # The models use [-1, 1]. Convert to [0, 1] to show the images.
        images = ((images + 1) / 2).clamp(0, 1)
        grid = make_grid(images, nrow=nrow, pad_value=1)
        # Take one grayscale channel and keep the original pixels visible.
        ax.imshow(grid[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
    figure.savefig(path, dpi=160)
    # Release the figure so each of the four outputs gets its own canvas.
    plt.close(figure)


def save_class_balance(path, classes, groups):
    figure, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    positions = range(len(classes))
    width = 0.26
    # Put one bar per group beside the others at each class and show shares as percentages.
    for offset, (name, shares) in zip((-width, 0, width), groups.items()):
        ax.bar([x + offset for x in positions], shares * 100, width=width, label=name)
    ax.set_xticks(positions, classes, rotation=30, ha="right")
    ax.set_ylabel("Images in each clothing class (%)")
    ax.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)
