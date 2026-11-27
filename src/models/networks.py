"""Shared layer designs for the VAE and DCGAN. Each model has its own weights"""

import torch.nn as nn


class ConvDecoder(nn.Module):
    """Grow a batch of latent codes into 32 by 32 images"""

    def __init__(self, latent_dim, out_channels, base=64):
        super().__init__()
        self.net = nn.Sequential(
            # Start with a 4 by 4 grid, 256 channels with the default settings
            nn.ConvTranspose2d(
                latent_dim, base * 4, kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            # Grow to 8 by 8 and then 16 by 16 while using fewer channels
            # Batch norm keeps the activations on a steady scale and ReLU clips negatives
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            # One grayscale channel at 32 by 32, tanh keeps pixels in [-1, 1]
            nn.ConvTranspose2d(base, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        # Reshape each code to a 1 by 1 grid so the first convolution can use it
        return self.net(z.view(z.size(0), -1, 1, 1))


class ConvEncoder(nn.Module):
    """Shrink each image to a vector of features or scores."""

    def __init__(self, in_channels, out_dim, base=64):
        super().__init__()
        self.features = nn.Sequential(
            # Shrink the image from 32 to 16 to 8 to 4 pixels on each side
            # Skip batch norm in the first block like in DCGAN
            nn.Conv2d(in_channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Bring the 4 by 4 grid down to one value per output channel
        # For the VAE that's a mean and log variance for each code coordinate
        # For D it's one score per image
        self.head = nn.Conv2d(base * 4, out_dim, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        # Flatten the final output so each image has one row
        return self.head(self.features(x)).flatten(1)

def init_dcgan_weights(module):
    """Use the same starting weight distributions for both models."""
    # Start conv weights near zero and batch norm scales near one
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)

def count_parameters(module):
    # Count all parameter values so we can log the model size
    return sum(p.numel() for p in module.parameters())
