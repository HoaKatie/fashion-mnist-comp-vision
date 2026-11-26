"""Define the shared image layers for the VAE and DCGAN.

ConvDecoder grows a latent vector into an image. ConvEncoder shrinks an image
into a vector. The two models use the same layer designs and learn their own
weights. vae.py and gan.py import these classes.
"""

import torch.nn as nn

class ConvDecoder(nn.Module):
    """Turn a batch of latent vectors into 32 by 32 images."""

    def __init__(self, latent_dim, out_channels, base=64):
        super().__init__()
        self.net = nn.Sequential(
            # 1. Turn the latent vector into a 4 by 4 grid with 256 channels by default.
            nn.ConvTranspose2d(latent_dim, base * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            # 2. Grow to 8 by 8, then 16 by 16, with fewer channels at each size.
            # Batch normalization scales the channel activations. ReLU clips negatives.
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            # 3. Grow to 32 by 32 with one grayscale channel. tanh gives [-1, 1] pixels.
            nn.ConvTranspose2d(base, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        # Each row in z is one code. Treat it as a 1 by 1 image with latent_dim channels.
        return self.net(z.view(z.size(0), -1, 1, 1))

class ConvEncoder(nn.Module):
    """Summarize each image as a short vector of numbers."""

    def __init__(self, in_channels, out_dim, base=64):
        super().__init__()
        self.features = nn.Sequential(
            # 1. Halve width and height at each step. The sizes are 32, 16, 8, then 4.
            # Skip batch normalization on the first layer, as in the DCGAN design.
            nn.Conv2d(in_channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 2. Collapse the final 4 by 4 grid to one value per output channel.
        # This gives the VAE two numbers per latent coordinate,
        # or gives the discriminator one score for the entire image.
        self.head = nn.Conv2d(base * 4, out_dim, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        # Run the feature layers and output layer, then return one vector per image.
        return self.head(self.features(x)).flatten(1)

def init_dcgan_weights(module):
    """Start both models with weights drawn from the same distributions."""
    # Convolution weights start near zero, and batch normalization scales start near one.
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)

def count_parameters(module):
    # Add up the number of values in each parameter tensor for the training log.
    return sum(p.numel() for p in module.parameters())
