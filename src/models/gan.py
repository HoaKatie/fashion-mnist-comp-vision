"""Define the two parts of the DCGAN.

The generator turns noise into images. The discriminator scores images for
training. train_gan.py trains both, and compare.py calls decode() on the
saved generator to make samples.
"""

import torch.nn as nn

from src.models.networks import ConvDecoder, ConvEncoder

class GAN(nn.Module):
    def __init__(self, latent_dim, channels, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        # G grows each latent vector into a grayscale image using the same design as the VAE decoder.
        self.generator = ConvDecoder(latent_dim, channels, base)
        # D shrinks each real or generated image into one score. The loss applies sigmoid internally.
        self.discriminator = ConvEncoder(channels, 1, base)

    def decode(self, z):
        # Expose the generator under the same method name as the VAE so compare.py can call both.
        return self.generator(z)
