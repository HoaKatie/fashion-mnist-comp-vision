"""The DCGAN generator makes fake clothes and the discriminator tries to spot them.

compare.py only needs the generator.
"""

import torch.nn as nn

from src.models.networks import ConvDecoder, ConvEncoder

class GAN(nn.Module):
    def __init__(self, latent_dim, channels, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        # G turns noise into an image using the same layer design as the VAE decoder
        self.generator = ConvDecoder(latent_dim, channels, base)
        # D gives each image one raw score and the loss handles sigmoid
        self.discriminator = ConvEncoder(channels, 1, base)

    def decode(self, z):
        # Use decode for both models so compare.py can call them the same way
        return self.generator(z)
