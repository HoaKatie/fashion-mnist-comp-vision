"""Define how the VAE encodes, samples, and decodes an image.

train_vae.py uses the full path to learn reconstructions. compare.py uses
decode() to turn random noise into images with the saved decoder.
"""

import torch
import torch.nn as nn

from src.models.networks import ConvDecoder, ConvEncoder

class VAE(nn.Module):
    def __init__(self, latent_dim, channels, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        # A latent code has 32 entries by default. The encoder outputs 32 means
        # and 32 log-variances, and the decoder takes a single code of length 32.
        self.encoder = ConvEncoder(channels, 2 * latent_dim, base)
        self.decoder = ConvDecoder(latent_dim, channels, base)

    def encode(self, x):
        # Split the output across features, keeping one pair of vectors per image.
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        # logvar is log(sigma squared). Taking exp(logvar / 2) gives sigma.
        # Draw unit normal noise, multiply by sigma, and add mu to sample the code.
        # The noise is separate, so gradients can pass through the scale and shift.
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        # 1. Get the latent distribution for the input image.
        mu, logvar = self.encode(x)
        # 2. Sample a code from that distribution.
        z = self.reparameterize(mu, logvar)
        # 3. Decode the sample and return the Gaussian parameters for the KL loss.
        return self.decoder(z), mu, logvar

    def decode(self, z):
        # We can also decode random noise directly to generate a new image.
        return self.decoder(z)

    @torch.no_grad()
    def reconstruct(self, x):
        # Decode the mean code for a reconstruction without sampling noise.
        mu, _ = self.encode(x)
        return self.decoder(mu)
