"""The VAE learns to rebuild images by sampling a code for each one.

In compare.py I use the decoder to make new images from noise.
"""

import torch
import torch.nn as nn

from src.models.networks import ConvDecoder, ConvEncoder


class VAE(nn.Module):
    def __init__(self, latent_dim, channels, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        # With 32 latent values, the encoder needs 32 means and 32 log variances
        # The decoder only takes the one sampled code of length 32
        self.encoder = ConvEncoder(channels, 2 * latent_dim, base)
        self.decoder = ConvDecoder(latent_dim, channels, base)

    def encode(self, x):
        # Split into means and log variances, one pair of vectors per image
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        # logvar is log of variance, so exp(logvar / 2) gives the standard deviation
        # Scale standard normal noise by the standard deviation and shift by the mean
        # Drawing the noise separately lets gradients still reach the mean and scale
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        # Get the code distribution for this image
        mu, logvar = self.encode(x)
        # Draw one code from it
        z = self.reparameterize(mu, logvar)
        # Decode it and keep the mean and log variance for the KL loss
        return self.decoder(z), mu, logvar

    def decode(self, z):
        # Decode a random code to get a new image
        return self.decoder(z)

    @torch.no_grad()
    def reconstruct(self, x):
        # Use the mean code if we want a reconstruction without extra noise
        mu, _ = self.encode(x)
        return self.decoder(mu)
