"""Training losses for the VAE and the two parts of the DCGAN."""

import torch
import torch.nn.functional as F


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    """VAE loss is pixel error plus beta times KL. Return both parts for the log too."""
    batch = x.size(0)

    # Sum the pixel errors for each image and average over the batch
    # Use a per-image loss here so it matches how we average the KL term
    recon = F.mse_loss(x_recon, x, reduction="sum") / batch

    # KL measures how far the encoder's distribution is from standard normal noise
    # Take exp(logvar) to get the variance, then add up the terms for each code
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch

    # Train with the total and log the two parts separately
    return recon + beta * kl, recon, kl


def discriminator_loss(real_logits, fake_logits):
    """Real images get target 1 and fakes get 0 (Goodfellow et al., 2014)."""
    # Give real images ones and generated images zeros
    # BCE with logits handles sigmoid so pass the raw scores
    real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake = F.binary_cross_entropy_with_logits(
        fake_logits, torch.zeros_like(fake_logits)
    )
    return real + fake


def generator_loss(fake_logits):
    """Train the generator with real targets so it gets a useful gradient even when D spots most fakes."""
    # Give fakes a real target so G learns to raise their scores
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
