"""Define the three losses used to train the VAE and DCGAN.

train_vae.py calls vae_loss before its update. train_gan.py calls
discriminator_loss and generator_loss for its two updates.
"""

import torch
import torch.nn.functional as F

def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    """Negative ELBO = reconstruction + beta * KL, averaged over the batch. Returns (loss, recon, kl)"""
    batch = x.size(0)

    # Add the squared errors across pixels, then divide by the number of images.
    # This keeps the pixel loss on the scale used with the KL penalty.
    recon = F.mse_loss(x_recon, x, reduction="sum") / batch

    # Compute how far the encoder's Gaussian is from standard normal noise.
    # exp(logvar) gives the variance. Sum the latent terms and average over images.
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch

    # Use the combined loss for the update. Return both parts for the training log.
    return recon + beta * kl, recon, kl

def discriminator_loss(real_logits, fake_logits):
    """D is a binary classifier: real -> 1, fake -> 0 (Goodfellow et al., 2014)"""
    # Use target 1 for real images and target 0 for generated images.
    # The loss handles sigmoid internally, so the inputs are the unscaled scores.
    real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return real + fake

def generator_loss(fake_logits):
    """Non-saturating loss: G maximises log D(G(z)), which gives stronger gradients early in training"""
    # Give generated images target 1 so G learns to raise D's score on its own images.
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
