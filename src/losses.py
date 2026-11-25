"""Training objectives for the two compared methods"""
import torch
import torch.nn.functional as F

def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    """Negative ELBO = reconstruction + beta * KL, averaged over the batch. Returns (loss, recon, kl)"""
    batch = x.size(0)

    # SUM the squared error over the 1024 pixels (not the mean): the mean would make the KL
    #  approximate 1000x stronger, a hidden beta of around 1000 and much blurrier samples
    recon = F.mse_loss(x_recon, x, reduction="sum") / batch

    # KL between N(mu, sigma^2) and N(0, I) in closed form (Kingma & Welling 2014),
    # summed over the latent dims and averaged over the batch
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch

    return recon + beta * kl, recon, kl

def discriminator_loss(real_logits, fake_logits):
    """D is a binary classifier: real -> 1, fake -> 0 (Goodfellow et al., 2014)"""
    real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return real + fake

def generator_loss(fake_logits):
    """Non-saturating loss: G maximises log D(G(z)), which gives stronger gradients early in training"""
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
