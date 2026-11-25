"""Evaluation metrics: FID, precision/recall, mode coverage, sharpness and latent smoothness"""
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

def frechet_distance(feats_a, feats_b):
    """Fréchet distance between Gaussians fitted to two feature sets (Heusel et al., 2017)
    With Inception features this is the usual FID. Lower is better
    """
    a = np.asarray(feats_a, dtype=np.float64)
    b = np.asarray(feats_b, dtype=np.float64)
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)

    with warnings.catch_warnings():
        # Some features are always 0, so the covariance can be singular; sqrtm usually still works
        warnings.simplefilter("ignore", linalg.LinAlgWarning)
        covmean = linalg.sqrtm(cov_a @ cov_b)
    if not np.isfinite(covmean).all():  # if sqrtm failed, add a tiny value to the diagonal and retry
        ridge = 1e-6 * np.eye(cov_a.shape[0])
        covmean = linalg.sqrtm((cov_a + ridge) @ (cov_b + ridge))
    covmean = np.real(covmean)  # drop tiny imaginary parts caused by rounding

    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a) + np.trace(cov_b) - 2 * np.trace(covmean))

def _knn_radius(feats, k, batch=1000):
    """Distance from each point to its k-th nearest neighbour within the same set"""
    radii = []
    for chunk in feats.split(batch):
        dists = torch.cdist(chunk, feats)
        radii.append(dists.kthvalue(k + 1, dim=1).values)  # k+1: skip the distance to itself (0)
    return torch.cat(radii)

def _fraction_inside(queries, support, support_radii, batch=1000):
    """Fraction of query points that fall inside at least one k-NN ball of the support set"""
    inside = []
    for chunk in queries.split(batch):
        dists = torch.cdist(chunk, support)
        inside.append((dists <= support_radii.unsqueeze(0)).any(dim=1))
    return torch.cat(inside).float().mean().item()

def precision_recall(real_feats, fake_feats, k=3):
    """Precision and recall with k-NN balls (Kynkäänniemi et al., 2019)
    1. precision = share of fake images close to a real one (fidelity)
    2. recall = share of real images close to a fake one (diversity)
    """
    real = torch.as_tensor(real_feats, dtype=torch.float32)
    fake = torch.as_tensor(fake_feats, dtype=torch.float32)
    precision = _fraction_inside(fake, real, _knn_radius(real, k))
    recall = _fraction_inside(real, fake, _knn_radius(fake, k))
    return precision, recall

def mode_coverage(pred_labels, num_classes, reference_labels, min_share=0.01):
    """How the generated images spread over the classes, using the evaluation classifier
    1. modes_covered = classes with at least min_share of the images
    2. class_kl = KL(generated || real), 0 when as balanced as the real data
    """
    p = np.bincount(pred_labels, minlength=num_classes) / len(pred_labels)
    q = np.bincount(reference_labels, minlength=num_classes) / len(reference_labels)
    eps = 1e-10
    kl = float(np.sum(p * np.log((p + eps) / (q + eps))))
    return {"modes_covered": int((p >= min_share).sum()), "class_kl": kl, "class_histogram": p.tolist()}

def _gray01(images):
    """(B, C, H, W) in [-1, 1] -> (B, 1, H, W) grayscale in [0, 1]"""
    x = (images.clamp(-1, 1) + 1) / 2
    if x.size(1) == 3:
        weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
        x = (x * weights).sum(dim=1, keepdim=True)
    return x

LAPLACIAN_KERNEL = torch.tensor([[0.0, 1.0, 0.0],
                                 [1.0, -4.0, 1.0],
                                 [0.0, 1.0, 0.0]])

def laplacian_variance(images):
    """Variance of the Laplacian (an edge filter) per image. Weak edges give a low value, so blurry = low"""
    gray = _gray01(images)
    kernel = LAPLACIAN_KERNEL.to(gray).view(1, 1, 3, 3)
    response = F.conv2d(gray, kernel)  # 'valid' convolution
    return response.flatten(1).var(dim=1)

def high_frequency_ratio(images, cutoff=0.25):
    """Share of each image's spectral energy above `cutoff` (cycles/pixel). Blurry images have less"""
    gray = _gray01(images)
    gray = gray - gray.mean(dim=(-2, -1), keepdim=True)  # remove the mean brightness (DC component)
    power = torch.fft.fft2(gray).abs() ** 2
    h, w = gray.shape[-2:]
    fy = torch.fft.fftfreq(h, device=gray.device).view(-1, 1)
    fx = torch.fft.fftfreq(w, device=gray.device).view(1, -1)
    high = (fy ** 2 + fx ** 2).sqrt() > cutoff
    return power[..., high].flatten(1).sum(1) / power.flatten(1).sum(1).clamp_min(1e-12)

def psnr(x, y):
    """Peak signal-to-noise ratio in dB, per image, computed in [0, 1] (so MAX = 1)"""
    mse = (((x.clamp(-1, 1) - y.clamp(-1, 1)) / 2) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))

def nearest_neighbors(queries, reference, batch=500):
    """Pixel distance from each query image to its closest reference image, and that image's index"""
    ref = reference.flatten(1)
    dists, idxs = [], []
    for chunk in queries.flatten(1).split(batch):
        nearest = torch.cdist(chunk, ref).min(dim=1)
        dists.append(nearest.values)
        idxs.append(nearest.indices)
    return torch.cat(dists), torch.cat(idxs)

def slerp(z0, z1, t):
    """Spherical interpolation from z0 to z1. A straight line would pass near 0, where the model saw few z"""
    omega = torch.acos((F.normalize(z0, dim=-1) * F.normalize(z1, dim=-1)).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7))
    omega = omega.view(-1, 1, 1)
    t = t.view(1, -1, 1)
    sin_omega = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / sin_omega) * z0.unsqueeze(1) + \
           (torch.sin(t * omega) / sin_omega) * z1.unsqueeze(1)

def path_smoothness(path_feats):
    """How evenly the images change along each path: CV of the step sizes. Lower = smoother"""
    steps = (path_feats[:, 1:] - path_feats[:, :-1]).norm(dim=-1)  # (P, S-1)
    cv = steps.std(dim=1) / steps.mean(dim=1).clamp_min(1e-12)
    return cv.mean().item()