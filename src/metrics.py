"""Calculate the image and latent-space scores used by compare.py.

The functions take image tensors, classifier features, or class counts and
turn them into scores. To compute them on the bundled models, run
python compare.py from the project folder.
"""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

def frechet_distance(feats_a, feats_b):
    """Compare the mean and covariance of two groups of image features."""
    # Use float64 for the matrix calculations. Each row describes one image.
    a = np.asarray(feats_a, dtype=np.float64)
    b = np.asarray(feats_b, dtype=np.float64)
    # The mean is the average feature vector. Covariance describes how features vary together.
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)

    with warnings.catch_warnings():
        # Some features can be constant, so the covariance matrix can be singular.
        warnings.simplefilter("ignore", linalg.LinAlgWarning)
        covmean = linalg.sqrtm(cov_a @ cov_b)
    # If the matrix square root gives invalid values, add a small diagonal term and try again.
    if not np.isfinite(covmean).all():
        ridge = 1e-6 * np.eye(cov_a.shape[0])
        covmean = linalg.sqrtm((cov_a + ridge) @ (cov_b + ridge))
    covmean = np.real(covmean)  # Drop tiny imaginary parts from rounding.

    # Add the differences in the means and covariances. Clamp tiny negative roundoff to zero.
    distance = ((mu_a - mu_b) ** 2).sum() + np.trace(cov_a) + np.trace(cov_b) - 2 * np.trace(covmean)
    return float(max(distance, 0.0))

def _knn_radius(feats, k, batch=1000):
    """Distance from each point to its k-th nearest neighbour within the same set"""
    radii = []
    # Compare each chunk with the whole set. Chunking keeps the distance matrix smaller.
    for chunk in feats.split(batch):
        dists = torch.cdist(chunk, feats)
        radii.append(dists.kthvalue(k + 1, dim=1).values)  # k+1: skip the distance to itself (0)
    return torch.cat(radii)

def _fraction_inside(queries, support, support_radii, batch=1000):
    """Fraction of query points that fall inside at least one k-NN ball of the support set"""
    inside = []
    for chunk in queries.split(batch):
        dists = torch.cdist(chunk, support)
        # Count a query if its distance to any support image fits inside that image's radius.
        inside.append((dists <= support_radii.unsqueeze(0)).any(dim=1))
    return torch.cat(inside).float().mean().item()

def precision_recall(real_feats, fake_feats, k=3):
    """Precision and recall with k-NN balls (Kynkäänniemi et al., 2019)
    1. precision = share of fake images close to a real one (fidelity)
    2. recall = share of real images close to a fake one (diversity)
    """
    real = torch.as_tensor(real_feats, dtype=torch.float32)
    fake = torch.as_tensor(fake_feats, dtype=torch.float32)
    # Precision tests fake images against real neighbourhoods. Recall swaps the two sets.
    precision = _fraction_inside(fake, real, _knn_radius(real, k))
    recall = _fraction_inside(real, fake, _knn_radius(fake, k))
    return precision, recall

def class_shares(labels, num_classes):
    # Count images per label and divide by the total. Include classes with zero images.
    return torch.bincount(labels, minlength=num_classes).double() / len(labels)

def class_kl(shares, real_shares):
    # Compare the two class shares and weight each log ratio by the share in the group being scored.
    # The small offset keeps the logarithm defined when a class has no samples.
    ratio = (shares + 1e-10) / (real_shares + 1e-10)
    return (shares * ratio.log()).sum().item()

# Compare each centre pixel with its four nearest neighbours to pick up edges.
LAPLACIAN_KERNEL = torch.tensor([[0.0, 1.0, 0.0],
                                 [1.0, -4.0, 1.0],
                                 [0.0, 1.0, 0.0]])

def laplacian_variance(images):
    # Change [-1, 1] pixels back to [0, 1] before applying the edge filter.
    gray = (images.clamp(-1, 1) + 1) / 2
    kernel = LAPLACIAN_KERNEL.to(gray).view(1, 1, 3, 3)
    response = F.conv2d(gray, kernel)  # 'valid' convolution
    # Take the variance across pixels in each filtered image. Blur usually lowers it.
    return response.flatten(1).var(dim=1)

def slerp(z0, z1, t):
    """Spherical interpolation from z0 to z1. A straight line would pass near 0, where the model saw few z"""
    # The dot product of the unit vectors gives the cosine of the angle between them.
    # Clamp away from the endpoints so the sine used in the denominator stays nonzero.
    omega = torch.acos((F.normalize(z0, dim=-1) * F.normalize(z1, dim=-1)).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7))
    # Give the angles and frame positions matching axes, then mix the two endpoints.
    omega = omega.view(-1, 1, 1)
    t = t.view(1, -1, 1)
    sin_omega = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / sin_omega) * z0.unsqueeze(1) + \
           (torch.sin(t * omega) / sin_omega) * z1.unsqueeze(1)

def path_smoothness(path_feats):
    """How evenly the images change along each path: CV of the step sizes. Lower = smoother"""
    # Measure the feature distance from each frame to the next frame on its path.
    steps = (path_feats[:, 1:] - path_feats[:, :-1]).norm(dim=-1)  # (P, S-1)
    # Divide the spread of the step sizes by their mean, then average over paths.
    cv = steps.std(dim=1) / steps.mean(dim=1).clamp_min(1e-12)
    return cv.mean().item()
