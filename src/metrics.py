"""Image and interpolation scores used for the VAE and DCGAN comparison.

Run python compare.py from the project folder to get these scores for the saved models.
"""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg


def frechet_distance(feats_a, feats_b):
    """Compare the centre and spread of two sets of image features."""
    # Use float64 for the matrix math to reduce rounding errors
    a = np.asarray(feats_a, dtype=np.float64)
    b = np.asarray(feats_b, dtype=np.float64)
    # Mean is the average feature vector, covariance tells how the features vary together
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)

    with warnings.catch_warnings():
        # Constant features can give a singular covariance matrix
        warnings.simplefilter("ignore", linalg.LinAlgWarning)
        covmean = linalg.sqrtm(cov_a @ cov_b)
    # Add a tiny value to the diagonal and retry if the square root gives invalid numbers
    if not np.isfinite(covmean).all():
        ridge = 1e-6 * np.eye(cov_a.shape[0])
        covmean = linalg.sqrtm((cov_a + ridge) @ (cov_b + ridge))
    covmean = np.real(covmean)  # Keep the real part after rounding errors

    # Combine the mean and covariance terms and clip small negative rounding errors
    distance = (
        ((mu_a - mu_b) ** 2).sum()
        + np.trace(cov_a)
        + np.trace(cov_b)
        - 2 * np.trace(covmean)
    )
    return float(max(distance, 0.0))


def _knn_radius(feats, k, batch=1000):
    """Find how far each point is from its kth closest neighbour."""
    radii = []
    # Do part of the set at a time so the distance matrix stays smaller
    for chunk in feats.split(batch):
        dists = torch.cdist(chunk, feats)
        radii.append(
            dists.kthvalue(k + 1, dim=1).values
        )  # Look at k + 1 because the first match is the point itself
    return torch.cat(radii)


def _fraction_inside(queries, support, support_radii, batch=1000):
    """Check what share of points are close enough to at least one point in the other set."""
    inside = []
    for chunk in queries.split(batch):
        dists = torch.cdist(chunk, support)
        # Count the query if it lands inside any of the neighbour radii
        inside.append((dists <= support_radii.unsqueeze(0)).any(dim=1))
    return torch.cat(inside).float().mean().item()


def precision_recall(real_feats, fake_feats, k=3):
    """Use neighbour distances to score fidelity and diversity (Following: Kynkaanniemi et al, 2019).
    Precision asks how many generated images are in real neighbourhoods.
    Recall asks how much of the real set is in generated neighbourhoods.
    """
    real = torch.as_tensor(real_feats, dtype=torch.float32)
    fake = torch.as_tensor(fake_feats, dtype=torch.float32)
    # Check fakes against real neighbours for precision, then swap for recall
    precision = _fraction_inside(fake, real, _knn_radius(real, k))
    recall = _fraction_inside(real, fake, _knn_radius(fake, k))
    return precision, recall


def class_shares(labels, num_classes):
    # Count each class, even empty ones, and divide by the number of images
    return torch.bincount(labels, minlength=num_classes).double() / len(labels)


def class_kl(shares, real_shares):
    # Weight each log ratio by that class's share in the set being scored
    # Add a tiny value so empty classes don't give log of zero
    ratio = (shares + 1e-10) / (real_shares + 1e-10)
    return (shares * ratio.log()).sum().item()


# Use each pixel and its four neighbours to pick up edges
LAPLACIAN_KERNEL = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def laplacian_variance(images):
    # Put the pixels back in [0, 1] before looking for edges
    gray = (images.clamp(-1, 1) + 1) / 2
    kernel = LAPLACIAN_KERNEL.to(gray).view(1, 1, 3, 3)
    response = F.conv2d(gray, kernel)  # No padding for the edge filter
    # Measure how much the edge response varies across pixels, blur usually lowers it
    return response.flatten(1).var(dim=1)


def slerp(z0, z1, t):
    """Blend the noise vectors along a curve to stay away from zero, where random codes rarely land."""
    # Get the angle from the dot product of the two unit vectors
    # Stay a little away from -1 and 1 to avoid dividing by zero
    omega = torch.acos(
        (F.normalize(z0, dim=-1) * F.normalize(z1, dim=-1))
        .sum(-1)
        .clamp(-1 + 1e-7, 1 - 1e-7)
    )
    # Line up the shapes so every path gets all the frame positions
    omega = omega.view(-1, 1, 1)
    t = t.view(1, -1, 1)
    sin_omega = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / sin_omega) * z0.unsqueeze(1) + (
        torch.sin(t * omega) / sin_omega
    ) * z1.unsqueeze(1)


def path_smoothness(path_feats):
    """Check how even the feature steps are on each path. A lower CV means more even steps."""
    # Measure how far the features move from one frame to the next
    steps = (path_feats[:, 1:] - path_feats[:, :-1]).norm(
        dim=-1
    )  # One row per path and one value per step
    # Divide the step spread by its average, then average those scores across paths
    cv = steps.std(dim=1) / steps.mean(dim=1).clamp_min(1e-12)
    return cv.mean().item()
