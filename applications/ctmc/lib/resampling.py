"""
Resampling algorithms for Sequential Monte Carlo (SMC).

This module implements various resampling strategies used in particle filters
and SMC methods, including systematic resampling, partial resampling, and
effective sample size (ESS) computation.

All algorithms work with log weights for numerical stability.
"""

from typing import Tuple

import torch
from jaxtyping import Float, Int


def compute_ess(
    weights: Float[torch.Tensor, "batch"], dim: int = 0
) -> Float[torch.Tensor, ""]:
    """Compute Effective Sample Size (ESS) from normalized weights.

    ESS measures how many "effective" independent samples we have:
    ESS = (Σ w_i)² / Σ w_i²

    ESS ∈ [1, K] where K is the number of particles.
    ESS = K when all weights are equal (best case).
    ESS = 1 when one weight dominates (worst case).

    Args:
        weights: Normalized weights (should sum to 1) [batch]
        dim: Dimension along which to compute ESS

    Returns:
        Effective sample size scalar
    """
    sum_weights = weights.sum(dim=dim)
    sum_weights_squared = (weights**2).sum(dim=dim)

    # Avoid division by zero
    ess = sum_weights**2 / torch.clamp(sum_weights_squared, min=1e-12)

    return ess


def compute_ess_from_log_weights(
    log_weights: Float[torch.Tensor, "batch"], dim: int = 0
) -> Float[torch.Tensor, ""]:
    """Compute ESS from log weights.

    Numerically stable version that normalizes log weights first.

    Args:
        log_weights: Log weights [batch]
        dim: Dimension along which to compute ESS

    Returns:
        Effective sample size scalar
    """
    normalized_weights = normalize_weights(log_weights, dim=dim)
    return compute_ess(normalized_weights, dim=dim)


def normalize_log_weights(
    log_weights: Float[torch.Tensor, "batch"], dim: int = 0
) -> Float[torch.Tensor, "batch"]:
    """Normalize log weights to prevent overflow/underflow.

    Subtracts the maximum and then log-sum-exp normalizes:
    log w̃_i = log w_i - max(log w) - log(Σ exp(log w_j - max(log w)))

    Args:
        log_weights: Unnormalized log weights [batch]
        dim: Dimension along which to normalize

    Returns:
        Normalized log weights [batch]
    """
    # Subtract maximum for numerical stability
    log_weights_stable = log_weights - log_weights.max(dim=dim, keepdim=True)[0]

    # Normalize using log-sum-exp
    log_weights_normalized = log_weights_stable - torch.logsumexp(
        log_weights_stable, dim=dim, keepdim=True
    )

    return log_weights_normalized


def normalize_weights(
    log_weights: Float[torch.Tensor, "batch"], dim: int = 0
) -> Float[torch.Tensor, "batch"]:
    """Convert log weights to normalized probabilities.

    Args:
        log_weights: Log weights [batch]
        dim: Dimension along which to normalize

    Returns:
        Normalized weights in probability space [batch]
    """
    log_weights_normalized = normalize_log_weights(log_weights, dim=dim)
    return torch.exp(log_weights_normalized)


def systematic_indices(
    weights: Float[torch.Tensor, "batch"],
) -> Int[torch.Tensor, "batch"]:
    """Systematic resampling indices for particle filters.

    This algorithm provides low-variance sampling by dividing the cumulative
    sum into N equal divisions and selecting one particle randomly from each.
    This guarantees that each sample is between 0 and 2/N apart.

    Args:
        weights: Normalized weights (should sum to 1) [batch]

    Returns:
        Resampled indices [batch]
    """
    N = len(weights)
    device = weights.device

    # Create N subdivisions with random position within each
    positions = (torch.rand(N, device=device) + torch.arange(N, device=device)) / N

    # Compute cumulative sum
    cumulative_sum = torch.cumsum(weights, dim=0)

    # Find indices using searchsorted
    indices = torch.searchsorted(cumulative_sum, positions, right=False)

    # Clamp to valid range
    indices = torch.clamp(indices, 0, N - 1)

    return indices


def partial_resample(
    particles: torch.Tensor,
    log_weights: Float[torch.Tensor, "batch"],
    resample_fraction: float = 0.2,
) -> Tuple[torch.Tensor, Float[torch.Tensor, "batch"]]:
    """Partial resampling strategy that only resamples low-weight particles.

    This approach maintains diversity by:
    1. Keeping high-weight particles unchanged
    2. Resampling only the lowest-weight particles
    3. Redistributing weight uniformly among resampled particles

    Args:
        particles: Particle states [batch, ...]
        log_weights: Log weights [batch]
        resample_fraction: Fraction of particles to resample (between 0 and 1)

    Returns:
        Tuple of (resampled_particles, resampled_log_weights)
    """
    batch_size = particles.shape[0]
    resample_count = int(resample_fraction * batch_size)

    if resample_count == 0:
        return particles, log_weights

    # Ensure even number for the algorithm
    if resample_count % 2 != 0:
        resample_count -= 1

    if resample_count == 0:
        return particles, log_weights

    # Sort particles by weight (ascending order)
    sorted_indices = torch.argsort(log_weights)

    # Split into low-weight (to resample) and high-weight (to keep)
    half_resample = resample_count // 2

    # Rearrange: [high_weight_particles, low_weight_particles]
    reordered_particles = torch.cat(
        [
            particles[sorted_indices[-half_resample:]],  # highest weights
            particles[sorted_indices[:-half_resample]],  # remaining particles
        ],
        dim=0,
    )

    reordered_log_weights = torch.cat(
        [
            log_weights[sorted_indices[-half_resample:]],
            log_weights[sorted_indices[:-half_resample]],
        ],
        dim=0,
    )

    # Resample among the first resample_count particles
    resample_weights = normalize_weights(reordered_log_weights[:resample_count])
    resampled_indices = systematic_indices(resample_weights)

    # Apply resampling
    reordered_particles[:resample_count] = reordered_particles[resampled_indices]

    # Compute normalization constant and uniform weights for resampled particles
    norm_constant = torch.logsumexp(
        reordered_log_weights[:resample_count], dim=0, keepdim=True
    )
    uniform_log_weight = norm_constant - torch.log(
        torch.tensor(resample_count, dtype=torch.float32, device=log_weights.device)
    )

    reordered_log_weights[:resample_count] = uniform_log_weight

    return reordered_particles, reordered_log_weights


def stratified_indices(
    weights: Float[torch.Tensor, "batch"],
) -> Int[torch.Tensor, "batch"]:
    """Stratified resampling indices for particle filters.

    This algorithm aims to make selections relatively uniformly across the
    particles. It divides the cumulative sum of the weights into N equal
    divisions, and then selects one particle randomly from each division. This
    guarantees that each sample is between 0 and 2/N apart.

    Args:
        weights: Normalized weights (should sum to 1) [batch]

    Returns:
        Resampled indices [batch]
    """
    N = len(weights)
    device = weights.device

    # Make N subdivisions, and choose a random position within each one
    positions = (torch.rand(N, device=device) + torch.arange(N, device=device)) / N

    # Initialize indices tensor
    indices = torch.zeros(N, dtype=torch.int32, device=device)

    # Compute cumulative sum
    cumulative_sum = torch.cumsum(weights, dim=0)

    # Find indices using the stratified algorithm
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1

    return indices


def multinomial_indices(
    weights: Float[torch.Tensor, "batch"],
) -> Int[torch.Tensor, "batch"]:
    """Multinomial resampling indices.

    Simple categorical sampling from the weight distribution.
    Higher variance than systematic resampling but easier to implement.

    Args:
        weights: Normalized weights [batch]

    Returns:
        Resampled indices [batch]
    """
    N = len(weights)
    categorical = torch.distributions.Categorical(probs=weights)
    return categorical.sample((N,))


def stratified_resample(
    particles: torch.Tensor, log_weights: Float[torch.Tensor, "batch"]
) -> Tuple[torch.Tensor, Float[torch.Tensor, "batch"]]:
    """Stratified resampling for SMC.

    Resamples ALL particles using stratified sampling, which provides lower
    variance than multinomial resampling by ensuring more uniform selection
    across the particle distribution. Sets all new weights to uniform (1/K).

    Args:
        particles: Particle states [batch, ...]
        log_weights: Log weights [batch]

    Returns:
        Tuple of (resampled_particles, uniform_log_weights)
    """
    import math

    batch_size = particles.shape[0]
    device = particles.device

    # Normalize log weights to probabilities
    normalized_weights = normalize_weights(log_weights, dim=0)

    # Get resampling indices using stratified sampling
    indices = stratified_indices(normalized_weights)

    # Resample particles according to indices
    resampled_particles = particles[indices]

    # Set all weights to uniform: log(1/K) = -log(K)
    uniform_log_weights = torch.full(
        (batch_size,), -math.log(batch_size), dtype=torch.float32, device=device
    )

    return resampled_particles, uniform_log_weights


def multinomial_resample(
    particles: torch.Tensor, log_weights: Float[torch.Tensor, "batch"]
) -> Tuple[torch.Tensor, Float[torch.Tensor, "batch"]]:
    """Standard multinomial resampling for SMC.

    Resamples ALL particles according to their normalized weights and sets
    all new weights to uniform (1/K). This is the standard multinomial
    resampling algorithm used in particle filters.

    Args:
        particles: Particle states [batch, ...]
        log_weights: Log weights [batch]

    Returns:
        Tuple of (resampled_particles, uniform_log_weights)
    """
    import math

    batch_size = particles.shape[0]
    device = particles.device

    # Normalize log weights to probabilities
    normalized_weights = normalize_weights(log_weights, dim=0)

    # Get resampling indices using multinomial sampling
    indices = multinomial_indices(normalized_weights)

    # Resample particles according to indices
    resampled_particles = particles[indices]

    # Set all weights to uniform: log(1/K) = -log(K)
    uniform_log_weights = torch.full(
        (batch_size,), -math.log(batch_size), dtype=torch.float32, device=device
    )

    return resampled_particles, uniform_log_weights


def get_resampling_fn(method: str):
    """Get resampling function by method name.

    Returns a function that takes (particles, log_weights, **kwargs) and
    returns (resampled_particles, resampled_log_weights).

    Args:
        method: Resampling method name ("partial", "multinomial", "stratified")

    Returns:
        Resampling function with signature (particles, log_weights, **kwargs) -> (particles, log_weights)

    Raises:
        ValueError: If method is not supported
    """
    if method == "partial":
        return partial_resample
    elif method == "multinomial":
        return multinomial_resample
    elif method == "stratified":
        return stratified_resample
    else:
        available_methods = ["partial", "multinomial", "stratified"]
        raise ValueError(
            f"Unknown resampling method '{method}'. Available methods: {available_methods}"
        )


def should_resample(
    log_weights: Float[torch.Tensor, "batch"],
    ess_threshold: float = 0.5,
) -> bool:
    """Determine whether resampling should be triggered.

    Resampling is triggered when:
    1. ESS falls below threshold * N (particle degeneracy)
    2. We've passed the minimum step threshold (avoid early resampling)

    Args:
        log_weights: Log weights [batch]
        ess_threshold: ESS threshold as fraction of particle count
        min_step_threshold: Minimum steps before allowing resampling (not used here)

    Returns:
        Whether to trigger resampling
    """
    N = len(log_weights)
    ess = compute_ess_from_log_weights(log_weights)

    return ess < ess_threshold * N
