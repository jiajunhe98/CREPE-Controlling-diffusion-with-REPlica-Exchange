import abc

import torch
import torch.nn as nn
from jaxtyping import Float


class Noise(abc.ABC, nn.Module):
    """
    Baseline forward method to get the total + rate of noise at a timestep
    """

    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

    """
    Assume time goes from 0 to 1
    """

    @abc.abstractmethod
    def rate_noise(
        self, t: Float[torch.Tensor, "batch"]
    ) -> Float[torch.Tensor, "batch"]:
        """
        Rate of change of noise ie g(t). This is sigma in the paper.
        """
        pass

    @abc.abstractmethod
    def total_noise(
        self, t: Float[torch.Tensor, "batch"]
    ) -> Float[torch.Tensor, "batch"]:
        """
        Total noise ie \int_0^t g(t) dt + g(0). This is sigma bar in the paper.
        """
        pass


class GeometricNoise(Noise, nn.Module):
    """
    Geometric noise distribution where σ(t) = σ_min^(1-t) * σ_max^t.

    This creates a smooth geometric interpolation between σ_min at t=0 and σ_max at t=1.
    The noise schedule is chosen such that the prior loss D_KL(p_T|0(·|x_0) || π) and
    the approximation of p_data with p_σ(0) are negligible.

    Commonly used with uniform transition matrices scaled down by 1/N, where p_base
    is taken to be uniform for the uniform graph case.

    Args:
        sigma_min: Minimum noise level at t=0 (default: 1e-3)
        sigma_max: Maximum noise level at t=1 (default: 1.0)
        learnable: Whether noise parameters should be learnable (default: False)
    """

    def __init__(self, sigma_min=1e-3, sigma_max=1, learnable=False):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])
        if learnable:
            self.sigmas = nn.Parameter(self.sigmas)
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(
        self, t: Float[torch.Tensor, "batch"]
    ) -> Float[torch.Tensor, "batch"]:
        return (
            self.sigmas[0] ** (1 - t)
            * self.sigmas[1] ** t
            * (self.sigmas[1].log() - self.sigmas[0].log())
        )

    def total_noise(
        self, t: Float[torch.Tensor, "batch"]
    ) -> Float[torch.Tensor, "batch"]:
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t


class LogLinearNoise(Noise, nn.Module):
    """
    Log Linear noise schedule built so that 1 - 1/e^(n(t)) interpolates between 0 and ~1
    when t goes from 0 to 1. Used for absorbing

    Total noise is -log(1 - (1 - eps) * t), so the sigma will be (1-eps) / (1 - (1 - eps) * t)

    That noise schedule is equivalen to Jiaxin's linear masking schedule where alpha(t) = 1-t with an epsilon
    correction for stability.

    Linear noise as implemented in Jiaxin's paper where alpha(t) = exp(-int_0^t beta(s)ds) = 1-t
    Hence beta(t) = 1/1-t.
    rate noise = beta(t)
    total noise = -log(1-t) + 1
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(
        self, t: Float[torch.Tensor, "batch"]
    ) -> Float[torch.Tensor, "batch"]:
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(
        self, t: Float[torch.Tensor, "batch"]
    ) -> Float[torch.Tensor, "batch"]:
        return -torch.log1p(-(1 - self.eps) * t)


class CosineNoise(Noise, nn.Module):
    """
    Cosine noise as seen in Jiaxin's paper.

    In the paper alpha_t = 1 - cos(pi / 2 * (1 - t)) and we know sigma(t) = - alpha'(t) / alpha(t)
    and \bar{sigma}(t) = - log(alpha(t))
    We use instead here alpha_t = 1 - cos(pi / 2 * (1 - (1 - eps)t))
    so sigma(t) = (1-eps)*pi / 2 * sin(pi / 2 * (1 - (1 - eps)t)) / (1 - cos(pi / 2 * (1 - (1 - eps)t)))
    and \bar{sigma}(t) = - log(alpha(t)) = - log(1 - cos(pi / 2 * (1 - (1 - eps)t)))
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def rate_noise(
        self, t: Float[torch.Tensor, "batch"]
    ) -> Float[torch.Tensor, "batch"]:
        return (
            ((1 - self.eps) * torch.pi / 2)
            * torch.sin(torch.pi / 2 * (1 - (1 - self.eps) * t))
            / (1 - torch.cos(torch.pi / 2 * (1 - (1 - self.eps) * t)))
        )

    def total_noise(
        self, t: Float[torch.Tensor, "batch"]
    ) -> Float[torch.Tensor, "batch"]:
        return -torch.log1p(-torch.cos(torch.pi / 2 * (1 - (1 - self.eps) * t)))
