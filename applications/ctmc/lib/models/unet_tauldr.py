from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models import utils as mutils
from lib.models.networks_tauldr import UNet


class ImageX0PredBase(nn.Module):
    def __init__(
        self,
        cfg,
        device: torch.device,
        rank: Optional[int] = None,
        cfg_train: Optional[bool] = False,
    ):
        super().__init__()

        self.fix_logistic: bool = cfg.model.fix_logistic
        ch: int = cfg.model.ch
        num_res_blocks: int = cfg.model.num_res_blocks
        num_scales: int = cfg.model.num_scales
        ch_mult: List[int] = cfg.model.ch_mult
        self.input_channels: int = cfg.model.input_channels
        self.output_channels: int = cfg.model.output_channels * cfg.data.S
        scale_count_to_put_attn: int = cfg.model.scale_count_to_put_attn
        data_min_max: Tuple[float, float] = cfg.model.data_min_max
        self.scaling_trick: bool = cfg.model.scaling_trick
        dropout: float = cfg.model.dropout
        skip_rescale: bool = cfg.model.skip_rescale
        do_time_embed: bool = True
        time_scale_factor: float = cfg.model.time_scale_factor
        time_embed_dim: int = cfg.model.time_embed_dim
        self.device: torch.device = device
        self.cfg_train: bool = cfg_train
        self.cfg = cfg  # Store cfg for later use

        # Set class_embed_dim based on cfg_train
        class_embed_dim: Optional[int] = None
        if cfg_train:
            class_embed_dim = cfg.model.class_embed_dim
            self.cls_embed = nn.Embedding(cfg.data.num_classes, class_embed_dim)

        # Better validation for image dimensions
        _, H, W = cfg.data.shape
        required_dim = 2 ** (num_scales - 1)
        if H % required_dim != 0 or W % required_dim != 0:
            raise ValueError(
                f"Image dimensions {H}x{W} must be divisible by {2 * required_dim} for {num_scales} scales. "
                f"Consider padding images to dimensions divisible by {2 * required_dim}."
            )

        # Ensure ch_mult has the right length
        if len(ch_mult) != num_scales:
            raise ValueError(
                f"Length of ch_mult ({len(ch_mult)}) must match num_scales ({num_scales})"
            )

        tmp_net: nn.Module = UNet(
            ch,
            num_res_blocks,
            num_scales,
            ch_mult,
            self.input_channels,
            self.output_channels // cfg.data.S,
            scale_count_to_put_attn,
            data_min_max,
            dropout,
            skip_rescale,
            do_time_embed,
            time_scale_factor,
            time_embed_dim,
            class_embed_dim,  # Pass class_embed_dim to UNet
        ).to(device)
        if cfg.distributed:
            self.net: nn.Module = DDP(tmp_net, device_ids=[rank])
        else:
            self.net: nn.Module = tmp_net

        self.S: int = cfg.data.S
        self.data_shape: Tuple[int, int, int] = cfg.data.shape

    def forward(
        self,
        x: Float[torch.Tensor, "b d"],  # Flattened input: batch, flattened dimensions
        sigma: Float[torch.Tensor, "b"],  # Time values for each sample in batch
        mask: Optional[
            Float[torch.Tensor, "b"]
        ] = None,  # Binary mask (0 or 1) for conditional generation
        class_labels: Optional[
            Int[torch.Tensor, "b"]
        ] = None,  # Class labels when using conditional generation
    ) -> Float[
        torch.Tensor, "b d s"
    ]:  # Output logits: batch, flattened dimensions, states
        """
        Returns logits over state space for each pixel

        Args:
            x: Flattened input tensor of shape (batch, flattened_dimensions)
            sigma: Time values for each sample in batch
            mask: Optional binary mask (0 or 1) indicating whether to apply class conditioning
            class_labels: Optional class labels for conditional generation (required if mask has any 1s)

        Returns:
            Logits over state space for each pixel
        """
        B: int
        D: int
        C: int
        H: int
        W: int
        S: int
        B, _ = x.shape
        C, H, W = self.data_shape
        D = C * H * W
        S = self.S
        x: Float[torch.Tensor, "b c h w"] = x.view(B, self.input_channels, H, W)
        C_out = self.output_channels // self.S

        cls_embed = None
        if self.cfg_train and mask is not None and class_labels is not None:
            # Get embeddings for all class labels
            cls_embed: Float[torch.Tensor, "b class_embed_dim"] = self.cls_embed(
                class_labels
            )

        net_out: Float[torch.Tensor, "b 2c h w"] = self.net(
            x, sigma, cls_embed, mask
        )  # (B, 2*C, H, W)
        # Truncated logistic output from https://arxiv.org/pdf/2107.03006.pdf

        mu: Float[torch.Tensor, "b c h w 1"] = net_out[:, 0:C_out, :, :].unsqueeze(-1)
        log_scale: Float[torch.Tensor, "b c h w 1"] = net_out[
            :, C_out:, :, :
        ].unsqueeze(-1)

        inv_scale: Float[torch.Tensor, "b c h w 1"] = torch.exp(-(log_scale - 2))

        bin_width: float = 2.0 / self.S
        bin_centers: Float[torch.Tensor, "1 1 1 1 s"] = torch.linspace(
            start=-1.0 + bin_width / 2,
            end=1.0 - bin_width / 2,
            steps=self.S,
            device=self.device,
        ).view(1, 1, 1, 1, self.S)

        sig_in_left: Float[torch.Tensor, "b c h w s"] = (
            bin_centers - bin_width / 2 - mu
        ) * inv_scale
        bin_left_logcdf: Float[torch.Tensor, "b c h w s"] = F.logsigmoid(sig_in_left)
        sig_in_right: Float[torch.Tensor, "b c h w s"] = (
            bin_centers + bin_width / 2 - mu
        ) * inv_scale
        bin_right_logcdf: Float[torch.Tensor, "b c h w s"] = F.logsigmoid(sig_in_right)

        logits_1: Float[torch.Tensor, "b c h w s"] = self._log_minus_exp(
            bin_right_logcdf, bin_left_logcdf
        )
        logits_2: Float[torch.Tensor, "b c h w s"] = self._log_minus_exp(
            -sig_in_left + bin_left_logcdf, -sig_in_right + bin_right_logcdf
        )

        if self.fix_logistic:
            logits: Float[torch.Tensor, "b c h w s"] = torch.min(logits_1, logits_2)
        else:
            logits: Float[torch.Tensor, "b c h w s"] = logits_1

        logits: Float[torch.Tensor, "b d s"] = logits.view(B, D, S)

        if self.scaling_trick:
            logits = mutils.scaling_trick(logits, sigma)

        return logits

    def _log_minus_exp(
        self,
        a: Float[torch.Tensor, "..."],
        b: Float[torch.Tensor, "..."],
        eps: float = 1e-6,
    ) -> Float[torch.Tensor, "..."]:
        """
        Compute log (exp(a) - exp(b)) for (b<a)
        From https://arxiv.org/pdf/2107.03006.pdf
        """
        return a + torch.log1p(-torch.exp(b - a) + eps)


class ConfigTauLDR:
    """Configuration for training."""

    def __init__(
        self,
        dataS: int = 256,
        num_scales: int = 4,
        num_res_blocks: int = 3,
        ch_mult: List[int] = [1, 2, 2, 2],
        scaling_trick: bool = False,
        class_embed_dim: int = 32,
        input_channels: int = 1,
        output_channels: int = 1,
        data_shape: Tuple[int, int, int] = (1, 28, 28),
        ch: int = 128,
    ):
        # Model config
        self.model = type("", (), {})()
        self.model.fix_logistic = False
        self.model.ch = ch
        self.model.num_res_blocks = num_res_blocks
        self.model.num_scales = num_scales
        self.model.ch_mult = ch_mult
        self.model.input_channels = input_channels
        self.model.output_channels = (
            output_channels if output_channels is not None else input_channels
        )
        self.model.scale_count_to_put_attn = 1
        self.model.data_min_max = (0, 255)  # CIFAR-10 pixel range
        self.model.dropout = 0.1
        self.model.skip_rescale = True
        self.model.time_scale_factor = 1000.0
        self.model.time_embed_dim = self.model.ch
        self.model.scaling_trick = scaling_trick
        self.model.class_embed_dim = class_embed_dim
        self.data = type("", (), {})()
        self.data.shape = data_shape
        self.data.num_classes = 10
        self.data.S = dataS
        self.distributed = False  # Set to True to use DistributedDataParallel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42


def get_unet_tauldr(
    cfg: Optional[ConfigTauLDR] = None,
    rank: Optional[int] = None,
    cfg_train: bool = False,
):
    """
    Create a UNet model for tau-LDR.

    Args:
        cfg: Configuration object
        rank: Rank for distributed training
        cfg_train: Whether to enable class conditioning for training

    Returns:
        ImageX0PredBase model
    """
    if cfg is None:
        cfg = ConfigTauLDR()
    return ImageX0PredBase(cfg, cfg.device, rank, cfg_train)
