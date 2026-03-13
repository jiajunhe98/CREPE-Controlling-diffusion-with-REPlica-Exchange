import torch
from jaxtyping import Float


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.

    Returns:
        A model function.
    """

    def model_fn(
        x: Float[torch.Tensor, "batch *datadim"],
        sigma: Float[torch.Tensor, "batch"],
        **kwargs,
    ) -> Float[torch.Tensor, "batch *datadim vocab"]:
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data of shape [batch, *datadim] where *datadim is the shape of the data.
            sigma: A mini-batch of noise levels of shape [batch].
            **kwargs: Additional arguments (e.g., class_labels, mask for classifier-free guidance).
        Returns:
            A mini-batch of model outputs of shape [batch, *datadim, vocab].
        """
        if train:
            model.train()
        else:
            model.eval()

        # output the raw values (we handle mlm training in losses.py)
        return model(x, sigma, **kwargs)

    return model_fn


def get_finetune_model_fn(model, train=False):
    """Create a function to give the output of the FinetuneSEDD model.

    Args:
        model: The FinetuneSEDD score model.
        train: `True` for training and `False` for evaluation.

    Returns:
        A model function.
    """

    def model_fn(x, cond, sigma):
        """Compute the output of the FinetuneSEDD model.

        Args:
            x: A mini-batch of input data.
            cond: A mini-batch of conditioning variables.
            sigma: A mini-batch of noise levels.

        Returns:
            A mini-batch of model outputs.
        """
        if train:
            model.train()
        else:
            model.eval()

        # FinetuneSEDD signature: model(x, cond, sigma)
        return model(x, cond, sigma)

    return model_fn


def get_finetune_score_fn(model, train=False, sampling=False):
    model_fn = get_finetune_model_fn(model, train=train)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):

        def score_fn(x, cond, t, sigma, with_aux=False):
            sigma = sigma.reshape(-1)
            score = model_fn(x, cond, sigma)

            if sampling:
                # when sampling return true score (not log used for training)
                return score.exp()

            return score

    return score_fn


def get_score_fn(model, train=False, sampling=False):
    """Create a function to give the score output of the model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        sampling: `True` when sampling (returns exp of raw scores).

    Returns:
        A score function.
    """
    if sampling:
        assert not train, "Must sample in eval mode"

    model_fn = get_model_fn(model, train=train)

    def score_fn(
        x: Float[torch.Tensor, "batch *datadim"],
        sigma: Float[torch.Tensor, "batch"],
        **kwargs,
    ) -> Float[torch.Tensor, "batch *datadim vocab"]:
        """Compute the score output of the model.

        Args:
            x: A mini-batch of input data of shape [batch, *datadim].
            sigma: A mini-batch of noise levels of shape [batch].
            **kwargs: Additional arguments (e.g., class_labels, mask for classifier-free guidance).
        Returns:
            A mini-batch of score outputs of shape [batch, *datadim, vocab].
        """
        sigma = sigma.reshape(-1)
        score = model_fn(x, sigma, **kwargs)

        if sampling:
            # when sampling return true score (not log used for training)
            return score.exp()

        return score

    return score_fn


def get_score_discdiff_fn(
    uncond_model, cond_model, cfg_temperature=1.0, train=False, sampling=True
):
    assert not train, "Must sample in eval mode"
    assert sampling, "Must sample in sampling mode"
    uncond_model_fn = lambda x, sigma: uncond_model(x, sigma)
    cond_model_fn = lambda x, sigma, cond: cond_model(x, sigma, cond)

    def score_fn(
        x: Float[torch.Tensor, "batch *datadim"],
        cond: torch.Tensor,
        sigma: Float[torch.Tensor, "batch"],
        with_aux: bool = False,
    ) -> Float[torch.Tensor, "batch *datadim vocab"]:
        sigma = sigma.reshape(-1)
        batch_size = x.shape[0]

        # Get conditional score (with class labels)
        cond_logits = cond_model_fn(x.long(), sigma, cond)

        cond_score = scaling_trick(cond_logits, sigma)

        # Get unconditional score (no class labels)
        uncond_logits = uncond_model_fn(x.long(), sigma)
        uncond_score = scaling_trick(uncond_logits, sigma)

        # Apply classifier-free guidance
        score = cfg_temperature * cond_score + (1.0 - cfg_temperature) * uncond_score

        if sampling:
            # when sampling return true score (not log used for training)
            score = score.exp()
            if with_aux:
                uncond_score = uncond_score.exp()
                cond_score = cond_score.exp()

        if with_aux:
            return score, (uncond_score, cond_score)

        return score

    return score_fn


def scaling_trick(
    raw_logits: Float[torch.Tensor, "batch *datadim vocab"],
    sigma: Float[torch.Tensor, "batch"],
) -> Float[torch.Tensor, "batch *datadim vocab"]:
    """
    Apply the critical scaling transformation for absorbing schedules.

    Based on Theorem 1 from Jingyang Ou et al. (2024): https://arxiv.org/pdf/2406.03736

    The transformation converts raw model outputs to properly scaled scores:
    s_θ(x_t, t) = (e^(-σ̄(t)) / (1 - e^(-σ̄(t)))) * s̃_θ(x_t, t)

    In log space:
    log s_θ(x_t, t) = -σ̄(t) - log(1 - e^(-σ̄(t))) + log s̃_θ(x_t, t)

    Args:
        raw_logits: Raw model outputs of shape [batch, *datadim, vocab]
        sigma: Cumulative noise σ̄(t) of shape [batch]

    Returns:
        Scaled score function of shape [batch, *datadim, vocab]
    """
    # Expand sigma to broadcast with raw_logits: [batch] -> [batch, 1, 1, ...]
    sigma_expanded = sigma[..., None, None]

    scaled_score = (
        -torch.log1p(-torch.exp(-sigma_expanded))  # -log(1 - e^(-σ))
        - sigma_expanded  # -σ
        + raw_logits  # + log s̃_θ(x_t, t)
    )

    return scaled_score


def get_cfg_score_fn(
    uncond_model, cond_model, cfg_temperature=1.0, train=False, sampling=False
):
    """Create a classifier-free guidance score function.

    Args:
        uncond_model: The unconditional score model.
        cond_model: The conditional score model.
        cfg_temperature: Strength of classifier-free guidance (default: 1.0).
        train: `True` for training and `False` for evaluation.
        sampling: `True` when sampling (returns exp of raw scores).

    Returns:
        A classifier-free guidance score function.
    """
    if sampling:
        assert not train, "Must sample in eval mode"

    uncond_model_fn = get_model_fn(uncond_model, train=train)
    cond_model_fn = get_model_fn(cond_model, train=train)

    def score_fn(
        x: Float[torch.Tensor, "batch *datadim"],
        cond: torch.Tensor,
        sigma: Float[torch.Tensor, "batch"],
        with_aux: bool = False,
    ) -> Float[torch.Tensor, "batch *datadim vocab"]:
        """Compute the classifier-free guidance score output.

        Args:
            x: A mini-batch of input data of shape [batch, *datadim].
            cond: Class labels for conditional generation.
            sigma: A mini-batch of noise levels of shape [batch].
            with_aux: Whether to return auxiliary outputs (uncond_score, cond_score).

        Returns:
            A mini-batch of score outputs with CFG applied.
        """
        sigma = sigma.reshape(-1)
        batch_size = x.shape[0]

        # Get conditional score (with class labels)
        mask_cond = torch.ones(batch_size, device=x.device)
        cond_score = cond_model_fn(x, sigma, class_labels=cond, mask=mask_cond)

        # Get unconditional score (no class labels)
        uncond_score = uncond_model_fn(x, sigma)

        # Apply classifier-free guidance
        score = cfg_temperature * cond_score + (1.0 - cfg_temperature) * uncond_score

        if sampling:
            # when sampling return true score (not log used for training)
            score = score.exp()
            if with_aux:
                uncond_score = uncond_score.exp()
                cond_score = cond_score.exp()

        if with_aux:
            return score, (uncond_score, cond_score)

        return score

    return score_fn


def get_cfg_score_finetune_fn(
    uncond_model, cond_model, cfg_temperature=1.0, train=False, sampling=False
):
    """Create a classifier-free guidance score function for FinetuneSEDD models.

    Args:
        uncond_model: The unconditional score model (base SEDD with signature: model(x, sigma)).
        cond_model: The conditional score model (FinetuneSEDD with signature: model(x, cond, sigma)).
        cfg_temperature: Strength of classifier-free guidance (default: 1.0).
        train: `True` for training and `False` for evaluation.
        sampling: `True` when sampling (returns exp of raw scores).

    Returns:
        A classifier-free guidance score function for FinetuneSEDD models.
    """
    if sampling:
        assert not train, "Must sample in eval mode"

    uncond_model_fn = get_model_fn(uncond_model, train=train)
    cond_model_fn = get_finetune_model_fn(cond_model, train=train)

    def score_fn(
        x: Float[torch.Tensor, "batch *datadim"],
        cond: torch.Tensor,
        sigma: Float[torch.Tensor, "batch"],
        with_aux: bool = False,
    ) -> Float[torch.Tensor, "batch *datadim vocab"]:
        """Compute the classifier-free guidance score output for FinetuneSEDD.

        Args:
            x: A mini-batch of input data of shape [batch, *datadim].
            cond: Conditioning variable for conditional generation.
            sigma: A mini-batch of noise levels of shape [batch].
            with_aux: Whether to return auxiliary outputs (uncond_score, cond_score).

        Returns:
            A mini-batch of score outputs with CFG applied.
        """
        sigma = sigma.reshape(-1)

        # Get conditional score using FinetuneSEDD signature: model(x, cond, sigma)
        cond_score = cond_model_fn(x, cond, sigma)

        # Get unconditional score using base SEDD signature: model(x, sigma)
        uncond_score = uncond_model_fn(x, sigma)

        # Apply classifier-free guidance
        score = cfg_temperature * cond_score + (1.0 - cfg_temperature) * uncond_score

        if sampling:
            # when sampling return true score (not log used for training)
            score = score.exp()
            if with_aux:
                uncond_score = uncond_score.exp()
                cond_score = cond_score.exp()

        if with_aux:
            return score, (uncond_score, cond_score)

        return score

    return score_fn


def get_deft_score_fn(
    uncond_model,
    cond_model,
    denoise_fn,
    deft_temperature=1.0,
    train=False,
    sampling=False,
):
    """Create a classifier-free guidance score function.

    Args:
        uncond_model: The unconditional score model.
        cond_model: The conditional score model.
        denoise_fn: The denoising function.
        deft_temperature: Strength of the proposal (default: 1.0).
        train: `True` for training and `False` for evaluation (only applied to cond_model).
        sampling: `True` when sampling (returns exp of raw scores).

    Returns:
        A DEFT score function.
    """
    if sampling:
        assert not train, "Must sample in eval mode"

    uncond_model_fn = get_model_fn(uncond_model, train=False)
    cond_model_fn = get_model_fn(cond_model, train=train)

    def score_fn(
        x: Float[torch.Tensor, "batch *datadim"],
        cond: torch.Tensor,
        sigma: Float[torch.Tensor, "batch"],
        with_aux: bool = False,
    ) -> Float[torch.Tensor, "batch *datadim vocab"]:
        """Compute the DEFT score output.

        Args:
            x: A mini-batch of input data of shape [batch, *datadim].
            cond: Class labels for conditional generation.
            sigma: A mini-batch of noise levels of shape [batch].
            with_aux: Whether to return auxiliary outputs (uncond_score, cond_score).

        Returns:
            A mini-batch of score outputs with DEFT applied.
        """
        sigma = sigma.reshape(-1)
        approximate_x0 = denoise_fn(x, sigma)

        # Concatenate x and approximate_x0
        x_concat = torch.cat([x, approximate_x0], dim=-1)

        cond_score = cond_model_fn(
            x_concat,
            sigma,
            class_labels=cond,
            mask=torch.ones(cond.shape[0], device=x.device),
        )

        uncond_score = uncond_model_fn(x, sigma)

        score = deft_temperature * cond_score + uncond_score

        if sampling:
            # when sampling return true score (not log used for training)
            score = score.exp()
            if with_aux:
                uncond_score = uncond_score.exp()
                cond_score = cond_score.exp()

        if with_aux:
            return score, (uncond_score, cond_score)

        return score

    return score_fn


def create_ema_model(
    model: torch.nn.Module, ema_decay: float = 0.999, device: str = "cuda"
) -> torch.optim.swa_utils.AveragedModel:
    """
    Create an Exponential Moving Average (EMA) version of the model for inference.

    Args:
        model: The neural network model to create EMA for
        ema_decay: Decay rate for exponential moving average (default: 0.999)
        device: Device to place the EMA model on

    Returns:
        EMA model that can be updated with ema_model.update_parameters(model)
    """
    ema_model = torch.optim.swa_utils.AveragedModel(
        model,
        device=device,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay),
        use_buffers=True,
    )
    return ema_model
