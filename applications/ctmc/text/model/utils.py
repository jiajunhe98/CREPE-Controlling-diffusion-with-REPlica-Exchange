import torch
import torch.nn.functional as F


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        mlm: If the input model is a mlm and models the base probability

    Returns:
        A model function.
    """

    def model_fn(x, sigma):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
              for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if train:
            model.train()
        else:
            model.eval()

            # otherwise output the raw values (we handle mlm training in losses.py)
        return model(x, sigma)

    return model_fn


def get_finetune_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        mlm: If the input model is a mlm and models the base probability

    Returns:
        A model function.
    """

    def model_fn(x, cond, sigma):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
              for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if train:
            model.train()
        else:
            model.eval()

            # otherwise output the raw values (we handle mlm training in losses.py)
        return model(x, cond, sigma)

    return model_fn


def get_deft_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        mlm: If the input model is a mlm and models the base probability

    Returns:
        A model function.
    """

    def model_fn(x, x0, cond, sigma):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
              for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if train:
            model.train()
        else:
            model.eval()

            # otherwise output the raw values (we handle mlm training in losses.py)
        return model(x, x0, cond, sigma)

    return model_fn


def get_score_fn(model, train=False, sampling=False):
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):

        def score_fn(x, sigma):
            sigma = sigma.reshape(-1)
            score = model_fn(x, sigma)

            if sampling:
                # when sampling return true score (not log used for training)
                return score.exp()

            return score

    return score_fn


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


def get_cfg_score_fn(
    uncond_model, cond_model, strength=1.0, train=False, sampling=False
):
    model_fn = get_model_fn(uncond_model, train=train)
    cond_model_fn = get_finetune_model_fn(cond_model, train=train)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):

        def score_fn(x, cond, t, sigma, with_aux=False):
            sigma = sigma.reshape(-1)
            cond_score = cond_model_fn(x, cond, sigma)
            uncond_score = model_fn(x, sigma)
            score = strength * cond_score + (1.0 - strength) * uncond_score

            if sampling:
                score = score.exp()

            if with_aux:
                return score, (uncond_score, cond_score)

            return score

    return score_fn


def get_deft_score_fn(
    uncond_model, cond_model, denoise_fn, strength=1.0, train=False, sampling=False
):
    if sampling:
        assert not train, "Must sample in eval mode"
    uncond_fn = get_model_fn(uncond_model, train=False)
    cond_fn = get_deft_model_fn(cond_model, train=train)

    def score_fn(x, cond, t, sigma, with_aux=False):
        sigma = sigma.reshape(-1)
        if len(t.shape) < 2:
            t = t[:, None]
        x0 = denoise_fn(x, t)
        uncond_score = uncond_fn(x, sigma)
        cond_score = cond_fn(x, x0, cond, sigma)
        score = uncond_score + strength * cond_score

        if sampling:
            score = score.exp()

        if with_aux:
            return score, (uncond_score, uncond_score + cond_score)
        else:
            return score

    return score_fn
