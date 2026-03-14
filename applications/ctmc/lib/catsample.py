import torch
import torch.nn.functional as F
from jaxtyping import Float, Int


def gumbel_softmax(categorical_probs, hard=False, eps=1e-9):
    logits = categorical_probs.clamp(min=1e-9).log()
    return F.gumbel_softmax(logits, hard=hard)


def sample_categorical(
    categorical_probs: Float[torch.Tensor, "... d"], method="hard"
) -> Int[torch.Tensor, "..."]:
    """
    sample from a categorical distribution with probabilities categorical_probs using the gumbel trick.
    Sample according to the last dimension of the tensor.
    """
    if method == "hard":
        """
        sample from a categorical distribution with probabilities categorical_probs using the gumbel trick
        """
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(
            f"Method {method} for sampling categorical variables is not valid."
        )
