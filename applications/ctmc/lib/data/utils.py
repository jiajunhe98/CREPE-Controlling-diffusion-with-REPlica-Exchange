import torch
from jaxtyping import Int


class Flatten:
    def __call__(self, x: Int[torch.Tensor, "c h w"]) -> Int[torch.Tensor, "c*h*w"]:
        return torch.flatten(x)
