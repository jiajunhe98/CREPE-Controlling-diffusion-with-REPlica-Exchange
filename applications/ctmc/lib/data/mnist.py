import pathlib
from typing import Optional, Tuple

import torch
from jaxtyping import Array, Int
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2

import lib.data.utils as dutils


class DiscreteMNISTDataset(Dataset):
    def __init__(
        self,
        root: pathlib.Path,
        train: bool = True,
        download: bool = True,
        with_targets: bool = False,
    ) -> None:
        super().__init__()
        # Define transforms
        self.transforms = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.int32, scale=False), dutils.Flatten()]
        )

        # Create dataset with transforms
        self.ds = datasets.MNIST(
            root=root, train=train, download=download, transform=self.transforms
        )
        self.with_targets = with_targets

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: int) -> Int[Array, "h*w"]:
        # Get the transformed data
        data = self.ds[index][0]

        if self.with_targets:
            return data, self.ds[index][1]  # Return data and target
        else:
            return data  # Return only data
