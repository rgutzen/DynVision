import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

__all__ = ["SupraLinearity"]


class SupraLinearity(nn.Module):
    """
    A module that applies a supra-linear transformation to the input by raising it to the power of n.
    See Rubin, van Hooser, Miller (2015) doi:10.1016/j.neuron.2014.12.026 for more details.
    """

    def __init__(
        self, n: float = 1.8, k: float = 0.04, requires_grad: bool = False
    ) -> None:
        super(SupraLinearity, self).__init__()

        self.n = nn.Parameter(
            torch.tensor(n, dtype=float), requires_grad=requires_grad
        )

        self.k = nn.Parameter(
            torch.tensor(k, dtype=float), requires_grad=requires_grad
        )

    def forward(self, x: Optional[Tensor] = None) -> Optional[Tensor]:
        if x is None:
            return None

        if torch.any(x < 0):
            return torch.pow(torch.abs(x), self.n) * torch.sign(x) * self.k
        else:
            return torch.pow(x, self.n) * self.k
