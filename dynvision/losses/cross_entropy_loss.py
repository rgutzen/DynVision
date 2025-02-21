from typing import Any, List

import torch
from torch.nn.modules import loss


class CrossEntropyLoss(loss.CrossEntropyLoss):

    def forward(self, output: List[Any], target: torch.Tensor):
        if isinstance(output, tuple):
            flat_outputs, *responses = output
        else:
            flat_outputs = output

        loss = super().forward(flat_outputs, target)

        return loss
