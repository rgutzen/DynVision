"""Cross Entropy Loss implementation with support for model responses.

This module provides a CrossEntropyLoss implementation that handles both standard
model outputs and models that return additional response information.

Example:
    ```python
    criterion = CrossEntropyLoss()
    
    # With standard output
    loss = criterion(model_output, targets)
    
    # With model responses
    loss = criterion((model_output, responses), targets)
    ```
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .base_loss import BaseLoss


class CrossEntropyLoss(BaseLoss):
    """Cross entropy loss that handles both standard outputs and model responses.

    This implementation extends the standard cross entropy loss to handle models
    that return both predictions and internal responses. It properly extracts the
    predictions while preserving the response information for other uses.

    Args:
        reduction: Specifies the reduction to apply to the output.
            'none': no reduction will be applied
            'mean': the sum of the output will be divided by the number of elements
            'sum': the output will be summed
        weight: A manual rescaling weight given to each class.
            If given, has to be a Tensor of size C
        ignore_index: Specifies a target value that is ignored and does not
            contribute to the input gradient. When size_average is True,
            the loss is averaged over non-ignored targets.
    """

    def __init__(
        self,
        reduction: str = "mean",
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> None:
        """Initialize the cross entropy loss function.

        Args:
            reduction: Reduction method to apply to the loss
            weight: Optional class weights
            ignore_index: Target value to ignore
        """
        super().__init__(reduction=reduction)
        self.weight = weight
        self.ignore_index = ignore_index
        self.allow_broadcasting = False  # Cross entropy requires exact shape match

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        responses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute the cross entropy loss.

        Args:
            outputs: Predicted class scores (logits)
            targets: Ground truth class indices
            responses: Optional model responses (unused)

        Returns:
            Computed cross entropy loss

        Note:
            The responses parameter is included for API compatibility but is not
            used in the cross entropy calculation.
        """
        return F.cross_entropy(
            outputs,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
