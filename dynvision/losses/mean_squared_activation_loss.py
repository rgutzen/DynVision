"""Mean Squared Activation Loss implementation.

This module provides a loss function that computes the mean squared error between
network activations and a target activation level. This can be used to:
- Regularize network activations towards a specific level
- Implement activation targeting objectives
- Control network activity levels

Example:
    ```python
    # Target activation level of 0.5
    criterion = MeanSquaredActivationLoss(target_activation=0.5)
    loss = criterion(model_outputs, targets)
    ```
"""

from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F

from .base_loss import BaseLoss


class MeanSquaredActivationLoss(BaseLoss):
    """Computes MSE between network activations and a target level.
    
    This loss calculates the mean squared error between network activations
    and a specified target activation level. It can be used to encourage
    specific activation patterns or implement regularization schemes.

    Args:
        target_activation: Target activation level to encourage
        reduction: Specifies the reduction to apply to the output.
            'none': no reduction will be applied
            'mean': the sum of the output will be divided by the number of elements
            'sum': the output will be summed
    """

    def __init__(
        self,
        target_activation: Union[float, torch.Tensor] = 0.0,
        reduction: str = 'mean'
    ) -> None:
        """Initialize the mean squared activation loss.

        Args:
            target_activation: Target activation level (default: 0.0)
            reduction: Reduction method to apply to the loss
        """
        super().__init__(reduction=reduction)
        self.target_activation = target_activation

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        responses: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the mean squared activation loss.

        Args:
            outputs: Network output activations
            targets: Target values (unused in this loss)
            responses: Optional model responses (unused in this loss)

        Returns:
            Computed MSE loss between outputs and target activation level

        Note:
            This loss ignores the targets parameter and uses the target_activation
            value instead. The targets parameter is included for API compatibility.
        """
        if isinstance(self.target_activation, (int, float)):
            target_values = torch.ones_like(outputs) * self.target_activation
        else:
            target_values = self.target_activation.to(outputs.device)

        return F.mse_loss(outputs, target_values, reduction='none')