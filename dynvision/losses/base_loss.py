"""Base class for implementing custom loss functions.

This module provides a base class for implementing custom loss functions with
consistent patterns for input handling, validation, and error reporting.

Example:
    ```python
    class CustomLoss(BaseLoss):
        def __init__(self, reduction: str = 'mean'):
            super().__init__(reduction=reduction)
            
        def compute_loss(self, outputs, targets):
            # Implement custom loss computation
            return loss_value
    ```
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    """Base class for custom loss functions.

    This class provides common functionality for loss functions including:
    - Consistent output handling (flat outputs and responses)
    - Input validation
    - Error reporting
    - Reduction modes (mean, sum, none)

    Args:
        reduction: Specifies the reduction to apply to the output.
            'none': no reduction will be applied
            'mean': the sum of the output will be divided by the number of elements
            'sum': the output will be summed
    """

    valid_reductions = ["none", "mean", "sum"]

    def __init__(self, reduction: str = "mean") -> None:
        """Initialize the base loss function.

        Args:
            reduction: Reduction method to apply to the loss.
                Must be one of ['none', 'mean', 'sum'].

        Raises:
            ValueError: If reduction is not one of the valid options.
        """
        super().__init__()
        if reduction not in self.valid_reductions:
            raise ValueError(
                f"Invalid reduction '{reduction}'. "
                f"Must be one of {self.valid_reductions}"
            )
        self.reduction = reduction

    def extract_outputs(
        self,
        outputs: Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
        require_responses: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Extract flat outputs and optional responses from model output.

        Args:
            outputs: Model outputs, either a tensor or a tuple of (outputs, responses)
            require_responses: Whether responses are required for this loss

        Returns:
            Tuple containing:
                - Flat outputs tensor
                - Dictionary of responses (if present) or None

        Raises:
            ValueError: If responses are required but not provided
        """
        if isinstance(outputs, tuple):
            flat_outputs, *extra = outputs
            responses = extra[0] if extra else None
        else:
            flat_outputs = outputs
            responses = None

        if require_responses and responses is None:
            raise ValueError(
                f"{self.__class__.__name__} requires model responses but none were provided"
            )

        return flat_outputs, responses

    def validate_shapes(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        allow_broadcasting: bool = True,
    ) -> None:
        """Validate that output and target shapes are compatible.

        Args:
            outputs: Model outputs tensor
            targets: Target values tensor
            allow_broadcasting: Whether to allow tensor broadcasting

        Raises:
            ValueError: If shapes are incompatible
        """
        if not allow_broadcasting and outputs.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Output shape {outputs.shape} does not match target shape {targets.shape}"
            )

        if allow_broadcasting:
            try:
                # Check if broadcasting would work
                _ = torch.broadcast_tensors(outputs, targets)
            except RuntimeError:
                raise ValueError(
                    f"Output shape {outputs.shape} cannot be broadcast to target shape {targets.shape}"
                )

    def apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply the specified reduction across samples in batch to the loss value.

        Args:
            loss: Unreduced loss tensor

        Returns:
            Reduced loss value according to the reduction method
        """
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # 'none' reduction

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        responses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute the loss value.

        This method should be implemented by subclasses to compute their specific loss.

        Args:
            outputs: Model outputs tensor
            targets: Target values tensor
            responses: Optional dictionary of model responses

        Returns:
            Computed loss value

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"Loss computation not implemented for {self.__class__.__name__}"
        )

    def forward(
        self,
        outputs: Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass to compute the loss.

        Args:
            outputs: Model outputs, either a tensor or a tuple of (outputs, responses)
            targets: Target values tensor

        Returns:
            Computed loss value
        """
        # Extract outputs and responses
        flat_outputs, responses = self.extract_outputs(
            outputs, require_responses=getattr(self, "requires_responses", False)
        )

        # Validate shapes
        self.validate_shapes(
            flat_outputs,
            targets,
            allow_broadcasting=getattr(self, "allow_broadcasting", True),
        )

        # Compute and reduce loss
        loss = self.compute_loss(
            outputs=flat_outputs, targets=targets, responses=responses
        )
        return self.apply_reduction(loss)
