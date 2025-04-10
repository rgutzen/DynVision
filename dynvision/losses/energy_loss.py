"""Energy Loss implementation for neural network activations.

This module provides an implementation of the Energy Loss, which calculates
the average activation energy per unit across all layers of a neural network.
This can be used to regularize network activations or implement specific
energy-based learning objectives.

Example:
    ```python
    criterion = EnergyLoss(reduction='mean')
    loss = criterion((outputs, responses), targets)
    ```
"""

import math
from typing import Dict, Optional

import torch

from .base_loss import BaseLoss


class EnergyLoss(BaseLoss):
    """Computes the average activation energy per unit across network layers.
    
    This loss calculates the L2 norm of activations for each layer, normalized by
    the number of units in that layer and the batch size. It can be used to:
    - Regularize network activations
    - Implement energy-based learning objectives
    - Monitor network activity levels
    
    The loss requires model responses (layer activations) to be provided along
    with the model outputs.

    Args:
        reduction: Specifies the reduction to apply to the output.
            'none': no reduction will be applied
            'mean': the sum of the output will be divided by the number of elements
            'sum': the output will be summed
    """

    def __init__(self, reduction: str = 'mean') -> None:
        """Initialize the energy loss function.

        Args:
            reduction: Reduction method to apply to the loss
        """
        super().__init__(reduction=reduction)
        self.requires_responses = True
        self.norm_factors = None

    def calculate_norm_factors(self, responses: Dict[str, torch.Tensor]) -> None:
        """Calculate normalization factors for each layer.

        This precomputes normalization factors based on the number of units in
        each layer to ensure fair comparison between layers of different sizes.

        Args:
            responses: Dictionary of layer responses
        """
        self.norm_factors = torch.ones(len(responses))
        for i, layer_response in enumerate(responses.values()):
            if isinstance(layer_response, list):
                # Only use the most recent batch response
                layer_response = layer_response[-1]
            
            # Calculate total number of units in the layer
            n_units = math.prod(layer_response.shape[1:])
            self.norm_factors[i] = n_units

    def compute_layer_energy(
        self,
        response: torch.Tensor,
        norm_factor: float
    ) -> torch.Tensor:
        """Compute the normalized energy for a single layer.

        Args:
            response: Layer activation tensor
            norm_factor: Normalization factor for the layer

        Returns:
            Normalized layer energy
        """
        # Compute the L2 norm and normalize by the precomputed factor
        return torch.norm(response, p=2) / norm_factor

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        responses: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the energy loss across all layers.

        Args:
            outputs: Model outputs (unused in energy calculation)
            targets: Target values (unused in energy calculation)
            responses: Dictionary of layer responses

        Returns:
            Computed energy loss

        Note:
            This loss only uses the responses and ignores the outputs and targets.
            They are included in the signature for API compatibility.
        """
        if self.norm_factors is None:
            self.calculate_norm_factors(responses)

        energies_per_layer = torch.zeros(
            len(responses),
            device=outputs.device
        )

        for i, layer_response in enumerate(responses.values()):
            if isinstance(layer_response, list):
                # Only use the most recent batch response
                layer_response = layer_response[-1]

            energies_per_layer[i] = self.compute_layer_energy(
                layer_response,
                self.norm_factors[i]
            )

        # Normalize by batch size before reduction
        batch_size = next(iter(responses.values()))[-1].shape[0]
        return energies_per_layer.mean() / batch_size