import math
from typing import Any, List

import torch
from torch.nn.modules import loss


class EnergyLoss(loss._Loss):
    """
    Calculates the average activation per unit.
    """

    def __init__(self):
        self.norm_factors = None
        super(EnergyLoss, self).__init__()

    def calculate_norm_factor(self, responses: dict):
        self.norm_factors = torch.ones(len(responses))
        for i, layer_response in enumerate(responses.values()):
            if isinstance(layer_response, list):
                # only select response from most recent batch
                layer_response = layer_response[-1]

            n_timeunits = math.prod(layer_response.shape[1:])
            self.norm_factors[i] = n_timeunits

    def forward(self, output: List[Any], target: torch.Tensor):
        if not isinstance(output, tuple):
            raise ValueError("EnergyLoss requires a tuple of outputs and responses!")

        flat_outputs, responses = output

        if self.norm_factors is None:
            self.calculate_norm_factor(responses)

        energies_per_unit = torch.zeros(len(responses), device=target.device)

        for i, layer_response in enumerate(responses.values()):
            if isinstance(layer_response, list):
                # only select response from most recent batch
                layer_response = layer_response[-1]

            # Compute the L2 norm and normalize by the precomputed factor
            energies_per_unit[i] = (
                torch.norm(layer_response, p=2) / self.norm_factors[i]
            )

        # Compute the mean energy loss and normalize by batch size
        batch_size = layer_response.shape[0]
        energy_loss = energies_per_unit.mean() / batch_size

        return energy_loss
