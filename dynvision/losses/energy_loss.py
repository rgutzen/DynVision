import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Tuple
from .base_loss import BaseLoss


class EnergyLoss(BaseLoss):
    """Energy loss that computes statistics during forward pass using hooks.

    This loss accumulates energy across all timesteps during the forward pass
    and normalizes by the total number of timesteps when reduction='mean'.
    Unlike CrossEntropyLoss which only considers valid (non-masked) timesteps,
    EnergyLoss computes total computational energy across all timesteps including
    null inputs and reaction windows, as energy consumption occurs regardless of
    label validity.

    The hooks fire once per monitored layer per timestep during the model's
    forward pass. Energy is accumulated across all hook calls and then normalized
    by the number of timesteps when compute_loss() is called.
    """

    def __init__(self, reduction: str = "mean", p: int = 1) -> None:
        super().__init__(reduction=reduction)
        self.requires_responses = False  # We don't need stored responses!
        self.allow_broadcasting = True
        self.batch_energy = {}
        self.hooks = []
        self.norm_factors = {}
        self.p = p
        self._last_device: Optional[torch.device] = None
        self._last_dtype: Optional[torch.dtype] = None
        self._hook_call_count = {}  # Track hook calls per module to infer n_timesteps

    def register_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on model modules to capture energy statistics."""
        self.remove_hooks()  # Clean up any existing hooks

        for name, module in model.named_modules():
            if self._should_monitor_module(module):
                hook = module.register_forward_hook(
                    lambda module, input, output, name=name: self._accumulate_energy(
                        name, output
                    )
                )
                self.hooks.append(hook)

    def _should_monitor_module(self, module: nn.Module) -> bool:
        """Determine which modules to monitor for energy calculation."""
        # Monitor conv modules, linear modules, but skip activations, pooling, etc.
        return isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d))

    def _accumulate_energy(self, module_name: str, activation: torch.Tensor) -> None:
        """Accumulate energy for a module across timesteps during forward pass.

        This method is called via hooks once per layer per timestep. Energy is
        accumulated across all timesteps to compute the total computational cost.
        """
        if activation is None:
            return

        # Calculate energy for this timestep
        batch_energy = torch.norm(
            activation, p=self.p, dim=tuple(range(1, activation.ndim))
        )

        if module_name not in self.norm_factors:
            # Calculate normalization factor once
            n_units = activation.shape[1:].numel()  # All dims except batch
            self.norm_factors[module_name] = n_units ** (1 / self.p)

        # Accumulate energy across timesteps instead of overwriting
        if module_name not in self.batch_energy:
            self.batch_energy[module_name] = batch_energy
            self._hook_call_count[module_name] = 1
        else:
            self.batch_energy[module_name] = self.batch_energy[module_name] + batch_energy
            self._hook_call_count[module_name] += 1

        self._last_device = batch_energy.device
        self._last_dtype = batch_energy.dtype

    def forward(
        self,
        outputs: Union[
            torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        ] = None,
        targets: torch.Tensor = None,
    ) -> torch.Tensor:

        loss = self.compute_loss()

        return self.apply_reduction(loss)

    def compute_loss(
        self,
        outputs: torch.Tensor = None,
        targets: torch.Tensor = None,
        responses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute energy loss from accumulated statistics across timesteps.

        The accumulated energy is normalized by the number of timesteps (inferred
        from hook call counts) and the number of monitored modules, then averaged
        over the batch dimension by apply_reduction.
        """
        if not self.batch_energy:
            device = self._last_device if self._last_device is not None else None
            dtype = self._last_dtype if self._last_dtype is not None else torch.float32
            zero = torch.zeros(1, device=device, dtype=dtype)
            return zero

        # Infer number of timesteps from hook call counts
        # All modules should have been called the same number of times (n_timesteps)
        n_timesteps = 1
        if self._hook_call_count:
            call_counts = list(self._hook_call_count.values())
            if len(set(call_counts)) > 1:
                # Warning: inconsistent hook call counts (shouldn't happen normally)
                pass
            n_timesteps = max(call_counts) if call_counts else 1

        total_energy: Optional[torch.Tensor] = None
        module_count = 0

        for module_name, batch_energy in self.batch_energy.items():
            # Get the accumulated energy and normalize by spatial dimensions
            norm_factor = self.norm_factors[module_name]

            # Ensure gradients flow through
            if batch_energy.requires_grad:
                # Normalize by spatial dimensions
                normalized_energy = batch_energy / norm_factor

                if total_energy is None:
                    total_energy = normalized_energy
                else:
                    total_energy = total_energy + normalized_energy
                module_count += 1

        # Normalize by number of timesteps and number of modules
        if module_count > 0 and total_energy is not None:
            # Average across modules and timesteps
            loss = total_energy / (module_count * n_timesteps)
        else:
            device = self._last_device if self._last_device is not None else None
            dtype = self._last_dtype if self._last_dtype is not None else torch.float32
            loss = torch.zeros(1, device=device, dtype=dtype)

        # Clear accumulators for next batch
        self.batch_energy = {}
        self._hook_call_count = {}

        return loss

    def remove_hooks(self) -> None:
        """Clean up registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        """Ensure hooks are cleaned up."""
        self.remove_hooks()
