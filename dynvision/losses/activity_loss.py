import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Tuple
from .base_loss import BaseLoss

ACTIVITY_MONITOR_ATTR = "_activity_monitor"
"""Attribute name used to mark modules for activity loss monitoring.

Set this attribute on any nn.Module to include it in the ActivityLoss computation.
The value controls how activity is measured:

- ``"absolute"``: L-p norm of activations, normalized by number of units.
  Penalizes total firing rate regardless of sign. Equivalent to the original
  activity loss behavior. Typically applied to post-nonlinearity activations.

- ``"signed"``: Magnitude of the mean activation (allows sign cancellation).
  Near-zero when excitatory and inhibitory drives are balanced (EI balance),
  large when activity is strongly one-sided. Apply to pre-nonlinearity outputs
  (e.g., ``tstep_{layer}`` or ``layer_{layer}``) for a biologically meaningful
  EI balance regularizer.

Examples::

    # Post-nonlinearity absolute activity (original behavior)
    model.nonlin_layer0._activity_monitor = "absolute"

    # Pre-nonlinearity EI balance (apply to tstep or main layer output)
    model.tstep_layer0._activity_monitor = "signed"

    # Recurrent-connection EI balance (apply to recurrent layer directly)
    model.layer0._activity_monitor = "signed"
"""


class ActivityLoss(BaseLoss):
    """Activity loss that measures neural activity during the forward pass using hooks.

    Accumulates activity across all timesteps during the forward pass and
    normalizes by the total number of timesteps when reduction='mean'.
    Unlike CrossEntropyLoss which only considers valid (non-masked) timesteps,
    ActivityLoss computes activity across all timesteps including null inputs and
    reaction windows, as neural activity occurs regardless of label validity.

    The hooks fire once per monitored module per timestep during the model's
    forward pass. Activity is accumulated across all hook calls and then normalized
    by the number of timesteps when compute_loss() is called.

    Modules are opted in by setting the ``_activity_monitor`` attribute (see
    :data:`ACTIVITY_MONITOR_ATTR`). Two modes are supported:

    - ``"absolute"``: L-p norm, normalized by number of units. Measures total
      firing rate. Apply to post-nonlinearity activations.

    - ``"signed"``: Mean activation magnitude (allows EI cancellation). Measures
      net activity imbalance: zero for perfect EI balance, large for imbalanced
      activity. Apply to pre-nonlinearity or recurrent-layer outputs.

    Note: Idle timesteps (when model._in_idle_period is True) are skipped from
    activity accumulation, as they are used only to bring the network into a
    stable state and should not contribute to the loss.
    """

    def __init__(self, reduction: str = "mean", ord: int = 1) -> None:
        super().__init__(reduction=reduction)
        self.requires_responses = False
        self.allow_broadcasting = True
        self.batch_activity = {}
        self.hooks = []
        self._norm_factors = (
            {}
        )  # cached per-module normalization factors (absolute mode)
        self.ord = ord
        self._last_device: Optional[torch.device] = None
        self._last_dtype: Optional[torch.dtype] = None
        self._hook_call_count = {}  # track hook calls per module to infer n_timesteps
        self._model_ref = None  # reference to model for checking idle period

    def register_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on marked modules to capture activity statistics."""
        self.remove_hooks()
        self._model_ref = model

        for name, module in model.named_modules():
            mode = getattr(module, ACTIVITY_MONITOR_ATTR, None)
            if mode is not None:
                hook = module.register_forward_hook(
                    lambda module, input, output, name=name, mode=mode: (
                        self._accumulate_activity(name, output, mode)
                    )
                )
                self.hooks.append(hook)

    def _should_monitor_module(self, module: nn.Module) -> bool:
        """Return True if the module is marked for activity monitoring."""
        # return getattr(
        #     module, ACTIVITY_MONITOR_ATTR, None
        # ) is not None or self._is_nonlinearity(module)
        return isinstance(
            module,
            (
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.Linear,
            ),
        )

    def _is_nonlinearity(self, module: nn.Module) -> bool:
        """Return True if the module is a nonlinearity (activation function)."""
        return isinstance(
            module,
            (
                nn.ReLU,
                nn.LeakyReLU,
                nn.ELU,
                nn.SELU,
                nn.PReLU,
                nn.GELU,
                nn.Sigmoid,
                nn.Tanh,
                nn.SiLU,
                nn.Mish,
                nn.Softplus,
                nn.Hardtanh,
                nn.Hardshrink,
                nn.Softshrink,
                nn.Softsign,
            ),
        )

    def _accumulate_activity(
        self, module_name: str, activation: torch.Tensor, mode: str
    ) -> None:
        """Accumulate activity for a module across timesteps during forward pass.

        Called via hooks once per marked module per timestep. Activity is accumulated
        across all timesteps; normalization happens in compute_loss().

        Args:
            module_name: Name of the module (used as accumulator key).
            activation: Output tensor of the module, shape (batch, ...).
            mode: Activity measurement mode, either ``"absolute"`` or ``"signed"``.
        """
        if activation is None:
            return

        # Skip accumulation during idle period
        if self._model_ref is not None and getattr(
            self._model_ref, "_in_idle_period", False
        ):
            return

        spatial_dims = tuple(range(1, activation.ndim))

        if mode == "signed":
            # Net activity: mean over all units, allowing excitatory/inhibitory cancellation.
            # Near-zero for balanced EI, large for imbalanced activity.
            batch_activity = activation.mean(dim=spatial_dims).abs()
        else:  # "absolute"
            # Total absolute activity: L-p norm, normalized by unit count.
            batch_activity = torch.linalg.vector_norm(
                activation, ord=self.ord, dim=spatial_dims
            )
            if module_name not in self._norm_factors:
                n_units = activation.shape[1:].numel()
                self._norm_factors[module_name] = n_units ** (1 / self.ord)
            batch_activity = batch_activity / self._norm_factors[module_name]

        # Accumulate activity across timesteps
        if module_name not in self.batch_activity:
            self.batch_activity[module_name] = batch_activity
            self._hook_call_count[module_name] = 1
        else:
            existing_activity = self.batch_activity[module_name]
            if existing_activity.device != batch_activity.device:
                existing_activity = existing_activity.to(batch_activity.device)
            self.batch_activity[module_name] = existing_activity + batch_activity
            self._hook_call_count[module_name] += 1

        self._last_device = batch_activity.device
        self._last_dtype = batch_activity.dtype

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
        """Compute activity loss from accumulated statistics across timesteps.

        The accumulated (pre-normalized) activity is averaged over modules and
        timesteps, then returned per batch element for apply_reduction.
        """
        if not self.batch_activity:
            device = self._last_device if self._last_device is not None else None
            dtype = self._last_dtype if self._last_dtype is not None else torch.float32
            return torch.zeros(1, device=device, dtype=dtype)

        # Infer number of timesteps from hook call counts
        call_counts = list(self._hook_call_count.values())
        n_timesteps = max(call_counts) if call_counts else 1

        total_activity: Optional[torch.Tensor] = None
        module_count = 0

        for batch_activity in self.batch_activity.values():
            # Activity values are already normalized per-module in _accumulate_activity
            total_activity = (
                batch_activity
                if total_activity is None
                else total_activity + batch_activity
            )
            module_count += 1

        if module_count > 0 and total_activity is not None:
            loss = total_activity / (module_count * n_timesteps)
        else:
            device = self._last_device if self._last_device is not None else None
            dtype = self._last_dtype if self._last_dtype is not None else torch.float32
            loss = torch.zeros(1, device=device, dtype=dtype)

        # Clear accumulators for next batch
        self.batch_activity = {}
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


# Backward-compatible alias
ActivityLoss = ActivityLoss
