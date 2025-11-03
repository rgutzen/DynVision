"""
CordsNet: Contextual Recurrent Deep Structured Network

This module implements CordsNet, a hierarchical recurrent neural network
with complex skip connections and spatial biases.

Reference:
- Wang et al. (2020) "CordsNet: Contextual Recurrent Deep Structured Network"
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

from dynvision.base import BaseModel
from dynvision.model_components import (
    RecurrentConnectedConv2d as RConv2d,
    SpatialBias,
    Skip,
    EulerStep,
)
from dynvision.project_paths import project_paths
from dynvision.utils import alias_kwargs

__all__ = ["CordsNet"]

logger = logging.getLogger(__name__)


class CordsNet(BaseModel):
    """
    CordsNet: Contextual Recurrent Deep Structured Network

    Reimplementation of CordsNet using DynVision's layer_operations pattern.
    Uses standard TemporalBase._forward() with proper delay coordination.

    Architecture:
        - 8 recurrent layers with complex skip connections
        - Channels: [64, 64, 64, 128, 128, 256, 256, 512, 512]
        - Strides: [1, 1, 2, 1, 2, 1, 2, 1]
        - Skip pattern: Each layer receives from previous + 2-layers-back
        - Spatial bias per layer (per-unit)

    Dynamics:
        Original update rule:
            r_new = (1 - alpha) * r_old + alpha * relu(recurrent(r) + feedforward(input) + bias)

        Equivalent to Euler integration with dt/tau = alpha:
            alpha = dt / tau

        Default: dt=1, tau=10 → alpha=0.1

    Args:
        n_timesteps: Number of timesteps to process (default: 100)
        input_dims: Input dimensions (t, c, y, x)
        n_classes: Number of output classes
        dt: Integration time step (controls temporal resolution)
        tau: Time constant (controls integration rate, alpha = dt/tau)
        t_feedforward: Delay for feedforward connections between layers
        t_recurrence: Delay for recurrent connections within each layer
        t_skip: Delay for skip connections
        idle_timesteps: Timesteps of spontaneous activity before input (default: 100)
        recurrence_type: Type of recurrent connection (default: "full" for 3x3 conv)
    """

    @alias_kwargs(
        tff="t_feedforward",
        trc="t_recurrence",
        tsk="t_skip",
        rctype="recurrence_type",
    )
    def __init__(
        self,
        n_timesteps: int = 100,
        input_dims: Tuple[int, int, int, int] = (100, 3, 224, 224),
        n_classes: int = 1000,
        dt: float = 1.0,
        tau: float = 10.0,  # alpha = dt/tau = 0.1 by default
        t_feedforward: float = 1.0,
        t_recurrence: float = 1.0,
        t_feedback=0.0,  # Not used in CordsNet
        t_skip: float = 1.0,
        recurrence_type: str = "full",  # 3x3 conv for recurrence
        idle_timesteps: int = 100,
        init_with_pretrained: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize CordsNet model.

        Note: The original CordsNet uses alpha=0.1, which corresponds to dt=1, tau=10.
        """
        self.init_with_pretrained = init_with_pretrained

        super().__init__(
            n_timesteps=n_timesteps,
            input_dims=input_dims,
            n_classes=n_classes,
            dt=float(dt),
            tau=float(tau),
            t_feedforward=float(t_feedforward),
            t_recurrence=float(t_recurrence),
            t_feedback=float(t_feedback),
            t_skip=float(t_skip),
            recurrence_type=recurrence_type,
            idle_timesteps=int(idle_timesteps),
            **kwargs,
        )

    def _init_parameters(self) -> None:
        """Initialize model parameters, optionally loading pretrained weights."""
        if self.init_with_pretrained:
            self.load_pretrained_state_dict(check_mismatch_layer=["out_fc"])
            # Make only the classifier trainable
            self.trainable_parameter_names = [
                p for p in list(self.state_dict().keys()) if "out_fc" in p
            ]
        else:
            self.trainable_parameter_names = list(self.state_dict().keys())

    def download_pretrained_state_dict(self) -> dict:
        """
        Download pretrained CordsNet weights.

        Note: GitHub blob URLs don't work for direct download. Need to use raw.githubusercontent.com
        or use the actual release/download link.
        """
        # TODO: Fix URL to actual pretrained weights location
        url = "https://raw.githubusercontent.com/wmws2/cordsnet/main/cordsnetr8.pth"
        save_path = project_paths.models / "CordsNet" / "cordsnet_pretrained.pth"

        if not save_path.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading pretrained CordsNet from {url}")

            import requests

            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Downloaded pretrained model to {save_path}")
            else:
                raise RuntimeError(
                    f"Failed to download pretrained weights from {url} "
                    f"(status code: {response.status_code})"
                )
        else:
            logger.info(f"Using cached pretrained model from {save_path}")

        state_dict = torch.load(save_path, map_location=self.device)
        return state_dict

    def translate_pretrained_layer_names(self) -> dict:
        """
        Map original CordsNet pretrained weight names to DynVision structure.

        Original structure:
        - inp_conv: initial 7x7 convolution on input
        - inp_avgpool: average pooling after initial conv
        - inp_skip: learned skip connection from input to layer2
        - area_conv[i]: recurrent connection for layer i (3x3 conv)
        - area_area[i]: feedforward connection for layer i (3x3 conv with stride)
        - conv_bias[i]: spatial bias for layer i [channels, height, width]
        - skip_area[0,1,2]: learned skip connections (1x1 conv with stride=2)
        - out_fc: final linear classifier

        DynVision structure:
        - layer0: RConv2d for input processing
        - layer{1-8}: RConv2d modules
          - layer.conv: feedforward (area_area)
          - layer.recurrence.conv: recurrent (area_conv)
        - addbias_layer{1-8}.bias: spatial biases (conv_bias)
        - addskip_layer{N}: Skip modules with learned convs
        - classifier: final classifier
        """
        translate_layer_names = {
            "inp_conv": "layer0.conv",
            "inp_avgpool": "pool_layer0",
            "out_fc": "classifier.2",
        }

        # Map layers 1-8
        for i in range(8):
            layer_name = f"layer{i+1}"
            # area_conv → recurrence conv
            translate_layer_names[f"area_conv.{i}"] = f"{layer_name}.recurrence.conv"
            # area_area → feedforward conv
            translate_layer_names[f"area_area.{i}"] = f"{layer_name}.conv"
            # conv_bias → spatial bias
            translate_layer_names[f"conv_bias.{i}"] = f"addbias_{layer_name}.bias"

        # Map skip connections
        translate_layer_names["inp_skip"] = "addskip_layer2.conv"
        translate_layer_names["skip_area.0"] = "addskip_layer4.conv"
        translate_layer_names["skip_area.1"] = "addskip_layer6.conv"
        translate_layer_names["skip_area.2"] = "addskip_layer8.conv"

        return translate_layer_names

    def reset(self, input_shape: Optional[Tuple[int, ...]] = None) -> None:
        """Reset all layer hidden states and dynamics solvers."""
        for layer_name in self.layer_names:
            # Reset layer hidden states
            layer = getattr(self, layer_name)
            if hasattr(layer, "reset"):
                layer.reset()

            # Reset dynamics solvers (layers 1-8 only)
            solver_name = f"tstep_{layer_name}"
            if hasattr(self, solver_name):
                solver = getattr(self, solver_name)
                solver.reset()

    def _define_architecture(self) -> None:
        """
        Define CordsNet architecture using DynVision's layer_operations pattern.

        Architecture:
        - layer0: Input processing (7x7 conv + pool, no recurrence)
        - layer1-8: Recurrent layers with 3x3 feedforward + 3x3 recurrence
        - Skip connections between non-consecutive layers
        - Spatial biases per layer

        Layer structure:
        - layer0: 3→64 (conv stride 2, pool stride 2 → 56x56)
        - layer1: 64→64 (stride 1, size 56x56)
        - layer2: 64→64 (stride 1, size 56x56) + skip from layer0
        - layer3: 64→128 (stride 2, size 28x28) + skip from layer1
        - layer4: 128→128 (stride 1, size 28x28) + skip from layer2
        - layer5: 128→256 (stride 2, size 14x14) + skip from layer3
        - layer6: 256→256 (stride 1, size 14x14) + skip from layer4
        - layer7: 256→512 (stride 2, size 7x7) + skip from layer5
        - layer8: 512→512 (stride 1, size 7x7) + skip from layer6
        """

        # Include input processing as layer0
        self.layer_names = [
            "layer0",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "layer5",
            "layer6",
            "layer7",
            "layer8",
        ]

        # Define operation sequence (matches DyRCNN pattern)
        self.layer_operations = [
            "layer",  # RConv2d (feedforward + recurrence)
            "pool",  # Pooling (layer0 only)
            "addskip",  # Skip connection (if defined)
            "addbias",  # Spatial bias (layers 1-8 only)
            "nonlin",  # Shared ReLU
            "tstep",  # Euler integration (continuous-time dynamics)
            "delay",  # Store and retrieve delayed activation
            "record",  # Record response
        ]

        # Shared nonlinearity
        self.nonlin = nn.ReLU(inplace=False)

        # Configuration arrays
        channels = [64, 64, 64, 128, 128, 256, 256, 512, 512]
        sizes = [56, 56, 28, 28, 14, 14, 7, 7]
        strides = [1, 1, 2, 1, 2, 1, 2, 1]

        # Layer 0: Input processing (no recurrence, no bias)
        self.layer0 = RConv2d(
            in_channels=self.n_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            recurrence_type="none",  # No recurrence for input layer
            history_length=max(self.t_feedforward, self.t_skip),
            parametrization=nn.utils.parametrizations.weight_norm,
        )
        self.pool_layer0 = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False
        )

        # Common parameters for recurrent layers
        common_params = {
            "bias": False,  # Using spatial bias modules instead
            "recurrence_type": self.recurrence_type,  # "full" = 3x3 conv
            "recurrence_kernel_size": 3,  # 3x3 recurrent convolution
            "recurrence_target": "output",  # Recurrence on layer output
            "recurrence_bias": False,  # No bias in recurrent conv
            "t_recurrence": self.t_recurrence,
            "history_length": max(self.t_feedforward, self.t_skip, self.t_recurrence),
            "parametrization": nn.utils.parametrizations.weight_norm,
        }

        # Layers 1-8: Recurrent layers with spatial biases
        for i in range(8):
            layer_name = f"layer{i+1}"

            # RConv2d layer: feedforward (area_area) + recurrence (area_conv)
            layer = RConv2d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=3,  # Feedforward kernel
                stride=strides[i],  # Feedforward stride
                padding=1,
                **common_params,
            )
            setattr(self, layer_name, layer)

            # Spatial bias module (per-unit bias)
            bias = SpatialBias(
                channels=channels[i + 1],
                height=sizes[i],
                width=sizes[i],
                init_value=0.0,
            )
            setattr(self, f"addbias_{layer_name}", bias)

            # Dynamics solver (Euler integration)
            # Original: r = (1-alpha)*r + alpha*relu(...)  where alpha = dt/tau
            tau = torch.nn.Parameter(
                torch.tensor(self.tau, dtype=torch.float32),
                requires_grad=False,
            )
            setattr(self, f"tau_{layer_name}", tau)

            solver = EulerStep(dt=self.dt, tau=tau)
            setattr(self, f"tstep_{layer_name}", solver)

        # Skip connections (matching original pattern)
        self._define_skip_connections()

        # Output classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.parametrizations.weight_norm(
                nn.Linear(512, self.n_classes, bias=True)
            ),
        )

    def _define_skip_connections(self) -> None:
        """
        Define skip connections matching original CordsNet pattern.

        Skip pattern:
        - layer2: from layer0 with learned 3x3 conv
        - layer3: from layer1 (direct, same channels)
        - layer4: from layer2 with learned 1x1 conv + stride
        - layer5: from layer3 (direct, different channels but auto-adapt)
        - layer6: from layer4 with learned 1x1 conv + stride
        - layer7: from layer5 (direct, different channels but auto-adapt)
        - layer8: from layer6 with learned 1x1 conv + stride
        """
        delay_skip = int(self.t_skip / self.dt)

        # layer2: learned skip from layer0 (3x3 conv, same channels)
        self.addskip_layer2 = Skip(
            source=self.layer0,
            in_channels=64,
            out_channels=64,
            delay_index=delay_skip,
            bias=False,
            parametrization=nn.utils.parametrizations.weight_norm,
            auto_adapt=False,
        )
        # Override with 3x3 conv instead of default 1x1
        self.addskip_layer2.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        )

        # layer3: direct skip from layer1 (same channels)
        self.addskip_layer3 = Skip(
            source=self.layer1,
            delay_index=delay_skip,
            auto_adapt=True,
        )

        # layer4: learned skip from layer2 (1x1 conv with stride=2)
        self.addskip_layer4 = Skip(
            source=self.layer2,
            in_channels=64,
            out_channels=128,
            scale_factor=2,  # Downsample by 2
            delay_index=delay_skip,
            bias=False,
            parametrization=nn.utils.parametrizations.weight_norm,
            auto_adapt=False,
        )

        # layer5: direct skip from layer3 (auto-adapt for channel mismatch)
        self.addskip_layer5 = Skip(
            source=self.layer3,
            delay_index=delay_skip,
            auto_adapt=True,
        )

        # layer6: learned skip from layer4 (1x1 conv with stride=2)
        self.addskip_layer6 = Skip(
            source=self.layer4,
            in_channels=128,
            out_channels=256,
            scale_factor=2,
            delay_index=delay_skip,
            bias=False,
            parametrization=nn.utils.parametrizations.weight_norm,
            auto_adapt=False,
        )

        # layer7: direct skip from layer5 (auto-adapt for channel mismatch)
        self.addskip_layer7 = Skip(
            source=self.layer5,
            delay_index=delay_skip,
            auto_adapt=True,
        )

        # layer8: learned skip from layer6 (1x1 conv with stride=2)
        self.addskip_layer8 = Skip(
            source=self.layer6,
            in_channels=256,
            out_channels=512,
            scale_factor=2,
            delay_index=delay_skip,
            bias=False,
            parametrization=nn.utils.parametrizations.weight_norm,
            auto_adapt=False,
        )


if __name__ == "__main__":
    # Test model creation
    input_shape = (100, 3, 224, 224)

    model = CordsNet(
        input_dims=input_shape,
        n_classes=1000,
        n_timesteps=100,
        dt=1,
        tau=10,
        t_feedforward=1,
        t_recurrence=1,
        t_skip=1,
        recurrence_type="full",
        idle_timesteps=100,
    )
    model.setup("fit")

    print(f"Model created successfully!")
    print(f"Layer names: {model.layer_names}")
    print(f"Layer operations: {model.layer_operations}")

    # Test forward pass
    x = torch.randn(1, *input_shape)
    y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Residual timesteps: {model.n_residual_timesteps}")
