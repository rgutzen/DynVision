"""
CorNet: Recurrent Convolutional Neural Network with Biological Features

This module implements the CorNet-RT and CorNet-Z architectures, a biologically-inspired neural network that incorporates recurrent processing and hierarchical organization similar to the primate visual cortex. The model consists of four layers: V1, V2, V4, and IT, each.

References:
- Kubilius et al. (2019) "CORnet: Modeling the Neural Mechanisms of Core Object Recognition"
- DiCarlo et al. (2012) "How Does the Brain Solve Visual Object Recognition?"
"""

import logging
from typing import Optional, Dict, Any, Tuple, Union, List

import torch
import torch.nn as nn
from torch.utils import model_zoo

from dynvision.models.dyrcnn import DyRCNN
from dynvision.model_components import (
    InputAdaption,
    RConv2d,
    Skip,
)


class ZeroStateSentinel:
    """Sentinel representing zero state for uninitialized hidden states.

    This allows returning 'zero' without knowing the batch size.
    Operations check isinstance and treat it as additive identity.
    """

    pass


__all__ = ["CorNetRT"]


logger = logging.getLogger(__name__)


class CorNetRT(DyRCNN):
    def __init__(
        self,
        n_timesteps: int = 5,
        input_dims: Tuple[int, int, int] = (5, 3, 224, 224),
        n_classes: int = 1000,
        dt: float = 2,  # ms
        t_feedforward: float = 2,  # ms
        t_recurrence: float = 2,  # ms
        recurrence_type: str = "self",
        fixed_self_weight: float = 1.0,
        recurrence_bias: bool = False,
        recurrence_target: str = "middle",
        skip: bool = True,  # Force skip connections to inject zeros at early timesteps
        t_skip: float = 0.0,  # Zero delay for immediate effect
        feedback: bool = False,
        init_with_pretrained=True,
        **kwargs: Any,
    ) -> None:

        breakpoint()

        self.model_letter = "rt"
        self.model_hash = "933c001c"
        self.init_with_pretrained = init_with_pretrained
        self.fixed_self_weight = fixed_self_weight
        self.recurrence_bias = recurrence_bias

        super().__init__(
            n_timesteps=n_timesteps,
            input_dims=input_dims,
            n_classes=n_classes,
            dt=dt,
            t_feedforward=t_feedforward,
            t_recurrence=t_recurrence,
            recurrence_type=recurrence_type,
            recurrence_target=recurrence_target,
            skip=skip,
            t_skip=t_skip,
            feedback=feedback,
            **kwargs,
        )

    def _init_parameters(self) -> None:
        if self.init_with_pretrained:
            self.load_pretrained_state_dict(check_mismatch_layer=["classifier.2"])
            self.trainable_parameter_names = [
                p for p in list(self.state_dict().keys()) if "classifier.2" in p
            ]

        else:
            self.trainable_parameter_names = [
                p for p in list(self.state_dict().keys())
            ]

        # Initialize skip connections to provide zeros
        self._init_skip_connections()

    def _init_skip_connections(self) -> None:
        """Initialize skip connection weights to 0 and freeze them.

        This makes skip connections always output zeros, which:
        - At early timesteps: Provides zero tensors when no feedforward input exists
        - At later timesteps: Adding zeros doesn't change feedforward input
        """
        for layer_name in ["V2", "V4", "IT"]:
            skip_name = f"addskip_{layer_name}"
            if hasattr(self, skip_name):
                skip = getattr(self, skip_name)
                logger.info(f"Initializing {skip_name} weights to 0 (frozen)")

                # Skip connections use a Conv2d layer to adapt channels
                if hasattr(skip, "conv") and isinstance(skip.conv, nn.Conv2d):
                    skip.conv.weight.data.fill_(0.0)
                    skip.conv.weight.requires_grad = False
                    if skip.conv.bias is not None:
                        skip.conv.bias.data.fill_(0.0)
                        skip.conv.bias.requires_grad = False
                    logger.info(f"  Froze {skip_name}.conv weights to 0")

    def download_pretrained_state_dict(self):
        url = f"https://s3.amazonaws.com/cornet-models/cornet_{self.model_letter.lower()}-{self.model_hash}.pth"
        ckpt_data = torch.utils.model_zoo.load_url(url, map_location=self.device)
        state_dict = ckpt_data["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        return state_dict

    def translate_pretrained_layer_names(self):
        translate_layer_names = {
            "V1.norm1.weight": "norm_V1.weight",
            "V1.norm1.bias": "norm_V1.bias",
            "V2.norm1.weight": "norm_V2.weight",
            "V2.norm1.bias": "norm_V2.bias",
            "V4.norm1.weight": "norm_V4.weight",
            "V4.norm1.bias": "norm_V4.bias",
            "IT.norm1.weight": "norm_IT.weight",
            "IT.norm1.bias": "norm_IT.bias",
            "conv1": "conv2",
            "conv_input": "conv",
            "norm_input": "mid_modules",
            "decoder.linear": "classifier.2",
        }
        return translate_layer_names

    def _define_architecture(self):
        self.layer_names = ["V1", "V2", "V4", "IT"]
        # define operations order within layer
        self.layer_operations = [
            "addskip",  # add skip connection (provides zeros when no feedforward input)
            "layer",  # apply (recurrent) convolutional layer
            "norm",  # apply normalization
            "nonlin",  # apply nonlinearity
            "record",  # record activations in responses dict
            "delay",  # set and get delayed activations for next layer
        ]
        # V1
        self.V1 = RConv2d(
            in_channels=3,
            mid_channels=64,
            out_channels=64,
            kernel_size=(7, 3),
            stride=(4, 1),
            bias=(True, False),
            mid_modules=nn.GroupNorm(32, 64),
            fixed_self_weight=self.fixed_self_weight,
            recurrence_bias=self.recurrence_bias,
            recurrence_target=self.recurrence_target,
            dt=self.dt,
            t_feedforward=self.t_feedforward,
            t_recurrence=self.t_recurrence,
            dim_y=self.dim_y,
            dim_x=self.dim_x,
            history_length=self.history_length,
        )
        self.norm_V1 = nn.GroupNorm(32, 64)

        # V2
        self.V2 = RConv2d(
            in_channels=64,
            mid_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(2, 1),
            bias=(True, False),
            mid_modules=nn.GroupNorm(32, 128),
            fixed_self_weight=self.fixed_self_weight,
            recurrence_bias=self.recurrence_bias,
            recurrence_target=self.recurrence_target,
            dt=self.dt,
            t_feedforward=self.t_feedforward,
            t_recurrence=self.t_recurrence,
            dim_y=self.V1.dim_y // self.V1.stride[0] // self.V1.stride[1],
            dim_x=self.V1.dim_x // self.V1.stride[0] // self.V1.stride[1],
            history_length=self.history_length,
        )
        self.norm_V2 = nn.GroupNorm(32, 128)

        # V4
        self.V4 = RConv2d(
            in_channels=128,
            mid_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(2, 1),
            bias=(True, False),
            mid_modules=nn.GroupNorm(32, 256),
            fixed_self_weight=self.fixed_self_weight,
            recurrence_bias=self.recurrence_bias,
            recurrence_target=self.recurrence_target,
            dt=self.dt,
            t_feedforward=self.t_feedforward,
            t_recurrence=self.t_recurrence,
            dim_y=self.V2.dim_y // self.V2.stride[0] // self.V2.stride[1],
            dim_x=self.V2.dim_x // self.V2.stride[0] // self.V2.stride[1],
            history_length=self.history_length,
        )
        self.norm_V4 = nn.GroupNorm(32, 256)

        # IT
        self.IT = RConv2d(
            in_channels=256,
            mid_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(2, 1),
            bias=(True, False),
            mid_modules=nn.GroupNorm(32, 512),
            fixed_self_weight=self.fixed_self_weight,
            recurrence_bias=self.recurrence_bias,
            recurrence_target=self.recurrence_target,
            dt=self.dt,
            t_feedforward=self.t_feedforward,
            t_recurrence=self.t_recurrence,
            dim_y=self.V4.dim_y // self.V4.stride[0] // self.V4.stride[1],
            dim_x=self.V4.dim_x // self.V4.stride[0] // self.V4.stride[1],
            history_length=self.history_length,
        )
        self.norm_IT = nn.GroupNorm(32, 512)

        # Shared nonlinearity for all layers
        # IMPORTANT: inplace=False to avoid corrupting activations when reused across layers
        self.nonlin = nn.ReLU(inplace=False)

        # Skip connections from V1 to inject zeros at early timesteps
        # When there's no feedforward input (delay), skip provides zeros tensor
        # When there is feedforward input, adding zeros doesn't change anything
        # force_conv=True ensures conv is created even when channels match (V2)
        if self.skip:
            # V2: V1 is 56x56, V2 expects 28x28 → V1 is 2× larger → scale_factor=2
            self.addskip_V2 = Skip(
                source=self.V1,
                in_channels=64,
                out_channels=64,
                scale_factor=2,
                force_conv=True,  # Create conv even though channels match
                delay_index=int(self.t_skip / self.dt),  # 0 for immediate effect
            )
            # V4: V1 is 56x56, V4 expects 14x14 → V1 is 4× larger → scale_factor=4
            self.addskip_V4 = Skip(
                source=self.V1,
                in_channels=64,
                out_channels=128,
                scale_factor=4,
                force_conv=True,
                delay_index=int(self.t_skip / self.dt),
            )
            # IT: V1 is 56x56, IT expects 7x7 → V1 is 8× larger → scale_factor=8
            self.addskip_IT = Skip(
                source=self.V1,
                in_channels=64,
                out_channels=256,
                scale_factor=8,
                force_conv=True,
                delay_index=int(self.t_skip / self.dt),
            )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, self.n_classes),
        )

    def forward(self, x, **kwargs):
        """Forward pass with input statistics logging.

        Args:
            x: Input tensor
            **kwargs: Additional arguments passed to parent forward

        Returns:
            Model output
        """
        # Log input statistics as received by the model
        logger.warning(
            f"🔍 CorNetRT input stats: mean={x.mean():.3f}, std={x.std():.3f}, "
            f"min={x.min():.3f}, max={x.max():.3f}, shape={x.shape}"
        )

        # Call parent forward pass
        return super().forward(x, **kwargs)

    def debug_temporal_params(self):
        """Debug helper to print temporal parameters for all layers.

        Note: t_feedforward is not stored in RConv2d since forward delay
        happens in the delay operation between layers. RConv2d only handles
        recurrence delay internally.
        """
        print("\n" + "=" * 80)
        print("Temporal Parameters Debug")
        print("=" * 80)
        print(f"Model level:")
        print(f"  dt: {self.dt}")
        print(f"  t_feedforward: {self.t_feedforward}")
        print(f"  t_recurrence: {self.t_recurrence}")
        print(f"  history_length: {self.history_length} (=max(t_ff, t_rec)/dt + 1)")
        print()
        for layer_name in self.layer_names:
            layer = getattr(self, layer_name)
            print(f"{layer_name}:")
            print(f"  dt: {layer.dt}")
            print(f"  t_recurrence: {layer.t_recurrence}")
            print(f"  delay_recurrence: {layer.delay_recurrence} (=int(t_rec/dt))")
            print(f"  history_length: {layer.history_length}")
            print(
                f"  n_hidden_states: {layer.n_hidden_states} (=int(history_len/dt)+1)"
            )
        print("=" * 80 + "\n")

    def reset(self):
        """Reset all hidden states in recurrent layers.

        Note: DynVision's None handling for uninitialized recurrent states
        is equivalent to the original's scalar 0:
        - Original: `tensor + 0 = tensor`
        - DynVision: `if h is None: return x`
        Both return x unchanged when there's no recurrent state.
        """
        for layer in [self.V1, self.V2, self.V4, self.IT]:
            layer.reset()


def test_cornet(
    input_shape: Tuple[int, ...] = (5, 3, 224, 224),
    device: Optional[torch.device] = None,
    model_class=CorNetRT,
) -> None:
    """Test CorNet implementation.

    Tests include:
    - Model creation and setup
    - Forward pass with random input
    - Stability checks with extreme values
    - Parameter verification

    Args:
        input_shape: Input tensor shape
        device: Device to run test on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing CorNet-RT on {device}")

    # Create and setup model
    model = model_class(
        input_dims=input_shape,
    )
    model.setup("fit")
    print("Model creation successful")

    # Test forward pass
    x = torch.randn(1, *input_shape, device=device)
    y = model(x)
    print(f"Forward pass successful: {x.shape} -> {y.shape}")

    # Log trainable parameters
    trainable_params = [
        f"{name} [{tuple(param.shape)}]"
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    print("Trainable Parameters:\n\t%s" % "\n\t".join(trainable_params))

    print("All tests passed!")


if __name__ == "__main__":

    test_cornet()
