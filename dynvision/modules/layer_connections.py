"""
Modules to implement connections between non-successive layers in convolutional neural networks.
"""

from typing import Union, Callable, Optional
import torch
import torch.nn as nn
from fractions import Fraction
import torch.nn.init as init

__all__ = ["SkipConnection", "Feedback"]


class ConnectionBase(nn.Module):
    """
    The connection module adds a hidden state (h) of a source module to the input tensor (x). There are two ways to use this module:

    a) Initialize without parameters `conn = Connection()`, and manually pass hidden state tensor to the forward call `conn(x, h)`.
    b) Initialize with a source module and delay index `conn = Connection(source=source, delay_index=0)` to internally retrieve the relevant hidden state when calling `conn(x)`. Note: the source module must have a `get_hidden_state` function that takes the delay_index as an argument.

    In either case, when h doesn't have the same shape as x, a 1x1 convolution is applied to h to match the shapes. If the scale_factor = shape h / shape x is a positive integer a corresponding stride is applied in the convolution, if scale_factor is < 1 a corresponding upsampling is applied using a given upsampling mode (default=bilinear).
    To define the convolution, you have to pass the `in_channels` (n_channels of h), `out_channels` (n_channels of x), and stride (factor between spatial dimension of h and x). If you don't pass them, the module will try to infer them from the source module.
    If the shape mismatch is not previously known or the architecture should remain more flexible, you can instead set `auto_adapt=True`, which will automatically infer the `in_channels`, `out_channels`, and `stride` upon the first forward pass that combines x and h. Note that this may cause issues with loading state_dicts or checkpoints.
    """

    def __init__(
        self,
        source: Optional[nn.Module] = None,
        delay_index: int = 0,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        scale_factor: float = 1,
        bias: bool = True,
        parametrization: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        upsample_mode: str = "bilinear",
        auto_adapt: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.source = source
        self.delay_index = delay_index
        self.bias = bias
        self.scale_factor = scale_factor
        self.parametrization = parametrization
        self.setup_transform = auto_adapt

        if not auto_adapt:
            # infer in_channels, out_channels from source module
            if in_channels is None:
                in_channels = source.out_channels
            if out_channels is None:
                out_channels = in_channels

            if scale_factor is not None and scale_factor >= 1:
                if not isinstance(scale_factor, int) and not scale_factor.is_integer():
                    raise ValueError(
                        "scale_factor must be an integer when greater than 1."
                    )
                # downsample h to match x
                x_proxy = torch.randn(out_channels, 1, 1)
                h_proxy = torch.randn(in_channels, scale_factor, scale_factor)
            else:
                # upsample h to match x
                s = Fraction(scale_factor).limit_denominator()
                x_proxy = torch.randn(out_channels, s.denominator, s.denominator)
                h_proxy = torch.randn(in_channels, s.numerator, s.numerator)

            # setup conv and upsample upon initialization
            self.conv = self._setup_conv(x=x_proxy, h=h_proxy)
            self.upsample = self._setup_upsample(
                x=x_proxy, h=h_proxy, mode=upsample_mode
            )

    def _setup_conv(self, x: torch.Tensor, h: torch.Tensor) -> Union[nn.Module, bool]:
        if x is None or h is None:
            return False
        else:
            self.setup_transform = False

        in_channels = h.shape[-3]
        out_channels = x.shape[-3]
        stride = h.shape[-1] // x.shape[-1] or 1

        if in_channels == out_channels and stride == 1:
            # no channel mismatch
            return False
        else:
            transform_conv = self.parametrization(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=self.bias,
                    device=h.device,
                )
            )
            self._initialize_weights(transform_conv)
            transform_conv = transform_conv.to(dtype=h.dtype)
            return transform_conv

    def _setup_upsample(
        self, x: torch.Tensor, h: torch.Tensor, mode: str = "bilinear"
    ) -> nn.Module:
        if x is None or h is None:
            return False

        scale_factor = x.shape[-1] / h.shape[-1]

        if scale_factor <= 1:
            return False
        else:
            return nn.Upsample(scale_factor=scale_factor, mode=mode)

    def _initialize_weights(self, conv_layer: nn.Conv2d):
        init.kaiming_normal_(conv_layer.weight, mode="fan_in", nonlinearity="relu")
        if conv_layer.bias is not None:
            init.zeros_(conv_layer.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        if x is None:
            return None

        if h is None:  # passed hidden state takes precedence
            try:
                h = self.source.get_hidden_state(self.delay_index)
            except Exception as e:
                print(e)
                print(
                    "source must have get_hidden_state function, and must be callable with delay_index as argument."
                )

        if self.setup_transform:
            self.conv = self._setup_conv(x, h)
            self.upsample = self._setup_upsample(x, h)

        if self.conv and h is not None:
            h = self.conv(h)

        if self.upsample and h is not None:
            h = self.upsample(h)

        if h is None:
            return x
        else:
            return x + h


class SkipConnection(ConnectionBase):
    """
    The skip connection module adds the input tensor (x) of an earlier layer to the output tensor (h) of a deeper layer.
    """

    pass


class Feedback(ConnectionBase):
    """
    The feedback connection module adds a hidden state (h) of a deeper layer to the input tensor (x) of an earlier layer.
    """

    pass


if __name__ == "__main__":
    input_shape = (3, 32, 32)
    batch_size = 1
    print("Input shape:", (batch_size, *input_shape))

    for class_name in __all__:
        print(f"Testing {class_name}:")

        # Create an instance of the class
        module = globals()[class_name]

        # Create a dummy source module with a get_hidden_state method
        class DummySource(nn.Module):
            def __init__(self, out_channels, scale=64):
                super(DummySource, self).__init__()
                self.out_channels = out_channels
                self.scale = scale

            def get_hidden_state(self, delay_index):
                return torch.randn(
                    batch_size, self.out_channels, self.scale, self.scale
                )

        if class_name == "SkipConnection":
            source = DummySource(out_channels=10, scale=64)
        elif class_name == "Feedback":
            source = DummySource(out_channels=1, scale=24)
        else:
            source = None

        model = module(source=source, delay_index=0, auto_adapt=True)

        # Generate random input
        random_input = torch.randn(batch_size, *input_shape)

        # Perform forward pass
        output = model(random_input)

        # Print number of parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {n_params}")

        # Define an arbitrary loss function
        loss_fn = nn.MSELoss()

        # Generate random target
        target = torch.randn_like(output)

        # Compute loss
        loss = loss_fn(output, target)

        # Perform backward pass
        loss.backward()

        # Print results
        print(f"Model Output: {output.shape if output is not None else 'None'}\n")

    print("All tests passed!")
