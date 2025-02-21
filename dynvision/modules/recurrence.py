import sys
from collections import deque
from typing import Callable, Optional, Union, Deque

import torch
import torch.nn as nn
from torchsummary import summary

from dynvision.modules.topographic_recurrence import (
    LocalLateralConnection,
    LocalSeparableConnection,
)
from dynvision.utils.utils import str_to_bool

__all__ = [
    "DepthwiseSeparableConnection",
    "DepthPointwiseConnection",
    "PointDepthwiseConnection",
    "FullConnection",
    "SelfConnection",
    "InputAdaption",
    "RecurrentConnectedConv2d",
]


class TemplateConnection(nn.Module):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super(TemplateConnection, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _init_parameters(self) -> None:
        raise NotImplementedError


class DepthwiseSeparableConnection(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        bias: bool = False,
        max_weight_init: float = 0.05,
        parametrization: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        **kwargs,
    ) -> None:
        super(DepthwiseSeparableConnection, self).__init__()
        self.max_weight_init = max_weight_init

        self.depthwise_conv = parametrization(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=bias,
            )
        )
        self.pointwise_conv = parametrization(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.depthwise_conv(x)
        h = self.pointwise_conv(h)
        return h

    def _init_parameters(self) -> None:
        max_w_init = self.max_weight_init
        for module in [self.depthwise_conv, self.pointwise_conv]:
            if hasattr(module, "weight.original0"):
                module.weight.original0.data.fill_(1.0)
            if hasattr(module, "weight.original1"):
                nn.init.uniform_(
                    module.weight.original1.weight, a=-max_w_init, b=max_w_init
                )
            elif hasattr(module, "weight"):
                nn.init.uniform_(module.weight, a=-max_w_init, b=max_w_init)
            if hasattr(module, "bias") and module.bias:
                nn.init.constant_(module.bias, 0)


class DepthPointwiseConnection(DepthwiseSeparableConnection):
    pass


class PointDepthwiseConnection(DepthwiseSeparableConnection):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pointwise_conv(x)
        h = self.depthwise_conv(h)
        return h


class FullConnection(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        bias: bool = False,
        max_weight_init: float = 0.05,
        parametrization: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        **kwargs,
    ) -> None:
        super(FullConnection, self).__init__()

        self.conv = parametrization(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=bias,
            )
        )

        self.max_weight_init = max_weight_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return h

    def _init_parameters(self) -> None:
        max_w_init = self.max_weight_init
        if hasattr(self.conv, "weight.original0"):
            self.conv.weight.original0.data.fill_(1.0)
        if hasattr(self.conv, "weight.original1"):
            nn.init.uniform_(
                self.conv.weight.original1.weight, a=-max_w_init, b=max_w_init
            )
        elif hasattr(self.conv, "weight"):
            nn.init.uniform_(self.conv.weight, a=-max_w_init, b=max_w_init)
        if hasattr(self.conv, "bias") and self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)


class SelfConnection(nn.Module):
    def __init__(
        self,
        fixed_weight: Optional[float] = None,
        max_weight_init: float = 0.2,
        bias: bool = True,
        **kwargs,
    ) -> None:
        super(SelfConnection, self).__init__()

        self.requires_grad = fixed_weight is None
        self.max_weight_init = max_weight_init
        self.fixed_weight = fixed_weight

        self.weight = nn.Parameter(torch.Tensor([1]), requires_grad=self.requires_grad)

        if bias:
            self.bias = nn.Parameter(torch.Tensor([0]), requires_grad=True)

        self._init_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "bias"):
            return x * self.weight + self.bias
        else:
            return x * self.weight

    def _init_parameters(self) -> None:
        if self.requires_grad:
            nn.init.uniform_(self.weight, a=-self.max_weight_init, b=0)
        else:
            self.weight.data.fill_(self.fixed_weight)
        if hasattr(self, "bias"):
            nn.init.constant_(self.bias, 0)


class InputAdaption(nn.Module):
    def __init__(
        self,
        fixed_weight: Optional[float] = None,
        max_weight_init: float = 0.2,
        **kwargs,
    ) -> None:
        super(InputAdaption, self).__init__()

        self.recurrence = SelfConnection(
            fixed_weight=fixed_weight,
            max_weight_init=max_weight_init,
            bias=False,
            **kwargs,
        )
        self.reset()

    def reset(self) -> None:
        self.hidden_state = None

    def forward(self, x: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if self.hidden_state is None:
            self.hidden_state = x
            return x

        else:
            h = self.recurrence(self.hidden_state)
            self.hidden_state = h
            return h


class RecurrentConnectedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = True,
        dim_y: Optional[int] = None,
        dim_x: Optional[int] = None,
        dt: float = 1,  # ms
        tau: float = 1,  # ms
        recurrence_type: str = "self",
        recurrence_delay: float = 0,  # ms
        recurrence_influence: Union[Callable, str] = "additive",
        history_length: Optional[float] = None,  # ms
        fixed_self_weight: Optional[float] = None,
        max_weight_init: float = 0.001,
        parametrization: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        device: Optional[Union[str, torch.device]] = None,
        feedforward_only: bool = False,
        **kwargs,
    ) -> None:
        super(RecurrentConnectedConv2d, self).__init__()

        model_args = {
            k: v for k, v in locals().items() if k not in ["self", "kwargs"]
        } | kwargs
        for key, value in model_args.items():
            setattr(self, key, value)
            
        self.feedforward_only = str_to_bool(feedforward_only)

        self.history_length = (
            recurrence_delay if history_length is None else history_length
        )
        self.n_deque = int(self.history_length / self.dt) + 1
        self.recurrence_delay_i = int(self.recurrence_delay / self.dt)

        if isinstance(self.recurrence_influence, str):
            if self.recurrence_influence == "additive":
                self.recurrence_influence = lambda x, h: x + h
            elif self.recurrence_influence == "multiplicative":
                self.recurrence_influence = lambda x, h: x * (1 + torch.tanh(h))
            else:
                raise ValueError(
                    f"Invalid recurrence influence keyword: {self.recurrence_influence}! Use a callable function instead."
                )

        self._define_architecture()
        self._init_parameters()
        self.reset()

    def _define_architecture(self) -> None:
        if self.feedforward_only:
            self.recurrence_type = None
            
        padding = self.kernel_size // 2 if self.padding is None else self.padding

        self.conv = self.parametrization(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=padding,
                bias=self.bias,
                device=self.device,
            )
        )

        # select recurrence
        recurrence_params = dict(
            kernel_size=self.kernel_size,
            bias=False,
            parametrization=self.parametrization,
            max_weight_init=self.max_weight_init,
            device=self.device,
        )

        if self.recurrence_type == "self":
            self.recurrence = SelfConnection(fixed_weight=self.fixed_self_weight)

        elif self.recurrence_type == "full":
            self.recurrence = FullConnection(
                in_channels=self.out_channels,
                **recurrence_params,
            )

        elif self.recurrence_type == "depthpointwise":
            self.recurrence = DepthPointwiseConnection(
                in_channels=self.out_channels,
                **recurrence_params,
            )

        elif self.recurrence_type == "pointdepthwise":
            self.recurrence = PointDepthwiseConnection(
                in_channels=self.out_channels,
                **recurrence_params,
            )

        elif self.recurrence_type == "local":
            self.recurrence = LocalLateralConnection(
                in_channels=self.out_channels,
                dim_y=self.dim_y // self.stride,
                dim_x=self.dim_x // self.stride,
                **recurrence_params,
            )

        elif self.recurrence_type == "localdepthwise":
            self.recurrence = LocalSeparableConnection(
                in_channels=self.out_channels,
                dim_y=self.dim_y // self.stride,
                dim_x=self.dim_x // self.stride,
                **recurrence_params,
            )

        elif self.recurrence_type is None or self.recurrence_type.lower() == "none":
            self.recurrence = None

        else:
            raise ValueError(f"Invalid recurrence type: {self.recurrence_type}")

        self.relu = nn.ReLU(inplace=True)

    def _init_parameters(self) -> None:
        max_w_init = self.max_weight_init
        if hasattr(self.conv, "weight.original0"):
            self.conv.weight.original0.data.fill_(1.0)
        if hasattr(self.conv, "weight.original1"):
            nn.init.kaiming_normal_(self.conv.weight.original1, nonlinearity="relu")
        elif hasattr(self.conv, "weight"):
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if hasattr(self.conv, "bias") and self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        if hasattr(self.recurrence, "_init_parameters"):
            self.recurrence._init_parameters()

    def reset(self) -> None:
        self.hidden_states: Deque[torch.Tensor] = deque(maxlen=self.n_deque)

    def get_hidden_state(self, i: Optional[int] = None) -> Optional[torch.Tensor]:
        if i is None:
            return self.hidden_states
        elif i >= 0:
            i -= self.n_deque

        if abs(i) > len(self.hidden_states):
            return None
        else:
            return self.hidden_states[i]

    def set_hidden_state(self, h: torch.Tensor, i: Optional[int] = None) -> None:
        if i is None:
            self.hidden_states.append(h)
        else:
            self.hidden_states[i] = h
        return None

    def forward(
        self, x: Optional[torch.Tensor] = None, h: Optional[torch.Tensor] = None, **kwargs
    ) -> Optional[torch.Tensor]:
        # Get previous activation for recurrent input
        if h is None:  # passed hidden state takes precedence
            h = self.get_hidden_state(-1 * self.recurrence_delay_i)

        # Response to recurrent input
        if h is None or self.recurrence is None:
            h = None
        else:
            h = self.recurrence(h)

        # Response to feedforward input
        if x is not None:
            x = self.conv(x)

        # Combine responses
        if x is None and h is None:
            out = None
            return out

        elif x is None:
            out = h
        elif h is None:
            out = x
        else:
            out = self.recurrence_influence(x, h)

        return out


if __name__ == "__main__":
    input_shape = (3, 32, 32)
    batch_size = 1
    print("Input shape:", (batch_size, *input_shape))

    for class_name in __all__:
        print(f"Testing {class_name}:")

        # Create an instance of the class
        module = globals()[class_name]
        if class_name == "RecurrentConnectedConv2d":
            model = module(
                in_channels=input_shape[0],
                out_channels=6,
                kernel_size=3,
                dim_y=input_shape[1],
                dim_x=input_shape[2],
            )
        elif class_name == "InputAdaption":
            model = module()
        elif class_name == "SelfConnection":
            model = module()
        else:
            model = module(in_channels=input_shape[0], kernel_size=3)

        # Print number of parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {n_params}")

        # Print model summary
        if not class_name in ["SelfConnection", "InputAdaption"]:
            print(summary(model, input_shape))

        # Generate random input
        if class_name == "InputAdaption":
            require_grad = True
        else:
            require_grad = False
        random_input = torch.randn(
            batch_size, *input_shape, requires_grad=require_grad
        )

        # Perform forward pass
        output = model(random_input)

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
