import torch
import torch.nn as nn
from torchsummary import summary

__all__ = ["Retina"]


class Retina(nn.Module):
    """
    Model of the retina and LGN as two convolutional layers (with ReLU activation in between). The out_channels of the second layer are the bottleneck_channel and less than the mid_channels (out_channels of the first layer). See Lindsey et al. (2019) doi:10.48550/arXiv.1901.00945 for more details.
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 36,
        out_channels: int = 18,
        kernel_size: int = 9,
        bias: bool = True,
    ) -> None:
        super(Retina, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.bias = bias

        self._define_architecture()
        self._init_parameters()

    def _define_architecture(self):
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            bias=self.bias,
        )

        self.conv2 = nn.Conv2d(
            in_channels=self.mid_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            bias=self.bias,
        )

        self.nonlin = nn.ReLU(inplace=True)

    def _init_parameters(self):
        for layer in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.nonlin(x)
        x = self.conv2(x)
        return x


if __name__ == "__main__":
    input_shape = (3, 32, 32)
    batch_size = 1
    print("Input shape:", (batch_size, *input_shape))

    for class_name in __all__:
        print(f"Testing {class_name}:")

        # Create an instance of the class
        module = globals()[class_name]
        model = module()

        # Print number of parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {n_params}")

        # Print model summary
        print(summary(model, input_shape))

        # Generate random input
        random_input = torch.randn(batch_size, *input_shape, requires_grad=True)

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
