import torch
import torch.nn as nn
import torch.nn.functional as F
from dynvision.model_components import LightningBase


class BLConvLayer(nn.Module):
    """BL recurrent convolutional layer

    Note that this is NOT A KERAS LAYER but is an object containing PyTorch layers

    Args:
        filters: Int, number of output filters in convolutions
        kernel_size: Int or tuple/list of 2 integers, specifying the height and
            width of the 2D convolution window. Can be a single integer to
            specify the same value for all spatial dimensions.
        layer_name: String, prefix for layers in the RCL
    """

    def __init__(self, in_channels, filters, kernel_size, layer_name):
        super(BLConvLayer, self).__init__()
        # Initialise convolutional layers
        self.b_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.l_conv = nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        # Holds the most recent bottom-up conv
        # Useful when the bottom-up input does not change, e.g. input image
        self.previous_b_conv = None

    def forward(self, b_input=None, l_input=None):
        conv_list = []

        if b_input is not None:
            # Run bottom-up conv and save result
            conv_list.append(self.b_conv(b_input))
            self.previous_b_conv = conv_list[-1]
        elif self.previous_b_conv is not None:
            # Use the most recent bottom-up conv
            conv_list.append(self.previous_b_conv)
        else:
            raise ValueError("b_input must be given on first pass")

        # Run lateral conv
        if l_input is not None:
            conv_list.append(self.l_conv(l_input))

        # Return element-wise sum of convolutions
        return sum(conv_list)


class BLNet(LightningBase):
    def __init__(
        self, in_channels, classes, n_timesteps=8, cumulative_readout=False, **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            classes=classes,
            n_timesteps=n_timesteps,
            cumulative_readout=cumulative_readout,
            **kwargs,
        )
        self._define_architecture()

    def _define_architecture(self):
        self.layers = nn.ModuleList(
            [
                BLConvLayer(self.in_channels, 96, 7, "RCL_0"),
                BLConvLayer(96, 128, 5, "RCL_1"),
                BLConvLayer(128, 192, 3, "RCL_2"),
                BLConvLayer(192, 256, 3, "RCL_3"),
                BLConvLayer(256, 512, 3, "RCL_4"),
                BLConvLayer(512, 1024, 3, "RCL_5"),
                BLConvLayer(1024, 2048, 1, "RCL_6"),
            ]
        )

        self.readout_dense = nn.Linear(2048, self.classes)

    def forward(self, x):
        activations = [[] for _ in range(self.n_timesteps)]
        presoftmax = [None] * self.n_timesteps
        outputs = [None] * self.n_timesteps

        for t in range(self.n_timesteps):
            for n, layer in enumerate(self.layers):
                if n == 0:
                    b_input = x if t == 0 else None
                else:
                    b_input = F.max_pool2d(activations[t][n - 1], kernel_size=(2, 2))

                l_input = None if t == 0 else activations[t - 1][n]
                x_tn = layer(b_input, l_input)
                x_tn = nn.BatchNorm2d(x_tn.size(1))(x_tn)
                x_tn = F.relu(x_tn)
                activations[t].append(x_tn)

            x = F.adaptive_avg_pool2d(activations[t][-1], output_size=(1, 1))
            x = x.view(x.size(0), -1)
            presoftmax[t] = self.readout_dense(x)

            if self.cumulative_readout and t > 0:
                x = sum(presoftmax[: t + 1])
            else:
                x = presoftmax[t]

            outputs[t] = F.softmax(x, dim=1)

        return outputs


if __name__ == "__main__":
    input_shape = (1, 112, 112)
    in_channels = input_shape[0]

    model = BLNet(
        in_channels=in_channels, classes=10, n_timesteps=8, cumulative_readout=False
    )

    # print(summary(model, input_shape))

    random_input = torch.randn(2, *input_shape)

    outputs = model(random_input)

    print(f"Random Input ({random_input.shape})")
    print(f"Time Steps: {len(outputs)}")
    for t, output in enumerate(outputs):
        print(f"Model Output t{t} ({output.shape}):\n\t{output}")

    # import matplotlib.pyplot as plt
    # import numpy as np
    # out = np.stack([o.detach().numpy() for o in outputs]).squeeze()
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(out)
    #
    # plt.show()
