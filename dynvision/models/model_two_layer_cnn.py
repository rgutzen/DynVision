import torch
import torch.nn as nn

from dynvision.modules import (
    LightningBase,
    EulerStep,
    InputAdaption,
    RecurrentConnectedConv2d,
    SpatialWave,
)
from dynvision.utils.utils import alias_kwargs, str_to_bool


class TwoLayerCNN(LightningBase):
    @alias_kwargs(
        trc="t_recurrence",
        tff="t_feedforward",
        rctype="recurrence_type",
        lr="learning_rate",
    )
    def __init__(
        self,
        n_features1=16,
        n_features2=36,
        n_classes=10,
        input_dims=(20, 1, 28, 28),
        stride1=1,
        stride2=1,
        kernel_size=3,
        bias=True,
        dt=2,  # ms
        tau=8,  # ms
        t_recurrence=6,  # ms
        t_feedforward=10,  # ms
        store_responses=True,
        recurrence_type="local",
        **kwargs,
    ):
        t_recurrence = float(t_recurrence)
        t_feedforward = float(t_feedforward)
        tau = float(tau)
        dt = float(dt)
        bias = str_to_bool(bias)

        self.delay_ff = int(t_feedforward / dt)
        self.delay_rc = int(t_recurrence / dt)

        model_args = {
            k: v for k, v in locals().items() if k not in ["self", "kwargs"]
        } | kwargs
        super(TwoLayerCNN, self).__init__(**model_args)

        # make all parameters trainable
        self.trainable_parameter_names = [p for p in list(self.state_dict().keys())]

        self.reset()

    def reset(self):
        self.layer1.reset()
        self.tstep_layer1.reset()
        self.layer2.reset()
        self.tstep_layer2.reset()

    def _define_architecture(self):
        self.layer_names = ["layer1", "layer2"]
        self.layer_operations = [
            "layer",  # apply (recurrent) convolutional layer
            "tstep",  # apply dynamical systems ode solver step
            "nonlin",  # apply nonlinearity
            "record",  # record activations in responses dict
            "delay",  # set and get delayed activations for next layer
            "pool",  # apply pooling
        ]

        # Input adaption
        # self.input_adaption = InputAdaption()

        # Define the convolutional layers
        layer_params = dict(
            bias=self.bias,
            recurrence_type=self.recurrence_type,
            dt=self.dt,
            tau=self.tau,
            history_length=self.t_feedforward,
            recurrence_delay=self.t_recurrence,
            parametrization=lambda x: x,  # nn.utils.parametrizations.weight_norm,
            device=self.device,
        )

        # Define the convolutional layers
        self.layer1 = RecurrentConnectedConv2d(
            in_channels=self.n_channels,
            out_channels=self.n_features1,
            kernel_size=self.kernel_size,
            stride=self.stride1,
            dim_y=self.dim_y,
            dim_x=self.dim_x,
            **layer_params,
        )
        self.tstep_layer1 = EulerStep(dt=self.dt, tau=self.tau)
        self.nonlin_layer1 = nn.ReLU(inplace=False)
        self.pool_layer1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer2 = RecurrentConnectedConv2d(
            in_channels=self.n_features1,
            out_channels=self.n_features2,
            kernel_size=self.kernel_size,
            stride=self.stride2,
            dim_y=self.layer1.dim_y // self.layer1.stride // self.pool_layer1.stride,
            dim_x=self.layer1.dim_x // self.layer1.stride // self.pool_layer1.stride,
            **layer_params,
        )
        self.tstep_layer2 = EulerStep(dt=self.dt, tau=self.tau)
        self.nonlin_layer2 = nn.ReLU(inplace=False)

        # Define the fully connected layer
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.layer2.out_channels, self.n_classes),
        )

    def _init_parameters(self):
        super()._init_parameters()

        trunc_normal_params = dict(mean=0.0, std=0.004)

        for layer_name in self.layer_names:
            layer = getattr(self, layer_name)
            if hasattr(layer, "conv.weight") and not hasattr(
                layer, "conv.weight.original1"
            ):
                nn.init.trunc_normal_(layer.conv.weight, **trunc_normal_params)

        nn.init.trunc_normal_(self.classifier[-1].weight, **trunc_normal_params)
        nn.init.constant_(self.classifier[-1].bias, 0)


if __name__ == "__main__":
    input_shape = (20, 1, 28, 28)

    model = TwoLayerCNN(input_dims=input_shape, n_classes=10, store_responses=True)

    random_input = torch.randn(*input_shape)

    outputs = model(random_input)

    print(f"Random Input ({random_input.shape})")
    print(f"Residual timesteps: {model.n_residual_timesteps}")
    print(f"Model Output Timesteps: {len(outputs)}")
    for t, output in enumerate(outputs):
        print(f"\tt {t}: ({output.shape})")
