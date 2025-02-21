from collections import deque

import torch
import torch.nn as nn

from dynvision.modules import (
    EulerStep,
    SupraLinearity,
    Retina,
    LightningBase,
    InputAdaption,
    RecurrentConnectedConv2d,
    SkipConnection,
    Feedback,
    SpatialWave,
)
from dynvision.utils.utils import alias_kwargs, str_to_bool


class FourLayerCNN(LightningBase):
    @alias_kwargs(
        trc="t_recurrence",
        tff="t_feedforward",
        rctype="recurrence_type",
        lr="learning_rate",
        taugrad="train_tau",
        supralin="supralinearity",
        winit="max_weight_init",
        inadapt="input_adaption_weight",
        ffonly="feedforward_only",
    )
    def __init__(
        self,
        n_classes: int = 200,
        input_dims: tuple = (14, 3, 64, 64),  # (t, c, y, x)
        dt: float = 2,  # ms
        tau: float = 8,  # ms
        t_feedforward: float = 10,  # ms
        t_recurrence: float = 6,  # ms
        recurrence_type: str = "none",
        train_tau: bool = False,
        bias: bool = True,
        max_weight_init: float = 0.001,
        supralinearity: float = 1,
        input_adaption_weight: float = 0,
        skip=False,
        feedback=False,
        feedforward_only=False,
        **kwargs,
    ) -> None:

        t_recurrence = float(t_recurrence)
        t_feedforward = float(t_feedforward)
        tau = float(tau)
        dt = float(dt)
        bias = str_to_bool(bias)
        train_tau = str_to_bool(train_tau)
        skip = str_to_bool(skip)
        feedback = str_to_bool(feedback)
        feedforward_only = str_to_bool(feedforward_only)
        input_adaption_weight = float(input_adaption_weight)

        self.delay_ff = int(t_feedforward / dt)
        self.delay_rc = int(t_recurrence / dt)

        model_args = {
            k: v for k, v in locals().items() if k not in ["self", "kwargs"]
        } | kwargs
        super(FourLayerCNN, self).__init__(**model_args)

        # make all parameters trainable
        self.trainable_parameter_names = [p for p in list(self.state_dict().keys())]

    def reset(self):
        self.V1.reset()
        self.tstep_V1.reset()
        self.V2.reset()
        self.tstep_V2.reset()
        self.V4.reset()
        self.tstep_V4.reset()
        self.IT.reset()
        self.tstep_IT.reset()
        if hasattr(self, "input_adaption"):
            self.input_adaption.reset()

    def _define_architecture(self):
        self.layer_names = ["V1", "V2", "V4", "IT"]
        # define operations order within layer
        self.layer_operations = [
            "layer",  # apply (recurrent) convolutional layer
            "addskip",  # add skip connection
            "addext",  # add external input
            "tstep",  # apply dynamical systems ode solver step
            "nonlin",  # apply nonlinearity
            "supralin",  # apply supralinearity
            "record",  # record activations in responses dict
            "delay",  # set and get delayed activations for next layer
            "pool",  # apply pooling
            "norm",  # apply normalization
        ]
        # Define traveling wave
        # if self.add_wave:
        #     timeresolution = 0.02  # s / timestep
        #     spatialresolution = 0.01  # m / pixel

        #     self.wave = SpatialWave(
        #         wavelength=None,  # 0.01 / spatialresolution,
        #         speed=1 * timeresolution / spatialresolution,  # pixel / timestep
        #         frequency=2 * timeresolution,  # 1 / timestep
        #         amplitude=1,
        #         offset=-0.3,
        #         direction=-1 / 2,
        #     )

        self.nonlin = nn.ReLU(inplace=False)
        if hasattr(self, "supralinearity") and float(self.supralinearity) != 1:
            self.supralin = SupraLinearity(
                n=float(self.supralinearity), requires_grad=False
            )

        # Define the convolutional layers
        layer_params = dict(
            bias=self.bias,
            recurrence_type=self.recurrence_type,
            dt=self.dt,
            tau=self.tau,
            history_length=self.t_feedforward,
            recurrence_delay=self.t_recurrence,
            max_weight_init=self.max_weight_init,
            parametrization=lambda x: x,  # nn.utils.parametrizations.weight_norm,
            device=self.device,
            feedforward_only=self.feedforward_only,
        )

        # Input adaption
        if self.input_adaption_weight:
            self.input_adaption = InputAdaption(
                fixed_weight=self.input_adaption_weight
            )

        # Retina
        # self.bottleneck_channels = 18
        # self.Retina = Retina(in_channels=self.n_channels, out_channels=self.bottleneck_channels, mid_channels=36, kernel_size=9, bias=self.bias)

        # V1
        self.V1 = RecurrentConnectedConv2d(
            in_channels=self.n_channels,  # self.Retina.out_channels,
            out_channels=36,
            kernel_size=3,
            stride=1,
            dim_y=self.dim_y,
            dim_x=self.dim_x,
            **layer_params,
        )
        self.tau_V1 = torch.nn.Parameter(
            torch.tensor(self.tau, dtype=float, device=self.device),
            requires_grad=self.train_tau,
        )
        self.tstep_V1 = EulerStep(dt=self.dt, tau=self.tau_V1, device=self.device)
        self.pool_V1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # V2
        self.V2 = RecurrentConnectedConv2d(
            in_channels=self.V1.out_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            dim_y=self.V1.dim_y // self.V1.stride // self.pool_V1.stride,
            dim_x=self.V1.dim_x // self.V1.stride // self.pool_V1.stride,
            **layer_params,
        )
        self.tau_V2 = torch.nn.Parameter(
            torch.tensor(self.tau, dtype=float, device=self.device),
            requires_grad=self.train_tau,
        )
        self.tstep_V2 = EulerStep(dt=self.dt, tau=self.tau_V2, device=self.device)
        self.pool_V2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # V4
        self.V4 = RecurrentConnectedConv2d(
            in_channels=self.V2.out_channels,
            out_channels=121,
            kernel_size=3,
            stride=1,
            dim_y=self.V2.dim_y // self.V2.stride // self.pool_V2.stride,
            dim_x=self.V2.dim_x // self.V2.stride // self.pool_V2.stride,
            **layer_params,
        )
        if self.skip:
            self.addskip_V4 = SkipConnection(
                source=self.V1,
                auto_adapt=True,
            )
        if self.feedback:
            self.addskip_V1 = Feedback(
                source=self.V4,
                auto_adapt=True,
            )
        self.tau_V4 = torch.nn.Parameter(
            torch.tensor(
                self.tau,
                dtype=float,
                device=self.device,
            ),
            requires_grad=self.train_tau,
        )
        self.tstep_V4 = EulerStep(dt=self.dt, tau=self.tau_V4, device=self.device)
        self.pool_V4 = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

        # IT
        self.IT = RecurrentConnectedConv2d(
            in_channels=self.V4.out_channels,
            out_channels=256,
            kernel_size=3,
            stride=2,
            dim_y=self.V4.dim_y // self.V4.stride // self.pool_V4.stride,
            dim_x=self.V4.dim_x // self.V4.stride // self.pool_V4.stride,
            **layer_params,
        )
        if self.skip:
            self.addskip_IT = SkipConnection(
                source=self.V2,
                auto_adapt=True,
            )
        if self.feedback:
            self.addskip_V2 = Feedback(
                source=self.IT,
                auto_adapt=True,
            )
        self.tau_IT = torch.nn.Parameter(
            torch.tensor(
                self.tau,
                dtype=float,
                device=self.device,
            ),
            requires_grad=self.train_tau,
        )
        self.tstep_IT = EulerStep(dt=self.dt, tau=self.tau_IT, device=self.device)

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.IT.out_channels, self.n_classes),
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
    input_shape = (20, 3, 64, 64)

    model = FourLayerCNN(input_dims=input_shape, n_classes=200, store_responses=True)

    random_input = torch.randn(1, *input_shape)

    t = 0
    output = model._forward(random_input[:, t, ...])

    outputs = model(random_input).squeeze()  # remove batch dimension

    print(f"Random Input ({random_input.shape})")
    print(f"Residual Timesteps: {model.n_residual_timesteps}")
    print(f"Model Output: {outputs.shape}")
    for t, output in enumerate(outputs):
        print(f"\tt {t}: ({output.shape})")
