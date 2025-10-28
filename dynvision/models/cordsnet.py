import requests
import torch
import torch.nn as nn

from dynvision.base import BaseModel

from dynvision.model_components import (
    EulerStep,
    RecurrentConnectedConv2d,
    Skip,
)
from dynvision.project_paths import project_paths
from dynvision.utils import alias_kwargs, str_to_bool

__all__ = ["CordsNet"]


class CordsNet(BaseModel):
    """
    CordsNet: Contextual Recurrent Deep Structured Network

    Reimplementation of CordsNet using the DynVision framework.

    Dynamics:
        The original CordsNet uses the update rule:
            r_new = (1 - alpha) * r_old + alpha * relu(recurrent(r) + feedforward(input))

        This is equivalent to Euler integration with dt/tau = alpha:
            r_new = r_old + (dt/tau) * (f(r, input) - r_old)
                  = r_old * (1 - dt/tau) + (dt/tau) * f(r, input)

        To match a specific alpha value from the original, set dt and tau such that:
            alpha = dt / tau

        Examples:
            - alpha=0.1: dt=1, tau=10 (default)
            - alpha=0.2: dt=2, tau=10 or dt=1, tau=5
            - alpha=0.05: dt=1, tau=20

    Args:
        n_classes (int): Number of output classes
        input_dims (tuple): Input dimensions (t, c, y, x)
        dt (float): Integration time step in ms
        tau (float): Time constant in ms (controls alpha = dt/tau)
        t_feedforward (float): Feedforward delay in ms
        t_recurrence (float): Recurrence delay in ms
        recurrence_type (str): Type of recurrent connection
        dynamics_solver (str): ODE solver type
        bias (bool): Whether to use bias in convolutions
        idle_timesteps (int): Number of timesteps to run without input for spontaneous activity
    """

    @alias_kwargs(
        tff="t_feedforward",
        trc="t_recurrence",
        rctype="recurrence_type",
        solver="dynamics_solver",
    )
    def __init__(
        self,
        n_classes=1000,
        input_dims=(20, 3, 224, 224),  # (t, c, y, x)
        dt=1,  # ms
        tau=10,  # ms (alpha = dt/tau = 0.1 by default)
        t_feedforward=1,  # ms
        t_recurrence=1,  # ms
        recurrence_type="full",
        dynamics_solver="euler",
        bias=True,
        idle_timesteps=100,  # Run 100 timesteps without input for spontaneous activity
        **kwargs,
    ) -> None:

        super().__init__(
            n_classes=n_classes,
            input_dims=input_dims,
            t_recurrence=float(t_recurrence),
            t_feedforward=float(t_feedforward),
            t_feedback=0.0,  # Not used in CordsNet
            t_skip=0.0,  # Not used in CordsNet
            tau=float(tau),
            dt=float(dt),
            bias=str_to_bool(bias),
            recurrence_type=recurrence_type,
            dynamics_solver=dynamics_solver,
            idle_timesteps=int(idle_timesteps),
            **kwargs,
        )
        self.delay_ff = int(t_feedforward / dt)
        self.delay_rc = int(t_recurrence / dt)
        self._define_architecture()
        self._init_parameters()

    def setup(self, stage):
        super().setup(stage)

    def _init_parameters(self):
        # Load pretrained weights
        self.load_pretrained_state_dict(check_mismatch_layer=["classifier.2"])

        # make only the classifier trainable
        self.trainable_parameter_names = [
            p for p in list(self.state_dict().keys()) if "classifier.2" in p
        ]

    def download_pretrained_state_dict(self, version=0):
        url = "https://github.com/wmws2/cordsnet/blob/main/cordsnetr8.pth"
        save_path = project_paths.models / "CordsNet" / "CordsNet_pretrained.pt"

        if not save_path.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded pretrained model to {save_path}")
            else:
                raise Exception(f"Failed to download file from {url}")
        else:
            print(f"Pretrained model already exists at {save_path}")

        state_dict = torch.load(save_path, map_location=self.device)

        # Original pretrained model has spatial biases [channels, height, width]
        # Keep them as is (don't average to [channels])

        return state_dict

    def translate_pretrained_layer_names(self):
        """
        Map original CordsNet pretrained weight names to new structure.

        Original structure:
        - area_conv[i]: recurrent connection for layer i
        - area_area[i]: feedforward connection for layer i
        - conv_bias[i]: spatial bias for layer i
        - inp_conv: initial convolution
        - inp_skip: skip to layer 2
        - skip_area[0,1,2]: skip connections to layers 4, 6, 8
        """
        translate_layer_names = {
            "inp_conv": "layer_inp",
            "inp_skip": "skip_layer2",
            "skip_area.0": "skip_layer4",
            "skip_area.1": "skip_layer6",
            "skip_area.2": "skip_layer8",
            "out_fc": "classifier.2",
        }

        # Map area_conv (recurrent) and area_area (feedforward) to layer structure
        for i in range(8):
            layer_name = f"layer{i+1}"
            # Recurrent connection
            translate_layer_names[f"area_conv.{i}"] = f"{layer_name}.recurrence.conv"
            # Feedforward connection
            translate_layer_names[f"area_area.{i}"] = f"{layer_name}.conv"
            # Spatial bias
            translate_layer_names[f"conv_bias.{i}"] = f"bias_{layer_name}"

        return translate_layer_names

    def reset(self):
        for layer_name in self.layer_names:
            getattr(self, layer_name).reset()
            getattr(self, f"tstep_{layer_name}").reset()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Override forward to handle the custom CordsNet temporal processing.

        The BaseModel's forward expects to call _forward once per timestep,
        but CordsNet processes all timesteps in a batch with layers in reverse order.
        """
        # Adjust input dimensions if needed
        if x.ndim == 4:  # (batch, channels, height, width)
            # Add timestep dimension
            x = x.unsqueeze(1)  # (batch, 1, channels, height, width)

        # Extract first frame (CordsNet processes single images for all timesteps)
        if x.size(1) > 1:
            # Use first frame
            x = x[:, 0]  # (batch, channels, height, width)
        else:
            x = x.squeeze(1)  # (batch, channels, height, width)

        # Reset states
        self.reset()

        # Call custom _forward that handles all timesteps internally
        output, responses = self._forward(x)

        return output

    def _get_layer_input(self, layer_idx: int, r: list, inp: torch.Tensor):
        """
        Get input for a specific layer based on the skip connection pattern.

        Pattern from original CordsNet:
        - layer1 (idx 0): input from inp
        - layer2 (idx 1): relu(r[0]) + skip(inp)
        - layer3 (idx 2): relu(r[1]) + relu(r[0])
        - layer4 (idx 3): relu(r[2]) + skip(relu(r[1]))
        - layer5 (idx 4): relu(r[3]) + relu(r[2])
        - layer6 (idx 5): relu(r[4]) + skip(relu(r[3]))
        - layer7 (idx 6): relu(r[5]) + relu(r[4])
        - layer8 (idx 7): relu(r[6]) + skip(relu(r[5]))
        """
        nonlin = nn.ReLU(inplace=False)

        if layer_idx == 0:  # layer1
            return inp
        elif layer_idx == 1:  # layer2
            return nonlin(r[0]) + self.skip_layer2(inp)
        elif layer_idx == 2:  # layer3
            return nonlin(r[1]) + nonlin(r[0])
        elif layer_idx == 3:  # layer4
            return nonlin(r[2]) + self.skip_layer4(nonlin(r[1]))
        elif layer_idx == 4:  # layer5
            return nonlin(r[3]) + nonlin(r[2])
        elif layer_idx == 5:  # layer6
            return nonlin(r[4]) + self.skip_layer6(nonlin(r[3]))
        elif layer_idx == 6:  # layer7
            return nonlin(r[5]) + nonlin(r[4])
        elif layer_idx == 7:  # layer8
            return nonlin(r[6]) + self.skip_layer8(nonlin(r[5]))
        else:
            raise ValueError(f"Invalid layer index: {layer_idx}")

    def _forward(
        self,
        x: torch.Tensor = None,
        t: int = None,
        feedforward_only: bool = False,
        store_responses: bool = True,
    ):
        """
        Custom forward pass matching original CordsNet dynamics.

        The original CordsNet:
        1. Runs idle_timesteps without input for spontaneous activity
        2. Runs n_timesteps with input
        3. Processes layers in reverse order (layer8→layer1) per timestep
        4. Uses update rule: r = (1-alpha)*r + alpha*relu(recurrent(r) + feedforward(input) + bias)
        5. Outputs: avgpool(relu(layer8) + relu(layer7))
        """
        if x is None:
            return None, {}

        batch_size = x.size(0)
        responses = {}

        # Process input through initial layers
        inp = self.pool_layer_inp(self.layer_inp(x))

        # Initialize layer states
        channels = [64, 64, 64, 128, 128, 256, 256, 512, 512]
        sizes = [56, 56, 28, 28, 14, 14, 7, 7]
        r = []
        for i in range(8):
            r.append(torch.zeros(batch_size, channels[i+1], sizes[i], sizes[i], device=x.device, dtype=x.dtype))

        # Run idle timesteps for spontaneous activity (without input)
        with torch.no_grad():
            for t in range(self.idle_timesteps):
                for layer_idx in range(7, -1, -1):  # Process in reverse order (7→0)
                    layer_name = f"layer{layer_idx + 1}"
                    layer = getattr(self, layer_name)
                    nonlin = getattr(self, f"nonlin_{layer_name}")
                    bias = getattr(self, f"bias_{layer_name}")
                    tstep = getattr(self, f"tstep_{layer_name}")

                    # Get input for this layer (using zero input for idle)
                    layer_input = self._get_layer_input(layer_idx, r, inp * 0)

                    # Apply layer (feedforward + recurrence)
                    x_layer = layer(layer_input)

                    # Apply bias and nonlinearity
                    x_layer = nonlin(x_layer + bias)

                    # Temporal integration
                    r[layer_idx] = tstep(r[layer_idx], x_layer)

        # Run regular timesteps with input
        for t in range(self.n_timesteps):
            for layer_idx in range(7, -1, -1):  # Process in reverse order (7→0)
                layer_name = f"layer{layer_idx + 1}"
                layer = getattr(self, layer_name)
                nonlin = getattr(self, f"nonlin_{layer_name}")
                bias = getattr(self, f"bias_{layer_name}")
                tstep = getattr(self, f"tstep_{layer_name}")

                # Get input for this layer
                layer_input = self._get_layer_input(layer_idx, r, inp)

                # Apply layer (feedforward + recurrence)
                x_layer = layer(layer_input)

                # Apply bias and nonlinearity
                x_layer = nonlin(x_layer + bias)

                # Temporal integration
                r[layer_idx] = tstep(r[layer_idx], x_layer)

        # Store responses if requested
        if store_responses:
            for i, layer_name in enumerate(self.layer_names):
                responses[layer_name] = r[i]

        # Compute output: avgpool(relu(layer8) + relu(layer7))
        summed_output = torch.relu(r[7]) + torch.relu(r[6])
        output = self.classifier(summed_output)

        if store_responses:
            responses[self.classifier_name] = output

        return output, responses

    def _define_architecture(self):
        # Faithful 8-layer structure matching original CordsNet
        # layer1: 64→64 (stride 1, size 56x56)
        # layer2: 64→64 (stride 1, size 56x56)
        # layer3: 64→128 (stride 2, size 28x28)
        # layer4: 128→128 (stride 1, size 28x28)
        # layer5: 128→256 (stride 2, size 14x14)
        # layer6: 256→256 (stride 1, size 14x14)
        # layer7: 256→512 (stride 2, size 7x7)
        # layer8: 512→512 (stride 1, size 7x7)

        self.layer_names = ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7", "layer8"]
        self.layer_operations = [
            "addskip",  # apply skip connection
            "layer",  # apply (recurrent) convolutional layer
            "nonlin",  # apply nonlinearity (ReLU before temporal integration)
            "addbias",  # apply spatial (per-unit) bias
            "tstep",  # apply dynamical systems ode solver step
            "record",  # record activations in responses dict
            "delay",  # set and get delayed activations for next layer
        ]

        # Channel configuration (matching original)
        channels = [64, 64, 64, 128, 128, 256, 256, 512, 512]
        sizes = [56, 56, 28, 28, 14, 14, 7, 7]
        strides = [1, 1, 2, 1, 2, 1, 2, 1]

        # Input layer
        self.layer_inp = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(
                in_channels=self.n_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        )
        self.pool_layer_inp = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False
        )

        # Common layer parameters
        common_layer_params = dict(
            bias=False,  # Bias disabled - using spatial bias instead
            recurrence_type=self.recurrence_type,
            dt=self.dt,
            tau=self.tau,
            history_length=self.t_feedforward,
            t_recurrence=self.t_recurrence,
            parametrization=nn.utils.parametrizations.weight_norm,
            device=self.device,
        )

        # Create 8 recurrent layers
        for i in range(8):
            layer_name = f"layer{i+1}"

            # Create recurrent convolutional layer
            layer = RecurrentConnectedConv2d(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=3,
                stride=strides[i],
                **common_layer_params,
            )
            setattr(self, layer_name, layer)

            # Create nonlinearity
            setattr(self, f"nonlin_{layer_name}", nn.ReLU(inplace=False))

            # Create temporal integration step
            setattr(self, f"tstep_{layer_name}", EulerStep(dt=self.dt, tau=self.tau))

            # Create spatial bias parameter
            bias = nn.Parameter(torch.zeros(channels[i+1], sizes[i], sizes[i]))
            setattr(self, f"bias_{layer_name}", bias)

        # Skip connections (matching original structure)
        # inp_skip: from input (64 channels) to layer2 (added to layer2 input)
        self.skip_layer2 = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        )

        # skip_area[0]: from layer2 (64ch) to layer4 (128ch), stride=2
        self.skip_layer4 = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0, bias=False)
        )

        # skip_area[1]: from layer4 (128ch) to layer6 (256ch), stride=2
        self.skip_layer6 = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0, bias=False)
        )

        # skip_area[2]: from layer6 (256ch) to layer8 (512ch), stride=2
        self.skip_layer8 = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0, bias=False)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.parametrizations.weight_norm(
                nn.Linear(512, self.n_classes, bias=True),
            ),
        )
        return None


if __name__ == "__main__":
    input_shape = (20, 3, 224, 224)

    model = CordsNet(
        input_dims=input_shape,
        n_classes=10,
        store_responses=True,
        tff=1,
        trc=1,
        tau=10,
        dt=1,
    )
    model.setup("fit")

    # print(torchsummary.summary(model, input_size=input_shape[1:]))
    # print(model.state_dict().keys())

    random_input = torch.randn(1, *input_shape)

    t = 0
    output = model.forward(random_input)

    outputs = model(random_input).squeeze()  # remove batch dimension

    print(f"Random Input ({random_input.shape})")
    print(f"Residual Timesteps: {model.n_residual_timesteps}")
    print(f"Model Output: {outputs.shape}")
