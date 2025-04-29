"""
https://huggingface.co/novelmartis/blt_vs_model/resolve/main/blt_vs_slt_111_biounroll_0_t_6_readout_multi_dataset_imagenet_num_1.pth
"""

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dynvision.model_components import LightningBase
from dynvision.project_paths import project_paths

__all__ = ["BLT", "BL", "BT", "B", "L"]


class BLT(LightningBase):
    """
    BLT model adapted from the original BLT_VS model to use LightningBase.
    Simulates the ventral stream of the visual cortex.
    """

    def __init__(
        self,
        input_dims: tuple = (20, 3, 224, 224),
        n_classes: int = 1000,  # Default to ImageNet classes to match pretrained weights
        recurrence_type: str = "none",
        fixed_self_weight: float = None,
        t_feedforward: float = 0,
        t_recurrence: float = 0,
        dt: float = 1,
        add_feats: int = 100,
        lateral_connections: bool = True,
        topdown_connections: bool = True,
        skip_connections: bool = True,
        hook_type: str = "None",
        **kwargs,
    ) -> None:
        super().__init__(
            input_dims=input_dims,
            n_classes=n_classes,
            recurrence_type=recurrence_type,
            fixed_self_weight=fixed_self_weight,
            t_feedforward=t_feedforward,
            t_recurrence=t_recurrence,
            dt=dt,
            add_feats=add_feats,
            lateral_connections=lateral_connections,
            topdown_connections=topdown_connections,
            skip_connections=skip_connections,
            hook_type=hook_type,
            image_size=input_dims[-1],  # Assuming square input
            **kwargs,
        )

        if self.image_size not in [224, 128]:
            raise ValueError("Image size must be 224 or 128.")

        self.n_classes = 1000
        # Define network areas and configurations
        self._setup_architecture_params()
        self._define_architecture()
        self._init_parameters()

    def _init_parameters(self):
        #  Load pretrained weights
        self.load_pretrained_state_dict(
            check_mismatch_layer=["classifier.readout_conv"]
        )
        # make only the classifier trainable
        self.trainable_parameter_names = [
            p for p in list(self.state_dict().keys()) if "classifier.readout_conv" in p
        ]

    def download_pretrained_state_dict(self):
        """Downloads pretrained weights for the BLT model from HuggingFace."""
        url = "https://huggingface.co/novelmartis/blt_vs_model/resolve/main/blt_vs_slt_111_biounroll_0_t_6_readout_multi_dataset_imagenet_num_1.pth"
        save_path = project_paths.models / "BLT" / "BLT_pretrained.pt"

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

        state_dict = torch.load(save_path, map_location="cpu")
        return state_dict

    def translate_pretrained_layer_names(self):
        translate_layer_names = {
            "connections.Retina": "Retina",
            "connections.LGN": "LGN",
            "connections.V1": "V1",
            "connections.V2": "V2",
            "connections.V3": "V3",
            "connections.V4": "V4",
            "connections.LOC": "LOC",
            "connections.Readout": "classifier",
        }
        return translate_layer_names

    def _setup_architecture_params(self):
        """Setup architecture parameters based on image size."""
        if self.image_size == 224:
            self.kernel_sizes = [7, 7, 5, 1, 5, 3, 3, 5]
            self.kernel_sizes_lateral = [0, 0, 5, 5, 5, 5, 5, 0]
        else:  # 128
            self.kernel_sizes = [5, 3, 3, 1, 3, 3, 3, 3]
            self.kernel_sizes_lateral = [0, 0, 3, 3, 3, 3, 3, 0]

        self.strides = [2, 2, 2, 1, 1, 1, 2, 2]
        self.paddings = (np.array(self.kernel_sizes) - 1) // 2
        self.channel_sizes = [
            32,
            32,
            576,
            480,
            352,
            256,
            352,
            int(self.n_classes + self.add_feats),
        ]

        # Top-down connections configuration
        self.topdown_connections_layers = [
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ]

    def _define_architecture(self):
        self.layer_names = [
            "Retina",
            "LGN",
            "V1",
            "V2",
            "V3",
            "V4",
            "LOC",
            "classifier",
        ]

        # Initialize network layers
        for i, layer_name in enumerate(self.layer_names[:-1]):
            setattr(
                self,
                layer_name,
                BLT_VS_Layer(
                    layer_n=i,
                    channel_sizes=self.channel_sizes,
                    strides=self.strides,
                    kernel_sizes=self.kernel_sizes,
                    kernel_sizes_lateral=self.kernel_sizes_lateral,
                    paddings=self.paddings,
                    lateral_connections=self.lateral_connections
                    and (self.kernel_sizes_lateral[i] > 0),
                    topdown_connections=self.topdown_connections
                    and self.topdown_connections_layers[i],
                    skip_connections_bu=self.skip_connections and (i == 5),
                    skip_connections_td=self.skip_connections and (i == 2),
                    image_size=self.image_size,
                ),
            )

        self.classifier = BLT_VS_Readout(
            layer_n=7,
            channel_sizes=self.channel_sizes,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            num_classes=self.n_classes,
        )

    def _forward(self, x, t=None, **kwargs):
        if t == 0:
            self.readout_output = None
            self.bu_activations = [None for _ in self.layer_names]
            self.td_activations = [None for _ in self.layer_names]
            self.bu_activations_old = [None for _ in self.layer_names]
            self.td_activations_old = [None for _ in self.layer_names]
            # Initial activation for Retina
            self.bu_activations_old[0], _ = self.Retina(bu_input=x)
            self.bu_activations[0] = self.bu_activations_old[0]

        else:
            # Update each area
            for idx, area in enumerate(self.layer_names[1:-1]):
                should_update = any(
                    [
                        self.bu_activations_old[idx] is not None,
                        (self.bu_activations_old[2] is not None and (idx + 1) == 5),
                        self.td_activations_old[idx + 2] is not None,
                        (self.td_activations_old[5] is not None and (idx + 1) == 2),
                    ]
                )

                if should_update:
                    bu_act, td_act = getattr(self, area)(
                        bu_input=self.bu_activations_old[idx],
                        bu_l_input=self.bu_activations_old[idx + 1],
                        td_input=self.td_activations_old[idx + 2],
                        td_l_input=self.td_activations_old[idx + 1],
                        bu_skip_input=(
                            self.bu_activations_old[2] if (idx + 1) == 5 else None
                        ),
                        td_skip_input=(
                            self.td_activations_old[5] if (idx + 1) == 2 else None
                        ),
                    )
                    self.bu_activations[idx + 1] = bu_act
                    self.td_activations[idx + 1] = td_act

            self.bu_activations_old = self.bu_activations[:]
            self.td_activations_old = self.td_activations[:]

            # Activate readout when LOC output is ready
            if self.bu_activations_old[-2] is not None:
                bu_act, td_act = self.classifier(bu_input=self.bu_activations_old[-2])
                self.readout_output = bu_act
                self.bu_activations[-1] = bu_act
                self.td_activations[-1] = td_act

        for i, layer_name in enumerate(self.layer_names[:-1]):
            activation = None
            if self.bu_activations[i] is not None:
                activation = self.bu_activations[i]
            if self.td_activations[i] is not None:
                activation += self.td_activations[i]

            responses = {layer_name: activation}

        responses["classifier"] = self.readout_output
        return self.readout_output, responses

        # if self.readout_type == "single":
        #     outputs = torch.stack(readout_output, dim=0)
        #     outputs = outputs.permute(1, 0, 2)
        #     readout_weights = F.softmax(self.readout_weights, dim=0)
        #     readout_weights = readout_weights.view(1, -1, 1)
        #     weighted_outputs = outputs * readout_weights
        #     return [weighted_outputs.sum(dim=1)]
        # else:
        #     return readout_output

    def reset(self):
        for area in self.layer_names[:-1]:
            getattr(self, area).reset()


# Keep the original layer implementations
class BLT_VS_Layer(nn.Module):
    """
    A single layer in the BLT model, representing a cortical area.
    """

    def __init__(
        self,
        layer_n,
        channel_sizes,
        strides,
        kernel_sizes,
        kernel_sizes_lateral,
        paddings,
        lateral_connections=True,
        topdown_connections=True,
        skip_connections_bu=False,
        skip_connections_td=False,
        image_size=224,
    ):
        super(BLT_VS_Layer, self).__init__()

        in_channels = 3 if layer_n == 0 else channel_sizes[layer_n - 1]
        out_channels = channel_sizes[layer_n]

        # Bottom-up convolution
        self.bu_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[layer_n],
            stride=strides[layer_n],
            padding=paddings[layer_n],
        )

        # Lateral connections
        if lateral_connections:
            kernel_size_lateral = kernel_sizes_lateral[layer_n]
            self.bu_l_conv_depthwise = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size_lateral,
                stride=1,
                padding="same",
                groups=out_channels,
            )
            self.bu_l_conv_pointwise = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.bu_l_conv_depthwise = NoOpModule()
            self.bu_l_conv_pointwise = NoOpModule()

        # Top-down connections
        if topdown_connections:
            self.td_conv = nn.ConvTranspose2d(
                in_channels=channel_sizes[layer_n + 1],
                out_channels=out_channels,
                kernel_size=kernel_sizes[layer_n + 1],
                stride=strides[layer_n + 1],
                padding=(kernel_sizes[layer_n + 1] - 1) // 2,
            )
            if lateral_connections:
                self.td_l_conv_depthwise = nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes_lateral[layer_n],
                    stride=1,
                    padding="same",
                    groups=out_channels,
                )
                self.td_l_conv_pointwise = nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            else:
                self.td_l_conv_depthwise = NoOpModule()
                self.td_l_conv_pointwise = NoOpModule()
        else:
            self.td_conv = NoOpModule()
            self.td_l_conv_depthwise = NoOpModule()
            self.td_l_conv_pointwise = NoOpModule()

        # Skip connections
        if skip_connections_bu:
            self.skip_bu_depthwise = nn.Conv2d(
                in_channels=channel_sizes[2],  # From V1
                out_channels=out_channels,
                kernel_size=7 if image_size == 224 else 5,
                stride=1,
                padding="same",
                groups=np.gcd(channel_sizes[2], out_channels),
            )
            self.skip_bu_pointwise = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.skip_bu_depthwise = NoOpModule()
            self.skip_bu_pointwise = NoOpModule()

        if skip_connections_td:
            self.skip_td_depthwise = nn.Conv2d(
                in_channels=channel_sizes[5],  # From V4
                out_channels=out_channels,
                kernel_size=3,  # V4 to V1 skip connection
                stride=1,
                padding="same",
                groups=np.gcd(channel_sizes[5], out_channels),
            )
            self.skip_td_pointwise = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.skip_td_depthwise = NoOpModule()
            self.skip_td_pointwise = NoOpModule()

        self.layer_norm_bu = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.layer_norm_td = nn.GroupNorm(num_groups=1, num_channels=out_channels)

    def forward(
        self,
        bu_input,
        bu_l_input=None,
        td_input=None,
        td_l_input=None,
        bu_skip_input=None,
        td_skip_input=None,
    ):
        """Forward pass for a single BLT_VS layer."""
        # Process bottom-up input
        bu_processed = self.bu_conv(bu_input) if bu_input is not None else 0

        # Process top-down input
        td_processed = (
            self.td_conv(td_input, output_size=bu_processed.size())
            if td_input is not None
            else 0
        )

        # Process bottom-up lateral input
        bu_l_processed = (
            self.bu_l_conv_pointwise(self.bu_l_conv_depthwise(bu_l_input))
            if bu_l_input is not None
            else 0
        )

        # Process top-down lateral input
        td_l_processed = (
            self.td_l_conv_pointwise(self.td_l_conv_depthwise(td_l_input))
            if td_l_input is not None
            else 0
        )

        # Process skip connections
        skip_bu_processed = (
            self.skip_bu_pointwise(self.skip_bu_depthwise(bu_skip_input))
            if bu_skip_input is not None
            else 0
        )
        skip_td_processed = (
            self.skip_td_pointwise(self.skip_td_depthwise(td_skip_input))
            if td_skip_input is not None
            else 0
        )

        # Compute sums
        bu_drive = bu_processed + bu_l_processed + skip_bu_processed
        bu_mod = bu_processed + skip_bu_processed
        td_drive = td_processed + td_l_processed + skip_td_processed
        td_mod = td_processed + skip_td_processed

        # Compute bottom-up output
        if isinstance(td_mod, torch.Tensor):
            if isinstance(bu_drive, torch.Tensor):
                bu_output = F.relu(bu_drive) * 2 * torch.sigmoid(td_mod)
            else:
                bu_output = torch.zeros_like(td_mod)
        else:
            bu_output = F.relu(bu_drive)

        # Compute top-down output
        if isinstance(bu_mod, torch.Tensor):
            if isinstance(td_drive, torch.Tensor):
                td_output = F.relu(td_drive) * 2 * torch.sigmoid(bu_mod)
            else:
                td_output = torch.zeros_like(bu_mod)
        else:
            td_output = F.relu(td_drive)

        bu_output = self.layer_norm_bu(bu_output)
        td_output = self.layer_norm_td(td_output)

        return bu_output, td_output

    def reset(self):
        """Reset any stateful components of the layer."""
        pass


class BLT_VS_Readout(nn.Module):
    """Readout layer for the BLT model."""

    def __init__(self, layer_n, channel_sizes, kernel_sizes, strides, num_classes):
        super(BLT_VS_Readout, self).__init__()

        self.num_classes = num_classes
        in_channels = channel_sizes[layer_n - 1]
        out_channels = channel_sizes[layer_n]

        self.readout_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[layer_n],
            stride=strides[layer_n],
            padding=(kernel_sizes[layer_n] - 1) // 2,
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer_norm_td = nn.GroupNorm(num_groups=1, num_channels=out_channels)

    def forward(self, bu_input):
        """Forward pass for the Readout layer."""
        output_intermediate = self.readout_conv(bu_input)
        output_pooled = self.global_avg_pool(output_intermediate).view(
            output_intermediate.size(0), -1
        )
        output = output_pooled[
            :, : self.num_classes
        ]  # Only pass classes to softmax and loss
        td_output = self.layer_norm_td(F.relu(output_intermediate))

        return output, td_output


class NoOpModule(nn.Module):
    """A no-operation module that returns zero regardless of the input."""

    def __init__(self):
        super(NoOpModule, self).__init__()

    def forward(self, *args, **kwargs):
        """Forward pass that returns zero."""
        return 0


def concat_or_not(bu_activation, td_activation, dim=1):
    """Helper function to concatenate bottom-up and top-down activations."""
    if bu_activation is None and td_activation is None:
        return None

    if bu_activation is None:
        bu_activation = torch.zeros_like(td_activation)

    if td_activation is None:
        td_activation = torch.zeros_like(bu_activation)

    return torch.cat([bu_activation, td_activation], dim=dim)


class BL(BLT):
    def __init__(self, topdown_connections=False, **kwargs):
        super().__init__(topdown_connections=topdown_connections, **kwargs)


class BT(BLT):
    def __init__(self, lateral_connections=False, **kwargs):
        super().__init__(lateral_connections=lateral_connections, **kwargs)


class B(BLT):
    def __init__(self, topdown_connections=False, lateral_connections=False, **kwargs):
        super().__init__(
            topdown_connections=topdown_connections,
            lateral_connections=lateral_connections,
            **kwargs,
        )


class L(BLT):
    def __init__(self, topdown_connections=False, skip_connections=False, **kwargs):
        super().__init__(
            topdown_connections=topdown_connections,
            skip_connections=skip_connections,
            **kwargs,
        )


if __name__ == "__main__":
    # Test the BLT model
    input_dims = (5, 3, 224, 224)
    random_input = torch.randn(1, *input_dims)

    model = BLT(
        input_dims=input_dims,
        n_classes=1000,
        add_feats=100,
        lateral_connections=True,
        topdown_connections=True,
        skip_connections=True,
        bio_unroll=True,
        hook_type="None",
        readout_type="multi",
    )
    model.setup("fit")
    print("Residual Timesteps:", model.n_residual_timesteps)

    output = model(random_input)

    trainable_params = [
        f"{name} [{tuple(param.shape)}]"
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    print("Trainable Parameters:\n\t", "\n\t".join(trainable_params))
    print()
    print(f"Random Input ({tuple(random_input.shape)})")
    print(f"Model Output ({tuple(output[0].shape)})")  # First output from list
