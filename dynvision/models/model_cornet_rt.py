from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models as torch_models

from dynvision.modules import LightningBase


def get_model(model_letter, pretrained=False, map_location=None, **kwargs):
    model_letter = model_letter.upper()
    model_hash = globals()[f"HASH_{model_letter}"]
    model = globals()[f"CORnet_{model_letter}"](**kwargs)
    model = torch.nn.DataParallel(model)
    if pretrained:
        url = f"https://s3.amazonaws.com/cornet-models/cornet_{model_letter.lower()}-{model_hash}.pth"
        ckpt_data = torch.utils.model_zoo.load_url(url, map_location=map_location)
        model.load_state_dict(ckpt_data["state_dict"])
    return model


class CorBlockRT(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self._define_architecture()
        self.reset()

    def _define_architecture(self):
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2,
        )
        self.norm1 = nn.GroupNorm(32, self.out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(32, self.out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def reset(self):
        self.hidden_state = None

    def get_hidden_state(self, i=None):
        return self.hidden_state

    def set_hidden_state(self, hidden_state):
        self.hidden_state = hidden_state

    def forward(self, x_0=None):
        if x_0 is None:  # at t=0, there is no input yet except to V1
            x_1 = None
        else:
            x_1 = self.conv1(x_0)
            x_1 = self.norm1(x_1)
            x_1 = self.relu1(x_1)

        # Combine responses
        if x_1 is None and self.hidden_state is None:
            return None
        elif x_1 is None:
            x = self.hidden_state
        elif self.hidden_state is None:  # at t=0, state is initialized to 0
            x = x_1
        else:
            x = x_1 + self.hidden_state

        x_2 = self.conv2(x)
        x_2 = self.norm2(x_2)
        x_2 = self.relu2(x_2)

        return x_2


class CorNetRT(LightningBase):
    def __init__(
        self,
        input_dims: tuple = (10, 3, 224, 224),
        n_classes: int = 1000,
        **kwargs,
    ) -> None:
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        super(CorNetRT, self).__init__(**kwargs)

        self.model_letter = "rt"
        self.model_hash = "933c001c"

        self.load_pretrained_state_dict(check_mismatch_layer=["classifier.2"])
        self.trainable_parameter_names = [
            p for p in list(self.state_dict().keys()) if "classifier.2" in p
        ]

        self.reset()

    def download_pretrained_state_dict(self):
        url = f"https://s3.amazonaws.com/cornet-models/cornet_{self.model_letter.lower()}-{self.model_hash}.pth"
        ckpt_data = torch.utils.model_zoo.load_url(url, map_location=self.device)
        state_dict = ckpt_data["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        return state_dict

    def translate_pretrained_layer_names(self):
        translate_layer_names = {
            "conv1": "conv2",
            "norm1": "norm2",
            "conv_input": "conv1",
            "norm_input": "norm1",
            "decoder.linear": "classifier.2",
        }
        return translate_layer_names

    def _define_architecture(self):
        self.layer_names = ["V1", "V2", "V4", "IT"]
        # define operations order within layer
        self.layer_operations = [
            "layer",  # apply (recurrent) convolutional layer
            "addskip",  # add skip connection
            "nonlin",  # apply nonlinearity
            "supralin",  # apply supralinearity
            "record",  # record activations in responses dict
            "delay",  # set and get delayed activations for next layer
            "pool",  # apply pooling
            "norm",  # apply normalization
        ]
        # V1
        self.V1 = CorBlockRT(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=4,
        )
        # V2
        self.V2 = CorBlockRT(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
        )
        # V4
        self.V4 = CorBlockRT(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=2,
        )
        # IT
        self.IT = CorBlockRT(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=2,
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, self.n_classes),
        )

    def reset(self):
        for layer in [self.V1, self.V2, self.V4, self.IT]:
            layer.reset()

    def _forward(
        self, x_0: torch.Tensor, t: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        x_0 = self._expand_input_channels(x_0)

        # V1
        h_1 = self.V1.get_hidden_state()  # previous output
        x_1 = self.V1(x_0)
        self.V1.set_hidden_state(x_1)

        # V2
        h_2 = self.V2.get_hidden_state()
        x_2 = self.V2(h_1)
        self.V2.set_hidden_state(x_2)

        # V4
        h_3 = self.V4.get_hidden_state()
        x_3 = self.V4(h_2)
        self.V4.set_hidden_state(x_3)

        # IT
        h_4 = self.IT.get_hidden_state()
        x_4 = self.IT(h_3)
        self.IT.set_hidden_state(x_4)

        # Classifier
        if x_4 is None:
            x = None
        else:
            x = self.classifier(x_4)

        responses = {
            "V1": x_1,
            "V2": x_2,
            "V4": x_3,
            "IT": x_4,
            "classifier": x,
        }
        return x, responses


if __name__ == "__main__":

    input_dims = (20, 1, 224, 224)

    random_input = torch.randn(1, *input_dims)

    model = CorNetRT(input_dims=input_dims)

    output = model(random_input)

    trainable_params = [
        f"{name} [{tuple(param.shape)}]"
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    print("Trainable Parameters:\n\t", "\n\t".join(trainable_params))
    print()
    print(f"Random Input ({tuple(random_input.shape)})")
    print(f"Model Output ({tuple(output.shape)})")
