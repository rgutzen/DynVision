from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models as torch_models

from dynvision.modules import LightningBase


class CorBlockZ(nn.Module):

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

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x_0=None):
        if x_0 is None:
            out = None
        else:
            x_1 = self.conv(x_0)
            x_2 = self.nonlin(x_1)
            x_3 = self.pool(x_2)
            out = x_3

        return out


class CorNetZ(LightningBase):
    def __init__(
        self,
        input_dims: tuple = (10, 3, 224, 224),
        n_classes: int = 1000,
        **kwargs,
    ) -> None:
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        super(CorNetZ, self).__init__(**kwargs)

        self.model_letter = "z"
        self.model_hash = "5c427c9c"

        self.load_pretrained_state_dict(check_mismatch_layer=["classifier.2"])

    def translate_pretrained_layer_names(self):
        translate_layer_names = {
            "decoder.linear": "classifier.2",
        }
        return translate_layer_names

    def download_pretrained_state_dict(self):
        url = f"https://s3.amazonaws.com/cornet-models/cornet_{self.model_letter.lower()}-{self.model_hash}.pth"
        ckpt_data = torch.utils.model_zoo.load_url(url, map_location=self.device)
        state_dict = ckpt_data["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        return state_dict

    def _define_architecture(self):
        # V1
        self.V1 = CorBlockZ(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
        )
        # V2
        self.V2 = CorBlockZ(
            in_channels=64,
            out_channels=128,
        )
        # V4
        self.V4 = CorBlockZ(
            in_channels=128,
            out_channels=256,
        )
        # IT
        self.IT = CorBlockZ(
            in_channels=256,
            out_channels=512,
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, self.n_classes),
        )

    def reset(self):
        pass

    def _forward(
        self, x_0: torch.Tensor, t: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        x_0 = self._expand_input_channels(x_0)

        # V1
        x_1 = self.V1(x_0)

        # V2
        x_2 = self.V2(x_1)

        # V4
        x_3 = self.V4(x_2)

        # IT
        x_4 = self.IT(x_3)

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

    model = CorNetZ(input_dims=input_dims)

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
