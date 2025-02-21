import torch
import torch.nn as nn
from torchvision import models as torch_models

from dynvision.modules import (
    LightningBase,
    InputAdaption,
    RecurrentConnectedConv2d,
    SpatialWave,
)


class AlexNet(LightningBase):
    def __init__(
        self,
        input_dims: tuple = (20, 3, 224, 224),
        n_classes: int = 1000,
        dropout: float = 0.5,
        fixed_self_weight: float = None,
        store_responses: bool = False,
        recurrence_type: str = "none",
        **kwargs,
    ) -> None:
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        super(AlexNet, self).__init__(**kwargs)

        self.load_pretrained_state_dict(check_mismatch_layer=["classifier.6"])
        # make all classifier layers trainable
        self.trainable_parameter_names = [
            p for p in list(self.state_dict().keys()) if "classifier.6" in p
        ]

    def download_pretrained_state_dict(self):
        pretrained_weights = torch_models.AlexNet_Weights.DEFAULT
        pretrained_weights = torch_models.AlexNet_Weights.verify(pretrained_weights)
        return pretrained_weights.get_state_dict()

    def translate_pretrained_layer_names(self):
        translate_layer_names = {
            "features.0": "layer1.conv",
            "features.3": "layer2.conv",
            "features.6": "layer3.conv",
            "features.8": "layer4.conv",
            "features.10": "layer5.conv",
        }
        return translate_layer_names

    def _define_architecture(self):
        # Input adaption
        # self.input_adaption = InputAdaption()
        layer_dims = (self.dim_y, self.dim_x)

        # Layer 1
        self.layer1 = RecurrentConnectedConv2d(
            in_channels=3,  # self.in_channels,
            out_channels=64,
            kernel_size=11,
            stride=4,
            padding=2,
            fixed_self_weight=self.fixed_self_weight,
            recurrence_type=self.recurrence_type,
        )
        self.relu1 = nn.ReLU(inplace=False)
        # layer_dims = self._calculate_layer_output_shape(layer_dims, 11, 4, 2)
        # self.norm1 = nn.LayerNorm([64, *layer_dims])
        # Pool 1
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # layer_dims = self._calculate_layer_output_shape(layer_dims, 3, 2)

        # Layer 2
        self.layer2 = RecurrentConnectedConv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=5,
            padding=2,
            fixed_self_weight=self.fixed_self_weight,
            recurrence_type=self.recurrence_type,
        )
        self.relu2 = nn.ReLU(inplace=False)
        # layer_dims = self._calculate_layer_output_shape(layer_dims, 5, 1, 2)
        # Layer Norm 2
        # self.norm2 = nn.LayerNorm([192, *layer_dims])
        # Pool 2
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # layer_dims = self._calculate_layer_output_shape(layer_dims, 3, 2)

        # Layer 3
        self.layer3 = RecurrentConnectedConv2d(
            in_channels=192,
            out_channels=384,
            kernel_size=3,
            padding=1,
            fixed_self_weight=self.fixed_self_weight,
            recurrence_type=self.recurrence_type,
        )
        self.relu3 = nn.ReLU(inplace=False)
        # layer_dims = self._calculate_layer_output_shape(layer_dims, 3, 1, 1)
        # Layer Norm 3
        # self.norm3 = nn.LayerNorm([384, *layer_dims])

        # Layer 4
        self.layer4 = RecurrentConnectedConv2d(
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            padding=1,
            fixed_self_weight=self.fixed_self_weight,
            recurrence_type=self.recurrence_type,
        )
        self.relu4 = nn.ReLU(inplace=False)
        # layer_dims = self._calculate_layer_output_shape(layer_dims, 3, 1, 1)
        # Layer Norm 4
        # self.norm4 = nn.LayerNorm([256, *layer_dims])

        # Layer 5
        self.layer5 = RecurrentConnectedConv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1,
            fixed_self_weight=self.fixed_self_weight,
            recurrence_type=self.recurrence_type,
        )
        self.relu5 = nn.ReLU(inplace=False)
        # layer_dims = self._calculate_layer_output_shape(layer_dims, 3, 1, 1)
        # Layer Norm 5
        # self.norm5 = nn.LayerNorm([256, *layer_dims])
        # Pool 5
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        # layer_dims = self._calculate_layer_output_shape(layer_dims, 3, 2)

        # AvgPool
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.n_classes),
        )

    def reset(self):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]:
            layer.reset()

    def _forward(
        self, x_0: torch.Tensor, t: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        x_0 = self._expand_input_channels(x_0)

        # layer 1
        x_1_0 = self.layer1(x_0)
        # x_1_1 = self.norm1(x_1_0)
        x_1_1 = self.relu1(x_1_0)
        x_1_2 = self.pool1(x_1_1)

        # layer 2
        x_2_0 = self.layer2(x_1_2)
        # x_2_1 = self.norm2(x_2_0)
        x_2_1 = self.relu2(x_2_0)
        x_2_2 = self.pool2(x_2_1)

        # layer 3
        x_3_0 = self.layer3(x_2_1)  #
        x_3_1 = self.relu3(x_3_0)
        # x_3_1 = self.norm3(x_3_0)

        # layer 4
        x_4_0 = self.layer4(x_3_1)
        x_4_1 = self.relu4(x_4_0)
        # x_4_1 = self.norm4(x_4_0)

        # layer 5
        x_5_0 = self.layer5(x_4_1)
        x_5_1 = self.relu5(x_5_0)
        # x_5_1 = self.norm5(x_5_0)
        # x_5_2 = self.pool5(x_5_1)

        # avgpool
        x = self.avgpool(x_5_1)  #

        # classifier
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        responses = {
            "layer1": x_1_0,
            "layer2": x_2_0,
            "layer3": x_3_0,
            "layer4": x_4_0,
            "layer5": x_5_0,
            "classifier": x,
        }
        return x, responses


if __name__ == "__main__":

    input_dims = (20, 1, 224, 224)

    random_input = torch.randn(1, *input_dims)

    model = AlexNet(input_dims=input_dims)

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
