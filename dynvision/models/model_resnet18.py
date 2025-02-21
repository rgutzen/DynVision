import torch
import torch.nn as nn
from torchvision import models as torch_models

from dynvision.modules import LightningBase


class ResNet18(LightningBase):
    def __init__(
        self,
        input_dims: tuple = (20, 3, 224, 224),
        n_classes: int = 1000,
        store_responses: bool = False,
        **kwargs,
    ) -> None:

        model_args = {
            k: v for k, v in locals().items() if k not in ["self", "kwargs"]
        } | kwargs
        super(ResNet18, self).__init__(**model_args)

        self._define_architecture()

        # make all classifier layers trainable
        # self.trainable_parameter_names = [p for p in list(self.state_dict().keys())]
        self.trainable_parameter_names = [
            p for p in list(self.state_dict().keys()) if "fc" in p
        ]

    def _define_architecture(self):
        self.model = torch_models.resnet18(
            weights=torch_models.ResNet18_Weights.DEFAULT
        )

        if self.n_classes != 1000:
            self.model.fc = nn.Linear(self.model.fc.in_features, self.n_classes)

    def get_classifier_dataframe(self, **kwargs):
        return super(ResNet18, self).get_classifier_dataframe(
            layer_name="fc", **kwargs
        )

    def reset(self):
        pass

    def _forward(
        self, x_0: torch.Tensor, t: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        return self.model(x_0), {}

    def _determine_residual_timesteps(self):
        return 0


if __name__ == "__main__":

    input_dims = (20, 1, 224, 224)

    random_input = torch.randn(1, *input_dims)

    model = ResNet18(input_dims=input_dims)

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
