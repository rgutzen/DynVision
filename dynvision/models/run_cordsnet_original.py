import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import v2
import numpy as np
from torch.utils.data import DataLoader
from dynvision.models.cordsnet_original import *
from PIL import Image


def process_image(pil_image):

    # this applies the preprocessing done on ImageNet images
    transform = transforms.Compose(
        [
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_tensor = transform(pil_image)
    return image_tensor.unsqueeze(0)  # [1, 3, 224, 224]


# set device to GPU
device = "cpu"

# alpha is defined as the discretization time step divided by the neuron time constant
# i.e. reciprocal of how many time steps are simulated for the duration of one neuron time constant
alpha = 0.2

# load dataset using torch, make sure the dataset is in the correct format
# ** THIS ONLY WORKS ON SQUARE IMAGES ** if your image is rectangular, pad with zeros first
# this returns a [1, 3, 224, 224] image
# feel free to stack a bunch of your own images
own_image = process_image(
    Image.open(
        "/home/rgutzen/01_PROJECTS/rhythmic_visual_attention/data/interim/imagenet/test_one/n01440764/n01440764_172.JPEG"
    ).convert("RGB")
)

# load model and pretrained weights (the publicly available model available on GitHub is trained on ImageNet)
model = cordsnet(dataset="imagenet", depth=8).to(device)
model.load_state_dict(
    torch.load(
        "/home/rgutzen/01_PROJECTS/rhythmic_visual_attention/models/CordsNet/CordsNet_pretrained.pt",
        map_location=torch.device("cpu"),
    )
)
model.eval()

with torch.no_grad():

    own_image = own_image.to(device)

    # there are 8 layers in this model, labeled 0 to 7
    # the last 2 layers (6 and 7) are the ones analyzed in the paper
    # feel free to change the "layers" argument to get the rest of the activities
    activity = model.record(own_image, device, alpha, layers=[6, 7])

# just to show how activity looks like, this gets the activity in layer 6
# the shape would be [200, number of images, C, H, W]
# for the sake of functioning like a CNN, the neurons are arranged in a "cuboid-like manner"
# so there are C x H x W neurons in total per layer
# the first 100 time steps are for the model to warm up
# the next 100 time steps are when the image is presented to the model
activity_in_layer_6 = activity[6]

# computing the output
out = model.out_avgpool(model.relu(activity[7]) + model.relu(activity[6]))
out = model.out_flatten(out)
out = model.out_fc(out)

# probability = torch.softmax(out, dim=1)
guess = torch.argmax(out, dim=1)

print("Predicted class index:", guess.unique())

import matplotlib.pyplot as plt

# plt.plot(probability[-1].detach().numpy())
# plt.xlabel("Class Index")
# plt.ylabel("Probability")
# plt.title("Predicted Class Probabilities")
# plt.show()

print(out.shape)
top_indices = torch.topk(out.mean(dim=0), 10).indices
plt.plot(out[:, top_indices].detach().numpy())
plt.xlabel("Time Steps")
plt.ylabel("Probability")
plt.title("Predicted Class Probabilities Over Time Steps")
plt.legend([f"Class {i}" for i in top_indices])
plt.show()
