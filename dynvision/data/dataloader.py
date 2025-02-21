import random
from dataclasses import replace
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from ffcv.fields.decoders import (
    IntDecoder,
    NDArrayDecoder,
    RandomResizedCropRGBImageDecoder,
    SimpleRGBImageDecoder,
)
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import Convert, NormalizeImage, ToDevice, ToTensor, ToTorchImage
from torch.utils.data import DataLoader

from dynvision.data.transforms import (
    get_data_transform,
    get_target_transform,
)
from dynvision.utils.utils import alias_kwargs, filter_kwargs


def _adjust_data_dimensions(data):
    if len(data.shape) == 2:
        # assuming (dim_y, dim_x)
        data = data.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif len(data.shape) == 3:
        # assuming (batch_size, dim_y, dim_x)
        data = data.unsqueeze(1).unsqueeze(1)
    elif len(data.shape) == 4:
        # assuming (batch_size, n_channels, dim_y, dim_x)
        data = data.unsqueeze(1)
    elif len(data.shape) == 5:
        # assuming (batch_size, n_timesteps, n_channels, dim_y, dim_x)
        pass
    else:
        raise ValueError(f"Invalid data shape: {data.shape}")
    return data


def _adjust_label_dimensions(label_indices):
    if len(label_indices.shape) == 1:
        # assuming (batch_size)
        label_indices = label_indices.unsqueeze(1)
    elif len(label_indices.shape) == 2:
        # assuming (batch_size, n_timesteps)
        pass
    else:
        raise ValueError(f"Invalid label shape: {label_indices.shape}")
    return label_indices


def _repeat_over_time(tensor, n_repeat):
    # assuming shape (batch_size, n_timesteps=1, ...)
    batch_size, n_timesteps, *shape = tensor.shape

    if n_timesteps != 1:
        raise ValueError(
            f"Tensor already has an extended time dimension (n_timesteps={n_timesteps})!"
        )

    tensor = tensor.expand(batch_size, n_repeat, *shape)
    return tensor


class StandardDataLoader(DataLoader):
    def __init__(self, *args, n_timesteps=1, **kwargs):
        kwargs = filter_kwargs(super().__init__, kwargs)
        super().__init__(*args, **kwargs)
        self.n_timesteps = int(n_timesteps)

    def __iter__(self):
        for sample in super().__iter__():
            data, label_indices, *extra = sample

            data = _adjust_data_dimensions(data)
            label_indices = _adjust_label_dimensions(label_indices)

            if self.n_timesteps > 1:
                data = _repeat_over_time(data, self.n_timesteps)
                label_indices = _repeat_over_time(label_indices, self.n_timesteps)

            extended_sample = [data, label_indices, *extra]
            yield extended_sample


class StimulusRepetitionDataLoader(StandardDataLoader):
    @alias_kwargs(repeat="n_timesteps")
    def __init__(self, *args, n_timesteps=20, **kwargs):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)


class StimulusDurationDataLoader(StandardDataLoader):
    @alias_kwargs(
        tsteps="n_timesteps",
        stim="stimulus_duration",
        intro="intro_duration",
        voidid="non_label_index",
    )
    def __init__(
        self,
        *args,
        n_timesteps=20,
        stimulus_duration=5,
        intro_duration=2,
        non_label_index=-1,
        void_value=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_timesteps = int(n_timesteps)
        self.stimulus_duration = int(stimulus_duration)
        self.intro_duration = int(intro_duration)
        self.non_label_index = int(non_label_index)
        self.void_value = float(void_value)
        self.outro_duration = (
            self.n_timesteps - self.stimulus_duration - self.intro_duration
        )

        if self.outro_duration < 0:
            raise ValueError(
                f"{self.__class__}:\n"
                "Not enough time steps for the stimulus duration and intro duration! "
                f"(n_timesteps={self.n_timesteps}, stimulus_duration={self.stimulus_duration}, intro_duration={self.intro_duration})"
            )

    def __iter__(self):
        for sample in DataLoader.__iter__(self):
            data, label_indices, *extra = sample

            data = _adjust_data_dimensions(data)
            label_indices = _adjust_label_dimensions(label_indices)

            non_label_indices = torch.ones_like(label_indices) * self.non_label_index
            void = torch.ones_like(data) * self.void_value

            data = torch.cat(
                (
                    _repeat_over_time(void, self.intro_duration),
                    _repeat_over_time(data, self.stimulus_duration),
                    _repeat_over_time(void, self.outro_duration),
                ),
                dim=1,
            )

            label_indices = torch.cat(
                (
                    _repeat_over_time(non_label_indices, self.intro_duration),
                    _repeat_over_time(label_indices, self.stimulus_duration),
                    _repeat_over_time(non_label_indices, self.outro_duration),
                ),
                dim=1,
            )

            extended_sample = [data, label_indices, *extra]
            yield extended_sample


class StimulusIntedvmlDataLoader(StandardDataLoader):
    @alias_kwargs(
        tsteps="n_timesteps",
        stim="stimulus_duration",
        intro="intro_duration",
        intedvml="intedvml_duration",
        voidid="non_label_index",
    )
    def __init__(
        self,
        *args,
        n_timesteps=30,
        stimulus_duration=2,
        intro_duration=1,
        intedvml_duration=2,
        non_label_index=-1,
        void_value=0,
        **kwargs,
    ):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)
        self.stimulus_duration = int(stimulus_duration)
        self.intro_duration = int(intro_duration)
        self.intedvml_duration = int(intedvml_duration)
        self.non_label_index = int(non_label_index)
        self.void_value = float(void_value)
        self.outro_duration = (
            self.n_timesteps
            - 2 * self.stimulus_duration
            - self.intedvml_duration
            - self.intro_duration
        )

        if self.outro_duration < 0:
            raise ValueError(
                f"{self.__class__}:\n"
                "Not enough time steps for the stimulus duration and intro duration! "
                f"(n_timesteps={self.n_timesteps}, stimulus_duration={self.stimulus_duration} x2, intro_duration={self.intro_duration}, intedvml_duration={self.intedvml_duration})"
            )

    def __iter__(self):
        for sample in DataLoader.__iter__(self):
            data, label_indices, *extra = sample

            data = _adjust_data_dimensions(data)
            label_indices = _adjust_label_dimensions(label_indices)

            non_label_indices = torch.ones_like(label_indices) * self.non_label_index
            void = torch.ones_like(data) * self.void_value

            data = torch.cat(
                (
                    _repeat_over_time(void, self.intro_duration),
                    _repeat_over_time(data, self.stimulus_duration),
                    _repeat_over_time(void, self.intedvml_duration),
                    _repeat_over_time(data, self.stimulus_duration),
                    _repeat_over_time(void, self.outro_duration),
                ),
                dim=1,
            )

            label_indices = torch.cat(
                (
                    _repeat_over_time(non_label_indices, self.intro_duration),
                    _repeat_over_time(label_indices, self.stimulus_duration),
                    _repeat_over_time(non_label_indices, self.intedvml_duration),
                    _repeat_over_time(label_indices, self.stimulus_duration),
                    _repeat_over_time(non_label_indices, self.outro_duration),
                ),
                dim=1,
            )

            extended_sample = [data, label_indices, *extra]
            yield extended_sample


class StimulusContrastDataLoader(StandardDataLoader):
    @alias_kwargs(
        tsteps="n_timesteps",
        stim="stimulus_duration",
        intro="intro_duration",
        contrast="stimulus_contrast",
        voidid="non_label_index",
    )
    def __init__(
        self,
        *args,
        n_timesteps=15,
        stimulus_duration=10,
        intro_duration=2,
        stimulus_contrast=1.0,
        non_label_index=-1,
        void_value=0,
        **kwargs,
    ):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)
        self.stimulus_duration = int(stimulus_duration)
        self.intro_duration = int(intro_duration)
        self.stimulus_contrast = float(stimulus_contrast)
        self.non_label_index = int(non_label_index)
        self.void_value = float(void_value)
        self.outro_duration = (
            self.n_timesteps - self.stimulus_duration - self.intro_duration
        )

        if self.outro_duration < 0:
            raise ValueError(
                f"{self.__class__}:\n"
                "Not enough time steps for the stimulus duration and intro duration! "
                f"(n_timesteps={self.n_timesteps}, stimulus_duration={self.stimulus_duration}, intro_duration={self.intro_duration})"
            )

    def __iter__(self):
        for sample in DataLoader.__iter__(self):
            data, label_indices, *extra = sample

            data = _adjust_data_dimensions(data) * self.stimulus_contrast
            label_indices = _adjust_label_dimensions(label_indices)

            non_label_indices = torch.ones_like(label_indices) * self.non_label_index
            void = torch.ones_like(data) * self.void_value

            data = torch.cat(
                (
                    _repeat_over_time(void, self.intro_duration),
                    _repeat_over_time(data, self.stimulus_duration),
                    _repeat_over_time(void, self.outro_duration),
                ),
                dim=1,
            )

            label_indices = torch.cat(
                (
                    _repeat_over_time(non_label_indices, self.intro_duration),
                    _repeat_over_time(label_indices, self.stimulus_duration),
                    _repeat_over_time(non_label_indices, self.outro_duration),
                ),
                dim=1,
            )

            extended_sample = [data, label_indices, *extra]
            yield extended_sample

    pass


class NoiseAdaptionDataLoader(StandardDataLoader):
    @alias_kwargs(
        tsteps="n_timesteps",
        tadapt="t_adaption",
        tdelay="t_delay",
        ttest="t_test",
        mnoise="noise_mean",
        snoise="noise_std",
        voidid="non_label_index",
    )
    def __init__(
        self,
        *args,
        n_timesteps=30,
        t_adaption=5,
        t_delay=5,
        t_test=5,
        noise="none",  # none, same, different
        noise_mean=0.0,
        noise_std=0.1,
        non_label_index=-1,
        void_value=0,
        **kwargs,
    ):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)
        self.t_adaption = int(t_adaption)
        self.t_delay = int(t_delay)
        self.t_test = int(t_test)
        self.non_label_index = int(non_label_index)
        self.void_value = float(void_value)
        self.noise = noise
        self.noise_mean = float(noise_mean)
        self.noise_std = float(noise_std)
        self.outro_duration = (
            self.n_timesteps - self.t_adaption - self.t_delay - self.t_test
        )

        if self.outro_duration < 0:
            raise ValueError(
                f"{self.__class__}:\n"
                "Not enough time steps for the stimulus durations! "
                f"(n_timesteps={self.n_timesteps}, t_adaption={self.t_adaption}, t_delay={self.t_delay}, t_test={self.t_test})"
            )

    def __iter__(self):
        for sample in DataLoader.__iter__(self):
            data, label_indices, *extra = sample

            data = _adjust_data_dimensions(data)
            label_indices = _adjust_label_dimensions(label_indices)

            non_label_indices = torch.ones_like(label_indices) * self.non_label_index
            void = torch.ones_like(data) * self.void_value

            if self.noise == "none":
                adapter_noise = void
            else:
                adapter_noise = torch.normal(
                    mean=self.noise_mean,
                    std=self.noise_std,
                    size=data.shape,
                )
            if self.noise == "none":
                test_noise = torch.zeros_like(data)
            elif self.noise == "same":
                test_noise = adapter_noise
            else:
                test_noise = torch.normal(
                    mean=self.noise_mean,
                    std=self.noise_std,
                    size=data.shape,
                )
            data += test_noise

            data = torch.cat(
                (
                    _repeat_over_time(adapter_noise, self.t_adaption),
                    _repeat_over_time(void, self.t_delay),
                    _repeat_over_time(data, self.t_test),
                    _repeat_over_time(void, self.outro_duration),
                ),
                dim=1,
            )

            label_indices = torch.cat(
                (
                    _repeat_over_time(non_label_indices, self.t_adaption),
                    _repeat_over_time(non_label_indices, self.t_delay),
                    _repeat_over_time(label_indices, self.t_test),
                    _repeat_over_time(non_label_indices, self.outro_duration),
                ),
                dim=1,
            )

            extended_sample = [data, label_indices, *extra]
            yield extended_sample

    pass


class DoubleStimulusDataLoader(StandardDataLoader):
    @alias_kwargs(
        tsteps="n_timesteps",
        left="left_bottom_stim",
        right="right_top_stim",
    )
    def __init__(
        self,
        *args,
        n_timesteps=20,
        left_bottom_stim=9,
        right_top_stim=8,
        topdown=False,
        **kwargs,
    ):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)
        # TODO: not yet wired with the configs!!
        self.left_bottom_stim = left_bottom_stim
        self.right_top_stim = right_top_stim
        self.topdown = topdown

        # self.split_dataset = {}

        # for label in (self.left_bottom_stim, self.right_top_stim):
        #     indices = torch.where(
        #         torch.Tensor(self.dataset.targets) == torch.Tensor([int(label)])
        #     )[0]
        #     split_data = torch.utils.data.Subset(self.dataset, indices)
        #     self.split_dataset[label] = split_data

    def _superimpose_images(self, img1, img2):
        """
        Superimpose img1 in the upper right and img2 in the bottom left on a larger canvas.
        Args:
        - img1 (PIL.Image): The image to be placed in the upper right corner.
        - img2 (PIL.Image): The image to be placed in the bottom left corner.
        Returns:
        - PIL.Image: The combined image.
        """
        batch_size, w, h = img1.shape

        # Create a larger canvas (twice the size in both dimensions)
        canvas_size = (batch_size, 2 * w, 2 * h)
        # get background colour - in MNIST it will be the most frequently encountered number: -0.4242
        background = -0.4242
        # background = 0.0
        combined_img = torch.full(canvas_size, background)

        # Place tensor1 in the top-right corner
        combined_img[..., :w, h:] = img1

        # Place tensor2 in the bottom-left corner
        combined_img[..., w:, :h] = img2

        return combined_img

    def __iter__(self):
        # for sample_left, sample_right in itertools.product(
        #     DataLoader.__iter__(self.split_dataset[self.left_bottom_stim]),
        #     DataLoader.__iter__(self.split_dataset[self.right_top_stim]),
        # ):
        for samples in DataLoader.__iter__(self):
            data, label_indices, *extra = samples
            batch_size = len(data)

            left_idx = torch.where(label_indices == self.left_bottom_stim)[0]
            right_idx = torch.where(label_indices == self.right_top_stim)[0]

            left_idx = random.choices(left_idx.tolist(), k=batch_size)
            right_idx = random.choices(right_idx.tolist(), k=batch_size)

            data_left = data[left_idx]
            data_right = data[right_idx]
            label_indices_right = label_indices[right_idx]
            label_indices_left = label_indices[left_idx]

            combined_image = self._superimpose_images(
                data_right.squeeze(), data_left.squeeze()
            )
            combined_image = _adjust_data_dimensions(combined_image)

            # scale down images to original size
            # combined_image = combined_image.unfold(dimension=-2, size=2, step=2).mean(
            #     dim=-1
            # )
            # combined_image = combined_image.unfold(dimension=-1, size=2, step=2).mean(
            #     dim=-1
            # )

            # plt.imshow(combined_image[0, 0, 0, :, :])
            # breakpoint()

            # For now set the default image label to be the upper right one
            label_indices_right = _adjust_label_dimensions(label_indices_right)
            label_indices_left = _adjust_label_dimensions(label_indices_left)

            if self.n_timesteps > 1:
                combined_image = _repeat_over_time(combined_image, self.n_timesteps)
                # for now assume half/half labels and assume that the wave is topdown
                # label_indices = _repeat_over_time(label_indices, self.n_timesteps)

                label_indices = torch.cat(
                    (
                        _repeat_over_time(
                            label_indices_right, int(self.n_timesteps / 2)
                        ),
                        _repeat_over_time(
                            label_indices_left, int(self.n_timesteps / 2)
                        ),
                    ),
                    dim=1,
                )
                # if not self.topdown:
                #     label_indices = label_indices.flip(dims=(1, 0))

            # vertically flip the image:
            # combined_image = vflip(combined_image)

            extended_sample = [combined_image, label_indices, *extra]
            yield extended_sample

        # Maybe do dimensions adjustment after combining the images
        # data_left = _adjust_data_dimensions(data_left)
        # data_right = _adjust_data_dimensions(data_right)

        # label_indices_left = _adjust_label_dimensions(label_indices_left)
        # label_indices_right = _adjust_label_dimensions(label_indices_right)

        # now combine them both into one big image

        # extended_sample = [data, label_indices, *extra]
        # yield extended_sample


def get_data_loader(
    dataset: torch.utils.data.Dataset, dataloader=None, **kwargs
) -> torch.utils.data.DataLoader:

    dataloader = dataloader or StandardDataLoader

    if isinstance(dataloader, str):
        if "DataLoader" not in dataloader:
            dataloader += "DataLoader"
        dataloader = globals().get(dataloader)

    return dataloader(dataset, **kwargs)


def get_train_val_loaders(
    dataset, train_ratio, batch_size, num_workers=0, n_timesteps=1
):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = get_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        n_timesteps=n_timesteps,
    )
    val_loader = get_data_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        n_timesteps=n_timesteps,
    )
    return train_loader, val_loader


class ExtendDataTime(torch.nn.Module):
    def __init__(self, n_timesteps=1):
        super(ExtendDataTime, self).__init__()
        self.n_timesteps = n_timesteps

    def forward(self, x):
        x = _adjust_data_dimensions(x)
        x = _repeat_over_time(x, self.n_timesteps)
        return x


class ExtendLabelTime(torch.nn.Module):
    def __init__(self, n_timesteps=1):
        super(ExtendLabelTime, self).__init__()
        self.n_timesteps = n_timesteps

    def forward(self, x):
        x = _adjust_label_dimensions(x)
        x = _repeat_over_time(x, self.n_timesteps)
        return x


# class ExtendDataTime(Operation):
#     def __init__(self, n_timesteps=1):
#         super().__init__()
#         self.n_timesteps = n_timesteps

#     def generate_code(self) -> Callable:
#         n_timesteps = self.n_timesteps

#         def extend_data_time(x, x_):
#             x = _adjust_data_dimensions(x)
#             x_ = _repeat_over_time(x, n_timesteps)
#             return x_

#         return extend_data_time

#     def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
#         return previous_state, None


# class ExtendLabelTime(Operation):
#     def __init__(self, n_timesteps=1):
#         super().__init__()
#         self.n_timesteps = n_timesteps

#     def generate_code(self) -> Callable:
#         n_timesteps = self.n_timesteps

#         def extend_label_time(x, x_):
#             x_ = _adjust_label_dimensions(x)
#             x_ = _repeat_over_time(x_, n_timesteps)
#             return x_

#         return extend_label_time

#     def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
#         return previous_state, None


def get_ffcv_dataloader(
    path,
    batch_size,
    n_timesteps=1,
    num_workers=8,
    data_transform=None,
    target_transform=None,
    order=OrderOption.RANDOM,
    os_cache=True,
    encoding="image",
    resolution=224,
    drop_last=True,
    **kwargs,
):
    data_transform = get_data_transform(data_transform)
    target_transform = get_target_transform(target_transform)

    data_transform = data_transform if data_transform else []
    target_transform = target_transform if target_transform else []

    # Data decoding and augmentation via tensors
    if encoding == "tensor":
        image_pipeline = [NDArrayDecoder(), ToTensor()]
    elif encoding == "image":
        image_pipeline = [
            RandomResizedCropRGBImageDecoder(output_size=(resolution, resolution))
        ]
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    if data_transform:
        image_pipeline += data_transform
    if n_timesteps > 1:
        image_pipeline += [ExtendDataTime(n_timesteps)]

    label_pipeline = [IntDecoder(), ToTensor()]
    if target_transform:
        label_pipeline += target_transform
    if n_timesteps > 1:
        label_pipeline += [ExtendLabelTime(n_timesteps)]

    # Pipeline for each data field
    pipelines = {"image": image_pipeline, "label": label_pipeline}

    # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
    return Loader(
        path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=order,
        pipelines=pipelines,
        os_cache=os_cache,
        drop_last=drop_last,
        **kwargs,
    )
