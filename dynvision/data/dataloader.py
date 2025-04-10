"""Data loading

Usage:
    from dynvision.data.dataloader import get_data_loader
    
    loader = get_data_loader(
        dataset,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )
"""

import logging
import multiprocessing
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader

from .operations import (
    _adjust_data_dimensions,
    _repeat_over_time,
    _adjust_label_dimensions,
)
from dynvision.utils import alias_kwargs, filter_kwargs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure worker pools
MAX_WORKERS = min(16, multiprocessing.cpu_count() - 1)


class StandardDataLoader(DataLoader):
    """Base data loader with performance optimizations.

    Performance Tips:
    - Use pin_memory=True for faster GPU transfer
    - Set num_workers based on CPU cores
    - Enable prefetch_factor for background loading
    - Use memory_format=torch.channels_last for GPU optimization
    - Enable mixed precision with dtype=torch.float16
    """

    def __init__(
        self,
        *args,
        n_timesteps: int = 1,
        memory_format: torch.memory_format = torch.contiguous_format,
        dtype: torch.dtype = torch.float16,
        device: Optional[str] = None,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = True,
        **kwargs,
    ):
        # Configure performance features
        kwargs["pin_memory"] = kwargs.get("pin_memory", True)
        kwargs["num_workers"] = kwargs.get(
            "num_workers", min(4, multiprocessing.cpu_count() // 2)
        )
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers and kwargs["num_workers"]
        kwargs = filter_kwargs(DataLoader, kwargs)

        super().__init__(*args, **kwargs)
        self.n_timesteps = int(n_timesteps)
        self.memory_format = memory_format
        self.dtype = dtype
        self.device = device

        logger.info(
            f"Initialized {self.__class__.__name__} with {self.num_workers} workers "
            f"and prefetch factor {self.prefetch_factor}"
        )

    def __iter__(self):
        try:
            for sample in super(StandardDataLoader, self).__iter__():
                data, label_indices, *extra = sample

                label_indices = _adjust_label_dimensions(label_indices)

                if self.n_timesteps > 1:
                    data = _repeat_over_time(data, self.n_timesteps)
                    label_indices = _repeat_over_time(label_indices, self.n_timesteps)

                yield [data, label_indices, *extra]

        except Exception as e:
            logger.error(f"Error in data loading: {str(e)}")
            raise


class StimulusRepetitionDataLoader(StandardDataLoader):
    @alias_kwargs(repeat="n_timesteps")
    def __init__(self, *args, n_timesteps=20, **kwargs):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)


class StimulusDurationDataLoader(StandardDataLoader):
    """DataLoader for stimulus duration experiments.

    This loader presents a stimulus for a specified duration, with intro and outro periods.
    The stimulus is presented in the middle of the sequence, with void values before and after.
    """

    @alias_kwargs(
        tsteps="n_timesteps",
        stim="stimulus_duration",
        intro="intro_duration",
        voidid="non_label_index",
    )
    def __init__(
        self,
        *args,
        n_timesteps: int = 20,
        stimulus_duration: int = 5,
        intro_duration: int = 2,
        non_label_index: int = -1,
        void_value: float = 0,
        **kwargs,
    ):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)

        # Validate and store parameters
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
                "Not enough timesteps for stimulus presentation! "
                f"(n_timesteps={self.n_timesteps}, "
                f"stimulus_duration={self.stimulus_duration}, "
                f"intro_duration={self.intro_duration})"
            )

    def __iter__(self):
        try:
            for sample in DataLoader.__iter__(self):
                data, label_indices, *extra = sample

                # Apply performance optimizations
                data = _adjust_data_dimensions(data, self.memory_format)
                if isinstance(data, torch.Tensor):
                    data = data.to(dtype=self.dtype)
                    if self.device:
                        data = data.to(self.device)

                label_indices = _adjust_label_dimensions(label_indices)

                # Create void tensors with optimized memory layout
                non_label_indices = torch.full_like(
                    label_indices,
                    self.non_label_index,
                    memory_format=self.memory_format,
                )
                void = torch.full_like(
                    data, self.void_value, memory_format=self.memory_format
                )

                # Combine sequences efficiently
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

                yield [data, label_indices, *extra]

        except Exception as e:
            logger.error(f"Error in stimulus duration loading: {str(e)}")
            raise


class StimulusIntervalDataLoader(StandardDataLoader):
    """DataLoader for stimulus interval experiments with performance optimizations.

    This loader presents a stimulus twice with an interval between presentations.
    The sequence consists of:
    1. Intro period (void)
    2. First stimulus presentation
    3. Interval period (void)
    4. Second stimulus presentation
    5. Outro period (void)
    """

    @alias_kwargs(
        tsteps="n_timesteps",
        stim="stimulus_duration",
        intro="intro_duration",
        interval="interval_duration",
        voidid="non_label_index",
    )
    def __init__(
        self,
        *args,
        n_timesteps: int = 30,
        stimulus_duration: int = 2,
        intro_duration: int = 1,
        interval_duration: int = 2,
        non_label_index: int = -1,
        void_value: float = 0,
        **kwargs,
    ):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)

        # Validate and store parameters
        self.stimulus_duration = int(stimulus_duration)
        self.intro_duration = int(intro_duration)
        self.interval_duration = int(interval_duration)
        self.non_label_index = int(non_label_index)
        self.void_value = float(void_value)
        self.outro_duration = (
            self.n_timesteps
            - 2 * self.stimulus_duration
            - self.interval_duration
            - self.intro_duration
        )

        if self.outro_duration < 0:
            raise ValueError(
                f"{self.__class__}:\n"
                "Not enough timesteps for stimulus sequence! "
                f"(n_timesteps={self.n_timesteps}, "
                f"stimulus_duration={self.stimulus_duration} x2, "
                f"intro_duration={self.intro_duration}, "
                f"interval_duration={self.interval_duration})"
            )

    def __iter__(self):
        try:
            for sample in DataLoader.__iter__(self):
                data, label_indices, *extra = sample

                # Apply performance optimizations
                data = _adjust_data_dimensions(data, self.memory_format)
                if isinstance(data, torch.Tensor):
                    data = data.to(dtype=self.dtype)
                    if self.device:
                        data = data.to(self.device)

                label_indices = _adjust_label_dimensions(label_indices)

                # Create void tensors with optimized memory layout
                non_label_indices = torch.full_like(
                    label_indices,
                    self.non_label_index,
                    memory_format=self.memory_format,
                )
                void = torch.full_like(
                    data, self.void_value, memory_format=self.memory_format
                )

                # Combine sequences efficiently
                data = torch.cat(
                    (
                        _repeat_over_time(void, self.intro_duration),
                        _repeat_over_time(data, self.stimulus_duration),
                        _repeat_over_time(void, self.interval_duration),
                        _repeat_over_time(data, self.stimulus_duration),
                        _repeat_over_time(void, self.outro_duration),
                    ),
                    dim=1,
                )

                label_indices = torch.cat(
                    (
                        _repeat_over_time(non_label_indices, self.intro_duration),
                        _repeat_over_time(label_indices, self.stimulus_duration),
                        _repeat_over_time(non_label_indices, self.interval_duration),
                        _repeat_over_time(label_indices, self.stimulus_duration),
                        _repeat_over_time(non_label_indices, self.outro_duration),
                    ),
                    dim=1,
                )

                yield [data, label_indices, *extra]

        except Exception as e:
            logger.error(f"Error in stimulus interval loading: {str(e)}")
            raise


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


def get_data_loader(
    dataset: torch.utils.data.Dataset, dataloader=None, **kwargs
) -> torch.utils.data.DataLoader:

    dataloader = dataloader or StandardDataLoader

    if isinstance(dataloader, str):
        if "DataLoader" not in dataloader:
            dataloader += "DataLoader"
        dataloader = globals().get(dataloader)

    kwargs.setdefault("num_workers", max(1, min(4, multiprocessing.cpu_count() // 2)))
    kwargs.setdefault("persistent_workers", True)
    try:
        return dataloader(dataset, **kwargs)
    except Exception as e:
        logger.error(f"Error initializing DataLoader: {str(e)}")
        raise


def get_data_loader_class(dataloader=None) -> torch.utils.data.DataLoader:

    dataloader = dataloader or StandardDataLoader

    if isinstance(dataloader, str):
        if "DataLoader" not in dataloader:
            dataloader += "DataLoader"
        dataloader = globals().get(dataloader)

    return dataloader


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
