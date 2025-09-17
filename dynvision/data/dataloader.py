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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import torch
import torch.jit
from torch.utils.data import DataLoader

from .operations import (
    _adjust_data_dimensions,
    _repeat_over_time,
    _adjust_label_dimensions,
)
from dynvision.utils import alias_kwargs, filter_kwargs

logger = logging.getLogger(__name__)


# JIT-compiled helper functions
@torch.jit.script
def _fill_tensor_period(
    output_data: torch.Tensor,
    output_labels: torch.Tensor,
    input_data: torch.Tensor,
    input_labels: torch.Tensor,
    start_idx: int,
    duration: int,
) -> int:
    """JIT-compiled function for filling any tensor period (intro, stimulus, outro)."""
    end_idx = start_idx + duration
    output_data[:, start_idx:end_idx] = input_data
    output_labels[:, start_idx:end_idx] = input_labels
    return end_idx


@torch.jit.script
def _expand_tensor_optimized(
    tensor: torch.Tensor, repeat_dim: int, n_repeats: int
) -> torch.Tensor:
    """JIT-compiled optimized tensor expansion."""
    shape = list(tensor.shape)
    shape[repeat_dim] = n_repeats
    return tensor.expand(shape)


class StandardDataLoader(DataLoader):
    """Base data loader with performance optimizations.

    Performance Tips:
    - Use pin_memory=True for faster GPU transfer
    - Set num_workers based on CPU cores
    - Enable prefetch_factor for background loading
    - Use memory_format=torch.channels_last for GPU optimization
    - Enable mixed precision with dtype=torch.float16
    """

    @alias_kwargs(data_timesteps="n_timesteps")
    def __init__(
        self,
        *args,
        n_timesteps: int = 1,
        memory_format: torch.memory_format = torch.contiguous_format,
        dtype: torch.dtype = torch.float16,
        device: Optional[str] = None,
        use_channels_last: bool = False,  # Default to False
        use_cuda_streams: bool = True,
        stream_priority: int = 0,
        max_cache_size: int = 1000,
        **kwargs,
    ):
        kwargs, _ = filter_kwargs(DataLoader, kwargs)

        super().__init__(*args, **kwargs)
        self.n_timesteps = int(n_timesteps)
        self.memory_format = memory_format
        self.dtype = dtype
        self.device = device
        self.use_channels_last = use_channels_last
        self.use_cuda_streams = use_cuda_streams
        self.max_cache_size = max_cache_size

        # Pre-compute optimal memory format based on data dimensions
        self._optimal_memory_format = (
            torch.channels_last if use_channels_last else torch.contiguous_format
        )

        # Initialize CUDA streams for overlapped computation
        if self.use_cuda_streams and torch.cuda.is_available():
            self._data_stream = torch.cuda.Stream(priority=stream_priority)
            self._copy_stream = torch.cuda.Stream(priority=stream_priority)
        else:
            self._data_stream = None
            self._copy_stream = None

        # Initialize caches with LRU eviction
        self._data_cache = {}
        self._label_cache = {}
        self._cache_access_order = OrderedDict()

    def _get_cached_tensors(self, data_shape, label_shape, device, dtype):
        """Get or create cached pre-allocated tensors with LRU eviction."""
        # Include memory format in cache key for proper separation
        cache_key = (
            data_shape,
            label_shape,
            device,
            dtype,
            self._optimal_memory_format,
        )

        # Update access order
        if cache_key in self._cache_access_order:
            self._cache_access_order.move_to_end(cache_key)
        else:
            self._cache_access_order[cache_key] = True

        if cache_key not in self._data_cache:
            # Evict old entries if cache is full
            if len(self._data_cache) >= self.max_cache_size:
                oldest_key = next(iter(self._cache_access_order))
                del self._data_cache[oldest_key]
                del self._label_cache[oldest_key]
                del self._cache_access_order[oldest_key]

            # Pre-allocate output tensors
            output_data_shape = (data_shape[0], self.n_timesteps, *data_shape[2:])
            output_label_shape = (label_shape[0], self.n_timesteps)

            # Create tensor with optimal memory layout (without pin_memory)
            self._data_cache[cache_key] = torch.empty(
                output_data_shape,
                dtype=dtype,
                device=device,
                memory_format=self._optimal_memory_format,
            )

            # For 4D+ tensors, ensure channels_last format if requested
            if len(output_data_shape) >= 4 and self.use_channels_last:
                self._data_cache[cache_key] = self._data_cache[cache_key].to(
                    memory_format=torch.channels_last
                )

            self._label_cache[cache_key] = torch.empty(
                output_label_shape, dtype=torch.long, device=device
            )

        return self._data_cache[cache_key], self._label_cache[cache_key]

    def _optimize_tensor_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to optimal memory layout for GPU processing."""
        if tensor.dim() >= 4 and self.use_channels_last:
            return tensor.to(memory_format=torch.channels_last)
        return tensor.contiguous()

    def _prefetch_to_device(self, tensor: torch.Tensor, device: str) -> torch.Tensor:
        """Asynchronously transfer tensor to device using copy stream."""
        if self._copy_stream is not None and device:
            with torch.cuda.stream(self._copy_stream):
                return tensor.to(device, non_blocking=True)
        return tensor.to(device) if device else tensor

    def _async_tensor_operations(
        self,
        data,
        label_indices,
        output_data,
        output_labels,
        non_input_value,
        non_label_index,
    ):
        """Perform tensor operations asynchronously using CUDA streams."""
        if self._data_stream is not None:
            with torch.cuda.stream(self._data_stream):
                # Optimize data layout asynchronously
                data = self._optimize_tensor_layout(data)
                label_indices = self._optimize_tensor_layout(label_indices)

                # Pre-fill output tensors with default values
                output_data.fill_(non_input_value)
                output_labels.fill_(non_label_index)

            # Synchronize before returning
            self._data_stream.synchronize()
        else:
            # Fallback for CPU or when streams are disabled
            data = self._optimize_tensor_layout(data)
            label_indices = self._optimize_tensor_layout(label_indices)
            output_data.fill_(non_input_value)
            output_labels.fill_(non_label_index)

        return data, label_indices

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
    """DataLoader for stimulus duration experiments with performance optimizations."""

    @alias_kwargs(
        tsteps="n_timesteps",
        data_timesteps="n_timesteps",
        stim="stimulus_duration",
        intro="intro_duration",
        voidid="non_label_index",
        voidinput="non_input_value",
    )
    def __init__(
        self,
        *args,
        n_timesteps: int = 20,
        stimulus_duration: int = 5,
        intro_duration: int = 0,
        non_label_index: int = -1,
        non_input_value: float = 0,
        **kwargs,
    ):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)

        # Validate and store parameters
        self.stimulus_duration = int(stimulus_duration)
        self.intro_duration = int(intro_duration)
        self.non_label_index = int(non_label_index)
        self.non_input_value = float(non_input_value)
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

        # Pre-compile JIT functions during initialization
        self._fill_period_jit = _fill_tensor_period
        self._expand_jit = _expand_tensor_optimized

    def __iter__(self):
        try:
            for sample in DataLoader.__iter__(self):
                data, label_indices, *extra = sample

                # Asynchronous device transfer
                if self.device:
                    data = self._prefetch_to_device(data, self.device)
                    label_indices = self._prefetch_to_device(
                        label_indices, self.device
                    )

                # Apply performance optimizations
                data = _adjust_data_dimensions(data, self._optimal_memory_format)
                data = self._optimize_tensor_layout(data)

                if isinstance(data, torch.Tensor):
                    data = data.to(dtype=self.dtype)

                label_indices = _adjust_label_dimensions(label_indices)

                # Get pre-allocated tensors
                output_data, output_labels = self._get_cached_tensors(
                    data.shape, label_indices.shape, data.device, data.dtype
                )

                # Async tensor operations and pre-fill with void values
                data, label_indices = self._async_tensor_operations(
                    data,
                    label_indices,
                    output_data,
                    output_labels,
                    self.non_input_value,
                    self.non_label_index,
                )

                # Use JIT-compiled functions for critical path
                time_idx = 0

                # Skip intro period since output is already pre-filled with void values
                time_idx += self.intro_duration

                # Stimulus period (JIT-compiled)
                if self.stimulus_duration > 0:
                    expanded_data = self._expand_jit(data, 1, self.stimulus_duration)
                    expanded_labels = self._expand_jit(
                        label_indices, 1, self.stimulus_duration
                    )

                    time_idx = self._fill_period_jit(
                        output_data,
                        output_labels,
                        expanded_data,
                        expanded_labels,
                        time_idx,
                        self.stimulus_duration,
                    )

                # Skip outro period since output is already pre-filled with void values

                # Return without cloning for zero-copy (safe since we use pre-allocated tensors)
                yield [output_data, output_labels, *extra]

        except Exception as e:
            logger.error(f"Error in stimulus duration loading: {str(e)}")
            raise


class StimulusIntervalDataLoader(StandardDataLoader):
    """DataLoader for stimulus interval experiments with performance optimizations."""

    @alias_kwargs(
        tsteps="n_timesteps",
        data_timesteps="n_timesteps",
        stim="stimulus_duration",
        intro="intro_duration",
        interval="interval_duration",
        voidid="non_label_index",
        voidinput="non_input_value",
    )
    def __init__(
        self,
        *args,
        n_timesteps: int = 30,
        stimulus_duration: int = 2,
        intro_duration: int = 0,
        interval_duration: int = 2,
        non_label_index: int = -1,
        non_input_value: float = 0,
        **kwargs,
    ):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)

        # Validate and store parameters
        self.stimulus_duration = int(stimulus_duration)
        self.intro_duration = int(intro_duration)
        self.interval_duration = int(interval_duration)
        self.non_label_index = int(non_label_index)
        self.non_input_value = float(non_input_value)
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

        # Pre-compile JIT functions
        self._fill_period_jit = _fill_tensor_period
        self._expand_jit = _expand_tensor_optimized

    def __iter__(self):
        try:
            for sample in DataLoader.__iter__(self):
                data, label_indices, *extra = sample

                # Asynchronous device transfer
                if self.device:
                    data = self._prefetch_to_device(data, self.device)
                    label_indices = self._prefetch_to_device(
                        label_indices, self.device
                    )

                # Apply performance optimizations
                data = _adjust_data_dimensions(data, self._optimal_memory_format)
                data = self._optimize_tensor_layout(data)

                if isinstance(data, torch.Tensor):
                    data = data.to(dtype=self.dtype)

                label_indices = _adjust_label_dimensions(label_indices)

                # Get pre-allocated tensors
                output_data, output_labels = self._get_cached_tensors(
                    data.shape, label_indices.shape, data.device, data.dtype
                )

                # Async tensor operations and pre-fill with void values
                data, label_indices = self._async_tensor_operations(
                    data,
                    label_indices,
                    output_data,
                    output_labels,
                    self.non_input_value,
                    self.non_label_index,
                )

                # Pre-expand data once for reuse
                expanded_data = self._expand_jit(data, 1, self.stimulus_duration)
                expanded_labels = self._expand_jit(
                    label_indices, 1, self.stimulus_duration
                )

                # Use JIT-compiled functions for critical path
                time_idx = self.intro_duration  # Skip intro (pre-filled)

                # First stimulus period
                if self.stimulus_duration > 0:
                    time_idx = self._fill_period_jit(
                        output_data,
                        output_labels,
                        expanded_data,
                        expanded_labels,
                        time_idx,
                        self.stimulus_duration,
                    )

                # Skip interval period (pre-filled)
                time_idx += self.interval_duration

                # Second stimulus period (reuse expanded tensors)
                if self.stimulus_duration > 0:
                    self._fill_period_jit(
                        output_data,
                        output_labels,
                        expanded_data,
                        expanded_labels,
                        time_idx,
                        self.stimulus_duration,
                    )

                yield [output_data, output_labels, *extra]

        except Exception as e:
            logger.error(f"Error in stimulus interval loading: {str(e)}")
            raise


class StimulusContrastDataLoader(StandardDataLoader):
    @alias_kwargs(
        tsteps="n_timesteps",
        data_timesteps="n_timesteps",
        stim="stimulus_duration",
        intro="intro_duration",
        contrast="stimulus_contrast",
        voidid="non_label_index",
        voidinput="non_input_value",
    )
    def __init__(
        self,
        *args,
        n_timesteps=15,
        stimulus_duration=10,
        intro_duration=2,
        stimulus_contrast=1.0,
        non_label_index=-1,
        non_input_value=0,
        **kwargs,
    ):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)
        self.stimulus_duration = int(stimulus_duration)
        self.intro_duration = int(intro_duration)
        self.stimulus_contrast = float(stimulus_contrast)
        self.non_label_index = int(non_label_index)
        self.non_input_value = float(non_input_value)
        self.outro_duration = (
            self.n_timesteps - self.stimulus_duration - self.intro_duration
        )

        if self.outro_duration < 0:
            raise ValueError(
                f"{self.__class__}:\n"
                "Not enough time steps for the stimulus duration and intro duration! "
                f"(n_timesteps={self.n_timesteps}, stimulus_duration={self.stimulus_duration}, intro_duration={self.intro_duration})"
            )

        # Pre-compile JIT functions
        self._fill_period_jit = _fill_tensor_period
        self._expand_jit = _expand_tensor_optimized

    def __iter__(self):
        for sample in DataLoader.__iter__(self):
            data, label_indices, *extra = sample

            # Apply contrast before other operations
            data = _adjust_data_dimensions(data) * self.stimulus_contrast
            data = self._optimize_tensor_layout(data)
            label_indices = _adjust_label_dimensions(label_indices)

            # Get pre-allocated tensors
            output_data, output_labels = self._get_cached_tensors(
                data.shape, label_indices.shape, data.device, data.dtype
            )

            # Async tensor operations and pre-fill with void values
            data, label_indices = self._async_tensor_operations(
                data,
                label_indices,
                output_data,
                output_labels,
                self.non_input_value,
                self.non_label_index,
            )

            # Use JIT-compiled functions
            time_idx = self.intro_duration  # Skip intro (pre-filled)

            # Stimulus period
            if self.stimulus_duration > 0:
                expanded_data = self._expand_jit(data, 1, self.stimulus_duration)
                expanded_labels = self._expand_jit(
                    label_indices, 1, self.stimulus_duration
                )

                self._fill_period_jit(
                    output_data,
                    output_labels,
                    expanded_data,
                    expanded_labels,
                    time_idx,
                    self.stimulus_duration,
                )

            yield [output_data, output_labels, *extra]


class StimulusNoiseDataLoader(StandardDataLoader):
    @alias_kwargs(
        tsteps="n_timesteps",
        data_timesteps="n_timesteps",
        stim="stimulus_duration",
        intro="intro_duration",
        noisetype="noise_type",
        noiselevel="noise_level",
        voidid="non_label_index",
        voidinput="non_input_value",
    )
    def __init__(
        self,
        *args,
        n_timesteps=15,
        stimulus_duration=10,
        intro_duration=2,
        noise_type="pixel",  # "pixel", "gaussian", "saltpepper"
        noise_level=0.1,
        non_label_index=-1,
        non_input_value=0,
        **kwargs,
    ):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)
        self.stimulus_duration = int(stimulus_duration)
        self.intro_duration = int(intro_duration)
        self.noise_type = str(noise_type).lower()
        self.noise_level = float(noise_level)
        self.non_label_index = int(non_label_index)
        self.non_input_value = float(non_input_value)
        self.outro_duration = (
            self.n_timesteps - self.stimulus_duration - self.intro_duration
        )

        if self.outro_duration < 0:
            raise ValueError(
                f"{self.__class__}:\n"
                "Not enough time steps for the stimulus duration and intro duration! "
                f"(n_timesteps={self.n_timesteps}, stimulus_duration={self.stimulus_duration}, intro_duration={self.intro_duration})"
            )

        # Pre-compile JIT functions
        self._fill_period_jit = _fill_tensor_period
        self._expand_jit = _expand_tensor_optimized

    def _apply_noise(self, data: torch.Tensor) -> torch.Tensor:
        """Apply noise to data tensor."""
        if self.noise_type == "gaussian":
            noise = torch.randn_like(data) * self.noise_level
            return data + noise
        elif self.noise_type == "pixel":
            noise = torch.rand_like(data) * self.noise_level
            return data + noise
        # Add other noise types as needed
        return data

    def __iter__(self):
        for sample in DataLoader.__iter__(self):
            data, label_indices, *extra = sample

            data = _adjust_data_dimensions(data)
            data = self._apply_noise(data)  # Apply noise
            data = self._optimize_tensor_layout(data)
            label_indices = _adjust_label_dimensions(label_indices)

            # Get pre-allocated tensors
            output_data, output_labels = self._get_cached_tensors(
                data.shape, label_indices.shape, data.device, data.dtype
            )

            # Async tensor operations and pre-fill with void values
            data, label_indices = self._async_tensor_operations(
                data,
                label_indices,
                output_data,
                output_labels,
                self.non_input_value,
                self.non_label_index,
            )

            # Use JIT-compiled functions
            time_idx = self.intro_duration  # Skip intro (pre-filled)

            # Stimulus period
            if self.stimulus_duration > 0:
                expanded_data = self._expand_jit(data, 1, self.stimulus_duration)
                expanded_labels = self._expand_jit(
                    label_indices, 1, self.stimulus_duration
                )

                self._fill_period_jit(
                    output_data,
                    output_labels,
                    expanded_data,
                    expanded_labels,
                    time_idx,
                    self.stimulus_duration,
                )

            yield [output_data, output_labels, *extra]


DATALOADER_CLASSES = {
    "StandardDataLoader": StandardDataLoader,
    "StimulusRepetitionDataLoader": StimulusRepetitionDataLoader,
    "StimulusDurationDataLoader": StimulusDurationDataLoader,
    "StimulusIntervalDataLoader": StimulusIntervalDataLoader,
    "StimulusContrastDataLoader": StimulusContrastDataLoader,
}


def get_data_loader(
    dataset: torch.utils.data.Dataset,
    dataloader: Optional[str] = None,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """
    Returns a DataLoader instance based on the specified dataloader name or class.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to load.
        dataloader (Optional[str]): The name of the DataLoader class or the class itself.
        data_timesteps (int): Number of timesteps for temporal data.
        **kwargs: Additional arguments for the DataLoader.

    Returns:
        torch.utils.data.DataLoader: An instance of the specified DataLoader.

    Raises:
        ValueError: If the specified DataLoader name is invalid.
        TypeError: If the provided dataloader is not a string or a valid DataLoader class.
        Exception: If there is an error during DataLoader initialization.
    """
    # Default to StandardDataLoader if no dataloader is specified
    dataloader = dataloader or StandardDataLoader

    # Validate and resolve the dataloader
    if isinstance(dataloader, str):
        if "DataLoader" not in dataloader:
            dataloader += "DataLoader"
        if dataloader in DATALOADER_CLASSES:
            dataloader_class = DATALOADER_CLASSES.get(dataloader)
        else:
            raise ValueError(f"Unknown DataLoader class: '{dataloader}'")
    elif issubclass(dataloader, DataLoader):
        dataloader_class = dataloader
    else:
        raise TypeError(
            f"Invalid dataloader type: {type(dataloader)}. "
            "Expected a string or a DataLoader subclass."
        )

    filtered_kwargs, _ = filter_kwargs(dataloader_class, kwargs)

    return dataloader_class(dataset, **filtered_kwargs)


def get_data_loader_class(dataloader=None) -> torch.utils.data.DataLoader:

    dataloader = dataloader or StandardDataLoader

    if isinstance(dataloader, str):
        if "DataLoader" not in dataloader:
            dataloader += "DataLoader"
        dataloader = globals().get(dataloader)

    return dataloader


def get_train_val_loaders(
    dataset: torch.utils.data.Dataset,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 0,
    data_timesteps: int = 1,
    dataloader: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders from a dataset.

    Args:
        dataset: The dataset to split
        train_ratio: Ratio of data to use for training
        batch_size: Batch size for both loaders
        num_workers: Number of worker processes
        data_timesteps: Number of timesteps for temporal data
        dataloader: Specific dataloader class to use
        **kwargs: Additional arguments for dataloaders

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Calculate split sizes
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Common loader arguments
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "data_timesteps": data_timesteps,
        "dataloader": dataloader,
        **kwargs,
    }
    common_kwargs.pop("shuffle", None)

    # Create loaders with appropriate shuffle settings
    train_loader = get_data_loader(train_dataset, shuffle=True, **common_kwargs)

    val_loader = get_data_loader(val_dataset, shuffle=False, **common_kwargs)

    return train_loader, val_loader
