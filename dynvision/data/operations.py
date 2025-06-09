import torch
import yaml
import logging
from dynvision.project_paths import project_paths

logger = logging.getLogger(__name__)


def _adjust_data_dimensions(
    data: torch.Tensor, memory_format: torch.memory_format = torch.contiguous_format
) -> torch.Tensor:
    """Adjust data dimensions to standard format.

    Converts input tensor to shape (batch_size, n_timesteps, n_channels, dim_y, dim_x)
    with optimized memory layout.

    Args:
        data: Input tensor with shape:
            - (dim_y, dim_x)
            - (batch_size, dim_y, dim_x)
            - (batch_size, n_channels, dim_y, dim_x)
            - (batch_size, n_timesteps, n_channels, dim_y, dim_x)
        memory_format: Memory format for tensor layout optimization

    Returns:
        Tensor with shape (batch_size, n_timesteps, n_channels, dim_y, dim_x)

    Raises:
        ValueError: If input shape is invalid
    """
    try:
        if data.dim() == 2:
            # (dim_y, dim_x) -> (1, 1, 1, dim_y, dim_x)
            data = data.view(1, 1, 1, data.size(0), data.size(1))
        elif data.dim() == 3:
            # (batch_size, dim_y, dim_x) -> (batch_size, 1, 1, dim_y, dim_x)
            data = data.view(data.size(0), 1, 1, data.size(1), data.size(2))
        elif data.dim() == 4:
            # (batch_size, n_channels, dim_y, dim_x) -> (batch_size, 1, n_channels, dim_y, dim_x)
            data = data.view(data.size(0), 1, data.size(1), data.size(2), data.size(3))
        elif data.dim() == 5:
            # Already correct shape
            pass
        else:
            raise ValueError(f"Invalid data shape: {data.shape}")

        # Optimize memory layout
        # return data.contiguous(memory_format=memory_format)  # requires dim=4
        return data

    except Exception as e:
        logger.error(f"Error adjusting data dimensions: {str(e)}")
        raise


def _adjust_label_dimensions(label_indices: torch.Tensor) -> torch.Tensor:
    """Adjust label dimensions to standard format.

    Converts input tensor to shape (batch_size, n_timesteps).

    Args:
        label_indices: Input tensor with shape:
            - (batch_size)
            - (batch_size, n_timesteps)

    Returns:
        Tensor with shape (batch_size, n_timesteps)

    Raises:
        ValueError: If input shape is invalid
    """
    try:
        if label_indices.dim() == 1:
            # (batch_size) -> (batch_size, 1)
            return label_indices.unsqueeze(1)
        elif label_indices.dim() == 2:
            # Already correct shape
            return label_indices
        elif label_indices.dim() == 3:
            # (batch_size, n_timesteps, 1) -> (batch_size, n_timesteps)
            return label_indices.squeeze(2)
        else:
            raise ValueError(f"Invalid label shape: {label_indices.shape}")

    except Exception as e:
        logger.error(f"Error adjusting label dimensions: {str(e)}")
        raise


def _repeat_over_time(tensor: torch.Tensor, n_repeat: int) -> torch.Tensor:
    """Repeat tensor along time dimension.

    Args:
        tensor: Input tensor with shape (batch_size, 1, ...)
        n_repeat: Number of times to repeat

    Returns:
        Tensor repeated along time dimension

    Raises:
        ValueError: If tensor already has time dimension
    """
    try:
        if tensor.size(1) != 1:
            raise ValueError(
                f"Tensor already has time dimension (n_timesteps={tensor.size(1)})"
            )

        # Optimize memory allocation
        shape = [1] * tensor.dim()
        shape[1] = n_repeat
        return tensor.expand(tensor.size(0), n_repeat, *tensor.shape[2:])

    except Exception as e:
        logger.error(f"Error repeating tensor: {str(e)}")
        raise


class IndexToLabel(torch.nn.Module):
    def __init__(self, data_name, data_group=None):
        super(IndexToLabel, self).__init__()

        with open(project_paths.scripts.configs / "config_data.yaml", "r") as file:
            data_groups = yaml.safe_load(file)["data_groups"]

        self.data_group_labels = data_groups[data_name][data_group]

    def _index_to_label(self, index: int) -> int:
        return int(self.data_group_labels[index])

    def forward(self, x):
        x = self._index_to_label(x)
        return x


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
