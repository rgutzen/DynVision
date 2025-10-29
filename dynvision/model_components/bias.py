"""
Bias modules for neural networks with different spatial resolutions.

This module provides standardized bias implementations at different granularities:
- SpatialBias: Per-unit bias (channels, height, width)
- FeatureBias: Per-channel bias (channels, 1, 1)
- ScalarBias: Single bias value per layer (1,)

These are useful for models like CordsNet that use spatially-varying biases.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "SpatialBias",
    "FeatureBias",
    "ScalarBias",
]


class SpatialBias(nn.Module):
    """
    Spatial bias with independent values for each spatial location.

    This creates a bias tensor of shape (channels, height, width) where
    each spatial location has its own bias value. This is the most flexible
    but also most parameter-intensive bias type.

    Used in models like CordsNet where each unit has its own bias.

    Args:
        channels: Number of feature channels
        height: Spatial height dimension
        width: Spatial width dimension
        init_value: Initial bias value (default: 0.0)
        requires_grad: Whether bias is trainable (default: True)

    Example:
        >>> bias = SpatialBias(channels=64, height=56, width=56)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> out = bias(x)
        >>> out.shape
        torch.Size([2, 64, 56, 56])
    """

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        init_value: float = 0.0,
        requires_grad: bool = True,
    ):
        super().__init__()

        self.channels = channels
        self.height = height
        self.width = width

        # Create bias parameter
        bias_tensor = torch.full(
            (channels, height, width),
            init_value,
            dtype=torch.float32,
        )
        self.bias = nn.Parameter(bias_tensor, requires_grad=requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add spatial bias to input tensor.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor with bias added
        """
        # Verify input shape matches bias shape
        if x.shape[1:] != self.bias.shape:
            raise ValueError(
                f"Input shape {x.shape[1:]} does not match bias shape {self.bias.shape}"
            )

        # Add bias (broadcast over batch dimension)
        return x + self.bias

    def __repr__(self):
        return (
            f"SpatialBias(channels={self.channels}, "
            f"height={self.height}, width={self.width}, "
            f"requires_grad={self.bias.requires_grad})"
        )


class FeatureBias(nn.Module):
    """
    Feature bias with one value per channel.

    This creates a bias tensor of shape (channels, 1, 1) that is broadcast
    across spatial dimensions. This is the standard bias type in CNNs.

    Args:
        channels: Number of feature channels
        init_value: Initial bias value (default: 0.0)
        requires_grad: Whether bias is trainable (default: True)

    Example:
        >>> bias = FeatureBias(channels=64)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> out = bias(x)
        >>> out.shape
        torch.Size([2, 64, 56, 56])
    """

    def __init__(
        self,
        channels: int,
        init_value: float = 0.0,
        requires_grad: bool = True,
    ):
        super().__init__()

        self.channels = channels

        # Create bias parameter with shape (channels, 1, 1) for broadcasting
        bias_tensor = torch.full(
            (channels, 1, 1),
            init_value,
            dtype=torch.float32,
        )
        self.bias = nn.Parameter(bias_tensor, requires_grad=requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add feature bias to input tensor.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor with bias added (broadcast across spatial dims)
        """
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Input has {x.shape[1]} channels but bias expects {self.channels}"
            )

        return x + self.bias

    def __repr__(self):
        return (
            f"FeatureBias(channels={self.channels}, "
            f"requires_grad={self.bias.requires_grad})"
        )


class ScalarBias(nn.Module):
    """
    Scalar bias with a single value for the entire layer.

    This creates a bias scalar that is broadcast across all dimensions.
    Rarely used but included for completeness.

    Args:
        init_value: Initial bias value (default: 0.0)
        requires_grad: Whether bias is trainable (default: True)

    Example:
        >>> bias = ScalarBias()
        >>> x = torch.randn(2, 64, 56, 56)
        >>> out = bias(x)
        >>> out.shape
        torch.Size([2, 64, 56, 56])
    """

    def __init__(
        self,
        init_value: float = 0.0,
        requires_grad: bool = True,
    ):
        super().__init__()

        # Create scalar bias parameter
        bias_tensor = torch.tensor(init_value, dtype=torch.float32)
        self.bias = nn.Parameter(bias_tensor, requires_grad=requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add scalar bias to input tensor.

        Args:
            x: Input tensor of any shape

        Returns:
            Output tensor with bias added (broadcast across all dims)
        """
        return x + self.bias

    def __repr__(self):
        return f"ScalarBias(requires_grad={self.bias.requires_grad})"


def create_bias(
    bias_type: str,
    channels: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    init_value: float = 0.0,
    requires_grad: bool = True,
) -> nn.Module:
    """
    Factory function to create bias modules.

    Args:
        bias_type: Type of bias ("spatial", "feature", "scalar", or "none")
        channels: Number of feature channels (required for spatial/feature)
        height: Spatial height (required for spatial)
        width: Spatial width (required for spatial)
        init_value: Initial bias value
        requires_grad: Whether bias is trainable

    Returns:
        Bias module of requested type, or Identity if bias_type="none"

    Example:
        >>> bias = create_bias("spatial", channels=64, height=56, width=56)
        >>> bias = create_bias("feature", channels=64)
        >>> bias = create_bias("scalar")
        >>> bias = create_bias("none")  # Returns Identity
    """
    bias_type = bias_type.lower()

    if bias_type in ["none", "null", None]:
        return nn.Identity()

    elif bias_type == "spatial":
        if channels is None or height is None or width is None:
            raise ValueError(
                "spatial bias requires channels, height, and width parameters"
            )
        return SpatialBias(
            channels=channels,
            height=height,
            width=width,
            init_value=init_value,
            requires_grad=requires_grad,
        )

    elif bias_type == "feature":
        if channels is None:
            raise ValueError("feature bias requires channels parameter")
        return FeatureBias(
            channels=channels,
            init_value=init_value,
            requires_grad=requires_grad,
        )

    elif bias_type == "scalar":
        return ScalarBias(
            init_value=init_value,
            requires_grad=requires_grad,
        )

    else:
        raise ValueError(
            f"Unknown bias type: {bias_type}. "
            f"Choose from: 'spatial', 'feature', 'scalar', 'none'"
        )


if __name__ == "__main__":
    # Test different bias types
    batch_size = 2
    channels = 64
    height, width = 56, 56

    x = torch.randn(batch_size, channels, height, width)

    print(f"Input shape: {x.shape}")
    print()

    # Test SpatialBias
    spatial_bias = SpatialBias(channels, height, width)
    out = spatial_bias(x)
    print(f"SpatialBias: {spatial_bias}")
    print(f"  Parameters: {sum(p.numel() for p in spatial_bias.parameters()):,}")
    print(f"  Output shape: {out.shape}")
    print()

    # Test FeatureBias
    feature_bias = FeatureBias(channels)
    out = feature_bias(x)
    print(f"FeatureBias: {feature_bias}")
    print(f"  Parameters: {sum(p.numel() for p in feature_bias.parameters()):,}")
    print(f"  Output shape: {out.shape}")
    print()

    # Test ScalarBias
    scalar_bias = ScalarBias()
    out = scalar_bias(x)
    print(f"ScalarBias: {scalar_bias}")
    print(f"  Parameters: {sum(p.numel() for p in scalar_bias.parameters()):,}")
    print(f"  Output shape: {out.shape}")
    print()

    # Test factory function
    bias = create_bias("spatial", channels, height, width)
    print(f"Factory (spatial): {bias}")

    bias = create_bias("feature", channels=channels)
    print(f"Factory (feature): {bias}")

    bias = create_bias("none")
    print(f"Factory (none): {bias}")

    print("\n✓ All bias types tested successfully!")
