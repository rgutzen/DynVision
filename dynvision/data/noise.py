"""
Efficient image noise implementations for PyTorch DataLoader usage.

This module provides optimized noise functions designed for use in PyTorch DataLoaders.
Each function takes a noise_level parameter (0.0 to 1.0) and an optional seed for reproducibility.

Functions:
- salt_pepper_noise: Adds random salt (white) and pepper (black) pixels
- poisson_noise: Adds Poisson noise (photon noise simulation)
- uniform_noise: Adds uniformly distributed noise
- gaussian_blur: Applies Gaussian blur filter
- motion_blur: Applies directional motion blur

Usage in DataLoader:
    from noise_module import salt_pepper_noise
    
    class NoisyDataset(Dataset):
        def __getitem__(self, idx):
            image = self.images[idx]  # shape: (C, H, W)
            image = image.unsqueeze(0)  # shape: (1, C, H, W)
            noisy_image = salt_pepper_noise(image, noise_level=0.3, seed=42)
            return noisy_image.squeeze(0), self.labels[idx]
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple
import warnings

__all__ = [
    "salt_pepper_noise",
    "poisson_noise",
    "uniform_noise",
    "gaussian_blur",
    "motion_blur",
]

# Global kernel cache for blur operations to avoid recomputation
_KERNEL_CACHE = {}

# Global scaling factors to normalize visual difficulty across noise types
NOISE_SCALING_FACTORS = {
    "salt_pepper": 0.3,
    "poisson": 3.0,
    "uniform": 3.5,
    "gaussian": 1.0,
    "motion": 1.3,
}


def _set_seed(seed: Optional[int], device: torch.device) -> None:
    """Set random seed for reproducible noise generation."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)


def _get_gaussian_kernel(
    kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    Get or create cached Gaussian kernel for blur operations.

    Args:
        kernel_size: Size of the kernel (must be odd)
        sigma: Standard deviation of Gaussian
        device: Device to create kernel on
        dtype: Data type for kernel

    Returns:
        2D Gaussian kernel tensor
    """
    cache_key = (kernel_size, sigma, device, dtype)

    if cache_key not in _KERNEL_CACHE:
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size, dtype=dtype, device=device)
        x = x - (kernel_size - 1) / 2
        gaussian_1d = torch.exp(-(x**2) / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()

        # Create 2D kernel via outer product
        kernel_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        _KERNEL_CACHE[cache_key] = kernel_2d

    return _KERNEL_CACHE[cache_key]


def _get_motion_kernel(
    kernel_size: int, angle: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Get or create cached motion blur kernel."""
    cache_key = (kernel_size, angle, device, dtype)

    if cache_key not in _KERNEL_CACHE:
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)

        # Create motion blur kernel
        kernel = torch.zeros(kernel_size, kernel_size, dtype=dtype, device=device)
        center = kernel_size // 2
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Create line kernel with wider motion line
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = j - center, i - center
                # Distance along motion direction
                proj_dist = abs(x * cos_a + y * sin_a)
                perp_dist = abs(-x * sin_a + y * cos_a)

                # Create wider motion line - increased from kernel_size // 4
                max_proj_dist = max(1, kernel_size // 3)  # Increased line length
                if proj_dist <= max_proj_dist and perp_dist <= 1.0:  # Increased width
                    kernel[i, j] = 1.0

        # Normalize kernel
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel = kernel / kernel_sum

        _KERNEL_CACHE[cache_key] = kernel

    return _KERNEL_CACHE[cache_key]


def salt_pepper_noise(
    images: torch.Tensor,
    noise_level: float = 0.1,
    seed: Optional[int] = None,
    salt_vs_pepper: float = 0.5,
    cached_noise_state: Optional[dict] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    Apply salt and pepper noise to images with optional state caching.

    Args:
        images: Input tensor of shape (batch_size, channels, height, width)
        noise_level: Noise intensity from 0.0 to 1.0 (automatically scaled)
        seed: Random seed for reproducible results
        salt_vs_pepper: Ratio of salt to pepper noise
        cached_noise_state: Pre-computed noise masks for reuse

    Returns:
        If cached_noise_state is None: (noisy_images, noise_state) tuple
        If cached_noise_state is provided: noisy_images tensor
    """
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must be between 0.0 and 1.0")

    # Apply global scaling factor
    effective_noise_level = min(
        1.0, noise_level * NOISE_SCALING_FACTORS["salt_pepper"]
    )

    if effective_noise_level == 0.0:
        return images if cached_noise_state is not None else (images, {})

    noisy_images = images.clone()

    if cached_noise_state is not None:
        # Use cached masks
        noise_mask = cached_noise_state["noise_mask"]
        salt_mask = cached_noise_state["salt_mask"]
    else:
        # Generate new masks
        _set_seed(seed, images.device)
        noise_mask = (
            torch.rand(images.shape, device=images.device, dtype=images.dtype)
            < effective_noise_level
        )
        salt_mask = (
            torch.rand(images.shape, device=images.device, dtype=images.dtype)
            < salt_vs_pepper
        )

        # Return state for caching
        noise_state = {
            "noise_mask": noise_mask,
            "salt_mask": salt_mask,
            "function_type": "salt_pepper",
        }

    # Apply salt and pepper noise using masks
    salt_locations = noise_mask & salt_mask
    pepper_locations = noise_mask & (~salt_mask)

    noisy_images[salt_locations] = (
        images.max() + 2.0
    )  # Use max + offset instead of 1.0
    noisy_images[pepper_locations] = (
        images.min() - 2.0
    )  # Use min - offset instead of 0.0

    if cached_noise_state is not None:
        return noisy_images
    else:
        return noisy_images, noise_state


def poisson_noise(
    images: torch.Tensor,
    noise_level: float = 0.1,
    seed: Optional[int] = None,
    cached_noise_state: Optional[dict] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """Apply Poisson noise to normalized images (mean=0, std=1)."""
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must be between 0.0 and 1.0")

    # Apply global scaling factor - REMOVE the min(1.0, ...) limitation
    effective_noise_level = noise_level * NOISE_SCALING_FACTORS["poisson"]

    if effective_noise_level == 0.0:
        empty_state = {"function_type": "poisson"}
        return images if cached_noise_state is not None else (images, empty_state)

    if cached_noise_state is not None:
        # Use cached noise pattern
        noise_pattern = cached_noise_state["noise_pattern"]
        noisy_images = images + noise_pattern
    else:
        # Generate new noise pattern
        _set_seed(seed, images.device)

        # For normalized images, shift to [0, 4] range for Poisson calculation
        images_shifted = images + 2.0  # Shift from ~[-2,3] to ~[0,5]
        images_positive = torch.clamp(images_shifted, min=0.1)  # Ensure positive

        # Scale lambda parameter based on noise level - INCREASE base scale
        base_lambda_scale = 20.0  # Increased from 10.0
        lambda_param = images_positive * effective_noise_level * base_lambda_scale

        # Add minimum lambda - scale this too
        lambda_param = lambda_param + effective_noise_level * 5.0  # Increased from 2.0

        # Generate Poisson noise
        poisson_samples = torch.poisson(lambda_param)

        # Convert to noise pattern
        expected_values = lambda_param
        noise_in_shifted_space = poisson_samples - expected_values

        # Scale noise to appropriate magnitude - make this more aggressive
        noise_std_scale = 1.0 * effective_noise_level  # Increased from 0.5
        noise_pattern = (
            noise_in_shifted_space * noise_std_scale / torch.sqrt(lambda_param + 1e-8)
        )

        noisy_images = images + noise_pattern
        noise_state = {"noise_pattern": noise_pattern, "function_type": "poisson"}

    # Allow more distortion at high noise levels
    noisy_images = torch.clamp(noisy_images, -6.0, 8.0)  # Expanded from [-4.0, 6.0]

    if cached_noise_state is not None:
        return noisy_images
    else:
        return noisy_images, noise_state


def uniform_noise(
    images: torch.Tensor,
    noise_level: float = 0.1,
    seed: Optional[int] = None,
    cached_noise_state: Optional[dict] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """Apply uniform noise with optional state caching."""
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must be between 0.0 and 1.0")

    # Apply global scaling factor - REMOVE ceiling limitation
    effective_noise_level = noise_level * NOISE_SCALING_FACTORS["uniform"]

    if effective_noise_level == 0.0:
        return images if cached_noise_state is not None else (images, {})

    if cached_noise_state is not None:
        # Use cached noise pattern
        noise = cached_noise_state["noise_pattern"]
    else:
        # Generate new noise pattern
        _set_seed(seed, images.device)

        # Scale noise range based on effective noise level
        # Base range [-0.5, 0.5] gets scaled by effective_noise_level
        base_noise = (
            torch.rand(images.shape, device=images.device, dtype=images.dtype) - 0.5
        )

        # Scale noise magnitude - make it much more aggressive
        noise_magnitude = effective_noise_level * 2.0  # Increased scaling
        noise = base_noise * noise_magnitude

        noise_state = {"noise_pattern": noise, "function_type": "uniform"}

    # Apply noise - no clamping to allow distortion
    noisy_images = images + noise

    if cached_noise_state is not None:
        return noisy_images
    else:
        return noisy_images, noise_state


def gaussian_blur(
    images: torch.Tensor,
    noise_level: float = 0.1,
    seed: Optional[int] = None,
    max_kernel_size: int = 35,  # Increased from 25
    cached_noise_state: Optional[dict] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """Apply Gaussian blur with optional kernel caching."""
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must be between 0.0 and 1.0")

    # Apply global scaling factor - REMOVE ceiling limitation
    effective_noise_level = noise_level * NOISE_SCALING_FACTORS["gaussian"]

    if effective_noise_level == 0.0:
        return images if cached_noise_state is not None else (images, {})

    if cached_noise_state is not None:
        # Use cached kernel
        kernel = cached_noise_state["kernel"]
        padding = cached_noise_state["padding"]
    else:
        # Generate new kernel with more aggressive scaling
        _set_seed(seed, images.device)

        # More aggressive kernel size progression
        kernel_size = int(3 + effective_noise_level * (max_kernel_size - 3))
        if kernel_size % 2 == 0:
            kernel_size += 1

        # More aggressive sigma scaling
        sigma = 0.3 + effective_noise_level * (
            kernel_size / 2.0
        )  # Increased from kernel_size/3.0

        # Get Gaussian kernel
        kernel_2d = _get_gaussian_kernel(
            kernel_size, sigma, images.device, images.dtype
        )
        num_channels = images.size(1)
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1)
        padding = kernel_size // 2

        noise_state = {
            "kernel": kernel,
            "padding": padding,
            "function_type": "gaussian_blur",
        }

    # Apply blur
    blurred = F.conv2d(images, kernel, padding=padding, groups=images.size(1))

    if cached_noise_state is not None:
        return blurred
    else:
        return blurred, noise_state


def motion_blur(
    images: torch.Tensor,
    noise_level: float = 0.1,
    seed: Optional[int] = None,
    max_kernel_size: int = 35,  # Increased from 25
) -> torch.Tensor:
    """Apply motion blur to images."""
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must be between 0.0 and 1.0")

    # Apply global scaling factor - REMOVE ceiling limitation
    effective_noise_level = noise_level * NOISE_SCALING_FACTORS["motion"]

    if effective_noise_level == 0.0:
        return images

    _set_seed(seed, images.device)

    # More aggressive kernel size progression
    kernel_size = int(
        3 + effective_noise_level * (max_kernel_size - 3)
    )  # Start from 3
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Random angle for motion direction
    angle = torch.rand(1).item() * 360.0

    # Get motion blur kernel
    kernel_2d = _get_motion_kernel(kernel_size, angle, images.device, images.dtype)

    # Prepare kernel for convolution
    num_channels = images.size(1)
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1)

    # Apply grouped convolution
    padding = kernel_size // 2
    blurred = F.conv2d(images, kernel, padding=padding, groups=num_channels)

    return blurred


def clear_kernel_cache():
    """Clear the internal kernel cache to free memory."""
    global _KERNEL_CACHE
    _KERNEL_CACHE.clear()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def create_simple_noise_visualization():
        """Create a simple visualization showing noise progression with global scaling applied."""
        print("Creating noise visualization with global scaling...")

        # Create a test image with clear patterns
        test_image = torch.zeros(1, 3, 128, 128)

        # Create a gradient background
        for i in range(128):
            test_image[0, :, i, :] = i / 127.0

        # Add geometric shapes for better noise visibility
        test_image[0, :, 20:40, 20:40] = 1.0  # White square
        test_image[0, :, 60:80, 60:100] = 0.5  # Gray rectangle

        # Black circle
        center_x, center_y = 100, 30
        for i in range(128):
            for j in range(128):
                if (i - center_x) ** 2 + (j - center_y) ** 2 < 15**2:
                    test_image[0, :, i, j] = 0.0

        # NORMALIZE THE TEST IMAGE: Convert from [0,1] to mean=0, std=1
        test_image_flat = test_image.view(-1)
        mean_val = test_image_flat.mean()
        std_val = test_image_flat.std()
        test_image = (test_image - mean_val) / (
            std_val + 1e-8
        )  # Add epsilon to avoid division by zero

        print(
            f"Normalized test image - Mean: {test_image.mean().item():.4f}, Std: {test_image.std().item():.4f}"
        )
        print(
            f"Value range: [{test_image.min().item():.2f}, {test_image.max().item():.2f}]"
        )

        # Noise functions
        noise_functions = {
            "Original": None,
            "Salt & Pepper": salt_pepper_noise,
            "Poisson": poisson_noise,
            "Uniform": uniform_noise,
            "Gaussian Blur": gaussian_blur,
            "Motion Blur": motion_blur,
        }

        noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        # Create figure
        fig, axes = plt.subplots(
            len(noise_functions), len(noise_levels), figsize=(15, 12)
        )
        fig.suptitle(
            "Noise Types with Global Scaling Applied\n"
            f"Scaling: Salt&Pepper={NOISE_SCALING_FACTORS['salt_pepper']}, "
            f"Poisson={NOISE_SCALING_FACTORS['poisson']}, "
            f"Uniform={NOISE_SCALING_FACTORS['uniform']}, "
            f"Gaussian={NOISE_SCALING_FACTORS['gaussian']}, "
            f"Motion={NOISE_SCALING_FACTORS['motion']}",
            fontsize=14,
            fontweight="bold",
        )

        # Column headers
        for j, level in enumerate(noise_levels):
            axes[0, j].set_title(f"Level: {level}", fontsize=11, fontweight="bold")

        # Generate images
        for i, (noise_name, noise_func) in enumerate(noise_functions.items()):
            # Row label
            axes[i, 0].set_ylabel(noise_name, fontsize=11, fontweight="bold")

            for j, noise_level in enumerate(noise_levels):

                if noise_name == "Original" or noise_level == 0.0:
                    result_image = test_image
                else:
                    try:
                        # Apply noise (scaling is now handled internally)
                        if noise_func == motion_blur:
                            result_image = noise_func(
                                test_image, noise_level=noise_level, seed=42
                            )
                        else:
                            result = noise_func(
                                test_image, noise_level=noise_level, seed=42
                            )
                            if isinstance(result, tuple):
                                result_image, _ = result
                            else:
                                result_image = result
                    except Exception as e:
                        print(
                            f"Error applying {noise_name} at level {noise_level}: {e}"
                        )
                        result_image = test_image

                # Plot - need to handle normalized images for display
                img_np = result_image[0, 0].detach().cpu().numpy()

                # Convert normalized image back to [0,1] for display
                if noise_name == "Original" or img_np.min() < 0:
                    # Normalize to [0,1] for visualization
                    img_display = (img_np - img_np.min()) / (
                        img_np.max() - img_np.min() + 1e-8
                    )
                else:
                    img_display = np.clip(img_np, 0, 1)

                axes[i, j].imshow(img_display, cmap="gray", vmin=0, vmax=1)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

                # Add value range annotation
                min_val = result_image.min().item()
                max_val = result_image.max().item()
                axes[i, j].text(
                    3,
                    123,
                    f"[{min_val:.2f}, {max_val:.2f}]",
                    fontsize=7,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.8),
                )

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()

    def print_scaling_info():
        """Print information about the scaling factors."""
        print("Global Noise Scaling Factors:")
        print("=" * 40)
        for noise_type, factor in NOISE_SCALING_FACTORS.items():
            print(f"{noise_type.replace('_', ' ').title():15}: {factor:4.1f}x")
        print("=" * 40)
        print("These factors are automatically applied within each noise function")
        print("to normalize visual difficulty across different noise types.\n")

    # Run visualization
    print_scaling_info()
    create_simple_noise_visualization()

    print(f"\n{'='*50}")
    print("VISUALIZATION COMPLETED!")
    print(f"{'='*50}")
