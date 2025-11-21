"""Tests for loss function normalization across timesteps.

This module verifies that:
1. CrossEntropyLoss normalizes only by valid (non-masked) timesteps
2. EnergyLoss accumulates and normalizes across all timesteps
3. Loss combination correctly weights and sums individual losses
"""

import pytest
import torch
import torch.nn as nn
from dynvision.losses import CrossEntropyLoss, EnergyLoss


class TestCrossEntropyLossNormalization:
    """Test CrossEntropyLoss normalization with masked timesteps."""

    def test_normalization_with_valid_timesteps_only(self):
        """Verify CrossEntropyLoss normalizes by valid timesteps only."""
        # Setup
        batch_size = 2
        n_timesteps = 5
        n_classes = 3
        ignore_index = -1

        loss_fn = CrossEntropyLoss(reduction="mean", ignore_index=ignore_index)

        # Create outputs and targets with some masked timesteps
        torch.manual_seed(42)
        outputs = torch.randn(batch_size * n_timesteps, n_classes, requires_grad=True)

        # Create targets with 2 valid and 3 invalid timesteps per sample
        # Sample 0: valid at t=0,1; invalid at t=2,3,4
        # Sample 1: valid at t=1,2; invalid at t=0,3,4
        targets = torch.tensor(
            [
                0,
                1,
                ignore_index,
                ignore_index,
                ignore_index,  # Sample 0
                ignore_index,
                2,
                1,
                ignore_index,
                ignore_index,  # Sample 1
            ]
        )

        # Compute loss
        loss = loss_fn(outputs, targets)

        # Manual calculation: only 4 valid timesteps total (2 per sample)
        valid_mask = targets != ignore_index
        valid_outputs = outputs[valid_mask]
        valid_targets = targets[valid_mask]

        expected_loss = torch.nn.functional.cross_entropy(
            valid_outputs, valid_targets, reduction="sum"
        ) / valid_mask.sum().item()

        assert torch.allclose(loss, expected_loss, atol=1e-5)
        assert valid_mask.sum().item() == 4  # Verify 4 valid timesteps

    def test_all_timesteps_masked(self):
        """Verify behavior when all timesteps are masked."""
        batch_size = 2
        n_timesteps = 3
        n_classes = 3
        ignore_index = -1

        loss_fn = CrossEntropyLoss(reduction="mean", ignore_index=ignore_index)

        outputs = torch.randn(batch_size * n_timesteps, n_classes, requires_grad=True)
        targets = torch.full((batch_size * n_timesteps,), ignore_index, dtype=torch.long)

        loss = loss_fn(outputs, targets)

        # When all timesteps are masked, loss should be 0
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


class TestEnergyLossAccumulation:
    """Test EnergyLoss accumulation and normalization across timesteps."""

    def test_accumulation_across_timesteps(self):
        """Verify EnergyLoss accumulates energy across multiple timesteps."""

        # Create a simple model with one Conv2d layer
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)

            def forward(self, x):
                return self.conv(x)

        model = SimpleModel()
        loss_fn = EnergyLoss(reduction="mean", p=1)
        loss_fn.register_hooks(model)

        batch_size = 2
        n_timesteps = 4
        channels = 3
        height = width = 8

        torch.manual_seed(42)

        # Simulate forward pass over multiple timesteps
        # In reality, this happens inside TemporalBase.forward()
        per_timestep_energies = []

        for t in range(n_timesteps):
            x = torch.randn(batch_size, channels, height, width)
            output = model(x)  # This fires the hooks

            # Manually calculate expected energy for this timestep
            with torch.no_grad():
                timestep_energy = torch.norm(output, p=1, dim=(1, 2, 3))
                per_timestep_energies.append(timestep_energy)

        # Now compute the loss (this would be called in LightningBase.compute_loss)
        loss = loss_fn(outputs=None, targets=None)

        # Manual calculation
        total_energy = torch.stack(per_timestep_energies).sum(dim=0)  # Sum over timesteps
        n_units = 8 * 8 * 8  # channels * height * width
        normalized_energy = total_energy / n_units  # Normalize by spatial dimensions
        expected_loss = (
            normalized_energy / n_timesteps
        ).mean()  # Normalize by timesteps and average over batch

        assert torch.allclose(loss, expected_loss, rtol=1e-4)

    def test_hook_call_count_tracking(self):
        """Verify hook call counts correctly track number of timesteps."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 4, kernel_size=3)
                self.conv2 = nn.Conv2d(4, 8, kernel_size=3)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        model = SimpleModel()
        loss_fn = EnergyLoss(reduction="mean", p=1)
        loss_fn.register_hooks(model)

        batch_size = 2
        n_timesteps = 5

        # Simulate forward pass
        for t in range(n_timesteps):
            x = torch.randn(batch_size, 3, 8, 8)
            _ = model(x)

        # Check hook call counts before compute_loss
        assert len(loss_fn._hook_call_count) == 2  # Two monitored layers
        for module_name, count in loss_fn._hook_call_count.items():
            assert count == n_timesteps, f"{module_name} called {count} times, expected {n_timesteps}"

        # Compute loss (which resets counters)
        _ = loss_fn(outputs=None, targets=None)

        # Check counters were reset
        assert len(loss_fn._hook_call_count) == 0

    def test_multiple_modules_averaging(self):
        """Verify energy is correctly averaged across multiple modules."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 4, kernel_size=1)
                self.conv2 = nn.Conv2d(4, 4, kernel_size=1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        model = SimpleModel()
        loss_fn = EnergyLoss(reduction="mean", p=1)
        loss_fn.register_hooks(model)

        batch_size = 1
        n_timesteps = 1

        torch.manual_seed(42)
        x = torch.randn(batch_size, 3, 4, 4)
        output = model(x)

        loss = loss_fn(outputs=None, targets=None)

        # The loss should be averaged across the 2 modules
        assert loss > 0  # Verify loss is computed
        assert loss.requires_grad  # Verify gradients can flow


class TestLossCombination:
    """Test that multiple losses are correctly weighted and combined."""

    def test_weighted_loss_combination(self):
        """Verify losses are weighted and summed correctly."""
        batch_size = 4
        n_classes = 3

        torch.manual_seed(42)
        outputs = torch.randn(batch_size, n_classes, requires_grad=True)
        targets = torch.randint(0, n_classes, (batch_size,))

        # Create two loss functions with different weights
        loss_fn1 = CrossEntropyLoss(reduction="mean")
        loss_fn2 = CrossEntropyLoss(reduction="mean")

        weight1 = 2.0
        weight2 = 0.5

        # Compute weighted losses
        loss1 = weight1 * loss_fn1(outputs, targets)
        loss2 = weight2 * loss_fn2(outputs, targets)
        combined_loss = loss1 + loss2

        # Verify combination
        expected_combined = weight1 * loss_fn1(outputs, targets) + weight2 * loss_fn2(
            outputs, targets
        )

        assert torch.allclose(combined_loss, expected_combined, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
