#!/usr/bin/env python
"""
Strategic comparison test for CorNet-RT: Original vs DynVision Reimplementation

This script systematically compares the original CorNet-RT model with the DynVision
reimplementation to identify any discrepancies in architecture, weights, or outputs.

Test Strategy:
1. Load both models with pretrained weights
2. Compare model weights
3. Run forward passes with identical inputs
4. Compare layer-by-layer activations
5. Analyze temporal dynamics across timesteps
6. Diagnose root cause of any divergence
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dynvision.models.cornet_original import CORnet_RT as OriginalCorNetRT
from dynvision.models.cornet_rt import CorNetRT as ReimplementedCorNetRT


@dataclass
class ComparisonMetrics:
    """Metrics for comparing two tensors."""
    mean_abs_diff: float
    max_abs_diff: float
    mean_rel_diff: float
    max_rel_diff: float
    orig_mean: float
    orig_std: float
    reimpl_mean: float
    reimpl_std: float

    def is_match(self, threshold: float = 1e-4) -> bool:
        """Check if tensors match within threshold."""
        return self.mean_abs_diff < threshold

    def __str__(self) -> str:
        status = "✓ MATCH" if self.is_match() else "✗ DIVERGE"
        return (
            f"{status}\n"
            f"  Original:  mean={self.orig_mean:8.6f}, std={self.orig_std:8.6f}\n"
            f"  Reimpl:    mean={self.reimpl_mean:8.6f}, std={self.reimpl_std:8.6f}\n"
            f"  Abs diff:  mean={self.mean_abs_diff:8.6f}, max={self.max_abs_diff:8.6f}\n"
            f"  Rel diff:  mean={self.mean_rel_diff:8.6f}, max={self.max_rel_diff:8.6f}"
        )


class ModelLoader:
    """Handles loading and setup of both model implementations."""

    @staticmethod
    def load_original(device: str = "cpu") -> OriginalCorNetRT:
        """Load original CorNet-RT with pretrained weights."""
        print("\n" + "=" * 80)
        print("Loading Original CorNet-RT")
        print("=" * 80)

        model = OriginalCorNetRT(times=5)

        # Download pretrained weights
        url = "https://s3.amazonaws.com/cornet-models/cornet_rt-933c001c.pth"
        ckpt_data = torch.hub.load_state_dict_from_url(url, map_location=device)
        state_dict = ckpt_data["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Model loaded: {param_count:,} parameters")

        return model, state_dict

    @staticmethod
    def load_reimplemented(device: str = "cpu") -> ReimplementedCorNetRT:
        """Load reimplemented CorNetRT with pretrained weights."""
        print("\n" + "=" * 80)
        print("Loading Reimplemented CorNetRT")
        print("=" * 80)

        model = ReimplementedCorNetRT(
            n_timesteps=5,
            input_dims=(5, 3, 224, 224),
            n_classes=1000,
        )
        model.setup("test")
        model.to(device)
        model.eval()

        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Model loaded: {param_count:,} parameters")
        print(f"  Config: n_timesteps={model.n_timesteps}, dt={model.dt}, "
              f"t_feedforward={model.t_feedforward}, t_recurrence={model.t_recurrence}")

        return model


class ModelComparator:
    """Compares original and reimplemented CorNet-RT models."""

    def __init__(self, original: nn.Module, reimplemented: nn.Module, device: str = "cpu"):
        self.original = original
        self.reimplemented = reimplemented
        self.device = device
        self.layer_names = ["V1", "V2", "V4", "IT"]

    def compare_weights(self, orig_state_dict: Dict) -> bool:
        """Compare weights between original and reimplemented models."""
        print("\n" + "=" * 80)
        print("WEIGHT COMPARISON")
        print("=" * 80)

        reimpl_state = self.reimplemented.state_dict()
        translate = self.reimplemented.translate_pretrained_layer_names()

        matches = 0
        mismatches = []

        for orig_name, orig_weight in orig_state_dict.items():
            # Translate parameter name
            reimpl_name = orig_name
            for old, new in translate.items():
                reimpl_name = reimpl_name.replace(old, new)

            if reimpl_name not in reimpl_state:
                mismatches.append((orig_name, "missing"))
                continue

            reimpl_weight = reimpl_state[reimpl_name]

            if orig_weight.shape != reimpl_weight.shape:
                mismatches.append((orig_name, "shape"))
                continue

            if torch.allclose(orig_weight, reimpl_weight, atol=1e-6):
                matches += 1
            else:
                mismatches.append((orig_name, "value"))

        print(f"Matched: {matches}/{len(orig_state_dict)} parameters")
        if mismatches:
            print(f"Mismatches: {len(mismatches)}")
            for name, reason in mismatches[:3]:
                print(f"  - {name}: {reason}")

        return len(mismatches) == 0

    def create_inputs(self, batch_size: int = 2, seed: int = 42) -> Dict[str, torch.Tensor]:
        """Create test inputs with ImageNet normalization."""
        print("\n" + "=" * 80)
        print("Creating Test Inputs")
        print("=" * 80)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create random image and normalize
        img = torch.randint(0, 256, (batch_size, 3, 224, 224), dtype=torch.float32)
        img = img / 255.0

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        img_normalized = normalize(img)

        # Original: [B, C, H, W]
        # Reimplemented: [B, T, C, H, W] - same image repeated
        img_temporal = img_normalized.unsqueeze(1).repeat(1, 5, 1, 1, 1)

        inputs = {
            "original": img_normalized.to(self.device),
            "reimplemented": img_temporal.to(self.device),
        }

        print(f"✓ Created inputs (batch_size={batch_size})")
        print(f"  Original shape: {inputs['original'].shape}")
        print(f"  Reimplemented shape: {inputs['reimplemented'].shape}")
        print(f"  Stats: mean={img_normalized.mean():.3f}, std={img_normalized.std():.3f}")

        return inputs

    @staticmethod
    def compute_metrics(orig: torch.Tensor, reimpl: torch.Tensor) -> ComparisonMetrics:
        """Compute comparison metrics between two tensors."""
        abs_diff = (orig - reimpl).abs()
        rel_diff = abs_diff / (orig.abs() + 1e-8)

        return ComparisonMetrics(
            mean_abs_diff=abs_diff.mean().item(),
            max_abs_diff=abs_diff.max().item(),
            mean_rel_diff=rel_diff.mean().item(),
            max_rel_diff=rel_diff.max().item(),
            orig_mean=orig.mean().item(),
            orig_std=orig.std().item(),
            reimpl_mean=reimpl.mean().item(),
            reimpl_std=reimpl.std().item(),
        )

    def compare_outputs(self, inputs: Dict[str, torch.Tensor]) -> ComparisonMetrics:
        """Compare final model outputs."""
        print("\n" + "=" * 80)
        print("FORWARD PASS COMPARISON")
        print("=" * 80)

        # Reset reimplemented model
        self.reimplemented.reset()

        with torch.no_grad():
            out_orig = self.original(inputs["original"])
            out_reimpl = self.reimplemented(inputs["reimplemented"])

        if out_reimpl is None:
            print("✗ ERROR: Reimplemented model returned None!")
            return None

        # Handle temporal dimension
        if out_reimpl.dim() == 3:  # [B, T, Classes]
            out_reimpl = out_reimpl[:, -1, :]  # Take last timestep

        metrics = self.compute_metrics(out_orig, out_reimpl)

        print("\nFinal Output Comparison:")
        print(metrics)

        # Compare top predictions
        print("\nTop-5 Predictions:")
        for i in range(min(2, out_orig.shape[0])):
            orig_top5 = torch.topk(out_orig[i], 5)
            reimpl_top5 = torch.topk(out_reimpl[i], 5)

            match = "✓" if orig_top5.indices[0] == reimpl_top5.indices[0] else "✗"
            print(f"  Sample {i}: {match}")
            print(f"    Original:     {orig_top5.indices.tolist()[:3]}")
            print(f"    Reimplemented: {reimpl_top5.indices.tolist()[:3]}")

        return metrics

    def compare_layer_activations(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, ComparisonMetrics]:
        """Compare activations at each layer using hooks."""
        print("\n" + "=" * 80)
        print("LAYER-BY-LAYER COMPARISON")
        print("=" * 80)

        # Setup hooks for original model
        # NOTE: The original model loops over timesteps internally (self.times=5)
        # Hooks will capture the FINAL output after all timesteps complete
        orig_activations = {}
        orig_hooks = []

        for name in self.layer_names:
            def make_hook(layer_name):
                def hook(module, input, output):
                    # This hook fires after each timestep, but we only want the final one
                    # The original model's forward() processes all timesteps internally
                    if isinstance(output, tuple):
                        orig_activations[layer_name] = output[0].detach().cpu()
                    else:
                        orig_activations[layer_name] = output.detach().cpu()
                return hook

            layer = getattr(self.original, name)
            hook = layer.register_forward_hook(make_hook(name))
            orig_hooks.append(hook)

        # Run original model (internally loops 5 times)
        self.original.eval()
        with torch.no_grad():
            _ = self.original(inputs["original"])

        print("\nOriginal model captured:", list(orig_activations.keys()))

        # For reimplemented model, manually run _forward() to capture responses_t dict
        # The "record" operation stores activations (after nonlin) in responses_t
        reimpl_activations = {}

        self.reimplemented.reset()
        self.reimplemented.eval()

        with torch.no_grad():
            # Run forward and capture responses for last timestep
            x_input = inputs["reimplemented"]
            batch_size = x_input.shape[0]
            n_timesteps = x_input.shape[1]

            # Reset and run to final timestep
            self.reimplemented.reset(input_shape=x_input.shape)

            # Process each timestep and track when divergence begins
            timestep_responses = {t: {} for t in range(n_timesteps)}

            for t in range(n_timesteps):
                x = x_input[:, t, ...]

                # Debug: Check V2 hidden state before forward
                if t <= 2:
                    v2_h = self.reimplemented.V2.get_hidden_state(delay=1)
                    v2_buffer_len = len(self.reimplemented.V2._hidden_states)
                    print(f"\nt={t} BEFORE forward: V2 buffer len={v2_buffer_len}, hidden_state(delay=1)={'None' if v2_h is None else f'tensor(shape={v2_h.shape}, mean={v2_h.mean().item():.6f})'}")

                _, responses_t = self.reimplemented._forward(
                    x, t=t, feedforward_only=self.reimplemented.feedforward_only, store_responses=True
                )

                # Debug: Check V2 hidden state after forward
                if t <= 2:
                    v2_h = self.reimplemented.V2.get_hidden_state(delay=1)
                    v2_buffer_len = len(self.reimplemented.V2._hidden_states)
                    print(f"t={t} AFTER forward: V2 buffer len={v2_buffer_len}, hidden_state(delay=1)={'None' if v2_h is None else f'tensor(shape={v2_h.shape}, mean={v2_h.mean().item():.6f})'}")

                # Store all timestep responses for analysis
                for layer_name, activation in responses_t.items():
                    if activation is not None:
                        timestep_responses[t][layer_name] = activation.detach().cpu()

                # Store responses from last timestep for comparison
                if t == n_timesteps - 1:
                    for layer_name, activation in responses_t.items():
                        if activation is not None:
                            reimpl_activations[layer_name] = activation.detach().cpu()

        print("\nTimestep-by-timestep analysis:")
        print("(Comparing final timestep t=4 against original)")
        for t in range(n_timesteps):
            print(f"t={t} responses captured: {list(timestep_responses[t].keys())}")

        # Compare layers
        results = {}
        for name in self.layer_names:
            if name not in orig_activations:
                print(f"{name}: ✗ No original activation captured")
                continue

            if name not in reimpl_activations:
                print(f"{name}: ✗ No reimplemented activation captured")
                continue

            orig_act = orig_activations[name]
            reimpl_act = reimpl_activations[name]

            if orig_act.shape != reimpl_act.shape:
                print(f"{name}: ✗ Shape mismatch: {orig_act.shape} vs {reimpl_act.shape}")
                continue

            metrics = self.compute_metrics(orig_act, reimpl_act)
            results[name] = metrics

            print(f"\n{name}:")
            print(metrics)

        # Cleanup hooks
        for hook in orig_hooks:
            hook.remove()

        return results


def main():
    """Run strategic comparison tests."""
    print("\n" + "=" * 80)
    print("CorNet-RT Strategic Comparison Test")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    loader = ModelLoader()
    original, orig_state = loader.load_original(device)
    reimplemented = loader.load_reimplemented(device)

    # Create comparator
    comparator = ModelComparator(original, reimplemented, device)

    # Compare weights
    weights_match = comparator.compare_weights(orig_state)

    # Create test inputs
    inputs = comparator.create_inputs(batch_size=2)

    # Compare outputs
    output_metrics = comparator.compare_outputs(inputs)

    # Compare layer activations
    layer_metrics = comparator.compare_layer_activations(inputs)

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Weights: {'✓ Match' if weights_match else '✗ Mismatch'}")
    if output_metrics:
        print(f"Final output: {'✓ Match' if output_metrics.is_match() else '✗ Diverge'} "
              f"(mean diff={output_metrics.mean_abs_diff:.6f})")

    if layer_metrics:
        print("\nLayer-by-layer:")
        for name, metrics in layer_metrics.items():
            status = "✓" if metrics.is_match() else "✗"
            print(f"  {name}: {status} (mean diff={metrics.mean_abs_diff:.6f})")


if __name__ == "__main__":
    main()
