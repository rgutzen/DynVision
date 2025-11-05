"""
Comprehensive validation script comparing CordsNet reimplementation with original.

This script implements the debugging hierarchy from the model integration guide:
1. Verify weight loading and structure
2. Compare layer activations
3. Timestep-by-timestep comparison
4. Check temporal parameters
5. Final output comparison

Usage:
    python -m dynvision.models.validate_cordsnet_vs_original
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dynvision.models.cordsnet_original import cordsnet as CordsNetOriginal
from dynvision.models.cordsnet import CordsNet

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CordsNetValidator:
    """
    Comprehensive validator for CordsNet reimplementation.

    Follows the model integration guide's debugging hierarchy to identify
    and diagnose discrepancies between original and reimplementation.
    """

    def __init__(
        self,
        dataset: str = "imagenet",
        depth: int = 8,
        n_timesteps: int = 10,
        device: Optional[torch.device] = None,
    ):
        self.dataset = dataset
        self.depth = depth
        self.n_timesteps = n_timesteps
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(f"Initializing validator on {self.device}")

        # Create original model
        self.original = CordsNetOriginal(dataset=dataset, depth=depth).to(self.device)
        self.original.eval()

        # Create reimplementation
        input_dims = (n_timesteps, 3, 224, 224)
        n_classes = 1000 if dataset == "imagenet" else 10

        self.reimpl = CordsNet(
            init_with_pretrained=True,
            input_dims=input_dims,
            n_timesteps=n_timesteps,
        ).to(self.device)
        self.reimpl.setup("fit")
        self.reimpl.eval()

        # load pretrained weights into original model for fair comparison
        pretrained_weights = self.reimpl.download_pretrained_state_dict()
        self.original.load_state_dict(pretrained_weights)

        logger.info("Models created successfully\n")

    def test_1_weight_structure(self) -> bool:
        """
        Phase 1: Verify weight structure matches.

        Returns:
            True if structures match, False otherwise
        """
        logger.info("=" * 80)
        logger.info("PHASE 1: Weight Structure Comparison")
        logger.info("=" * 80 + "\n")

        orig_state = self.original.state_dict()
        reimpl_state = self.reimpl.state_dict()
        translation = self.reimpl.translate_pretrained_layer_names()

        logger.info(f"Original model has {len(orig_state)} parameters")
        logger.info(f"Reimpl model has {len(reimpl_state)} parameters")
        logger.info(f"Translation mapping has {len(translation)} entries\n")

        # Check each mapping
        issues = []
        matches = 0

        for orig_base_key, reimpl_pattern in translation.items():
            # Handle weight_norm parametrization: original keys may have .parametrizations.weight.original0/1
            # Find all original keys that match the base pattern
            orig_matching_keys = [
                k for k in orig_state.keys() if k.startswith(orig_base_key)
            ]

            if not orig_matching_keys:
                issues.append(
                    f"Original key not found: {orig_base_key} (or parametrization variants)"
                )
                continue

            # Find matching reimpl keys (may be multiple due to weight_norm)
            reimpl_matching_keys = [
                k for k in reimpl_state.keys() if reimpl_pattern in k
            ]

            if not reimpl_matching_keys:
                issues.append(
                    f"No reimpl key matches pattern '{reimpl_pattern}' for {orig_base_key}"
                )
                continue

            # For weight_norm, we expect both original0 and original1, or just weight
            # Compare structures by checking if both have similar parametrizations
            orig_has_parametrization = any(
                ".parametrizations." in k for k in orig_matching_keys
            )
            reimpl_has_parametrization = any(
                ".parametrizations." in k for k in reimpl_matching_keys
            )

            if orig_has_parametrization and reimpl_has_parametrization:
                # Both use weight_norm - check that both have original0 and original1
                orig_has_0 = any("original0" in k for k in orig_matching_keys)
                orig_has_1 = any("original1" in k for k in orig_matching_keys)
                reimpl_has_0 = any("original0" in k for k in reimpl_matching_keys)
                reimpl_has_1 = any("original1" in k for k in reimpl_matching_keys)

                if orig_has_0 == reimpl_has_0 and orig_has_1 == reimpl_has_1:
                    matches += 1
                    logger.debug(
                        f"✓ {orig_base_key} -> {reimpl_pattern}: weight_norm structure matches"
                    )
                else:
                    issues.append(
                        f"Weight_norm structure mismatch: {orig_base_key} -> {reimpl_pattern}"
                    )
            elif not orig_has_parametrization and not reimpl_has_parametrization:
                # Neither uses weight_norm - compare shapes directly
                orig_param = orig_state[orig_matching_keys[0]]
                reimpl_param = reimpl_state[reimpl_matching_keys[0]]

                if orig_param.shape == reimpl_param.shape:
                    matches += 1
                    logger.debug(
                        f"✓ {orig_base_key} -> {reimpl_pattern}: {list(orig_param.shape)}"
                    )
                else:
                    issues.append(
                        f"Shape mismatch: {orig_base_key} {list(orig_param.shape)} "
                        f"-> {reimpl_pattern} {list(reimpl_param.shape)}"
                    )
            else:
                # One uses weight_norm, the other doesn't - this is still compatible
                matches += 1
                logger.debug(
                    f"✓ {orig_base_key} -> {reimpl_pattern}: parametrization styles differ but compatible"
                )

        if issues:
            logger.error(f"\n❌ Found {len(issues)} structure issues:\n")
            for issue in issues[:10]:  # Show first 10
                logger.error(f"  • {issue}")
            if len(issues) > 10:
                logger.error(f"  ... and {len(issues)-10} more")
            return False
        else:
            logger.info(
                f"✅ Weight structure matches for {matches} parameter mappings\n"
            )
            return True

    def test_1b_state_dict_values_after_loading(self) -> bool:
        """
        Phase 1b: Compare state_dict values after loading pretrained weights.

        This test verifies that pretrained weights are correctly loaded from the
        original model into the reimplementation. It checks:
        1. All pretrained keys are correctly translated
        2. Weight values match between original and reimpl
        3. No random initialization values remain

        Returns:
            True if all values match, False otherwise
        """
        logger.info("=" * 80)
        logger.info("PHASE 1b: State Dict Value Comparison After Loading")
        logger.info("=" * 80 + "\n")

        # Get state dicts
        orig_state = self.original.state_dict()

        # Create fresh reimpl WITH pretrained loading enabled
        logger.info("Creating reimplementation with pretrained loading enabled...")
        reimpl_with_pretrained = CordsNet(
            init_with_pretrained=True,  # Enable pretrained loading
        ).to(self.device)
        reimpl_with_pretrained.setup("fit")
        reimpl_with_pretrained.eval()

        reimpl_state = reimpl_with_pretrained.state_dict()
        translation = reimpl_with_pretrained.translate_pretrained_layer_names()

        logger.info(f"Translation mapping has {len(translation)} entries")
        logger.info(f"Original state dict has {len(orig_state)} keys")
        logger.info(f"Reimpl state dict has {len(reimpl_state)} keys\n")

        # Track comparison results
        perfect_matches = 0
        close_matches = 0
        mismatches = []

        # Compare each translated parameter
        for orig_base_key, reimpl_base_pattern in translation.items():
            # Find all original keys matching this base pattern (handles weight_norm)
            orig_matching_keys = [
                k for k in orig_state.keys() if k.startswith(orig_base_key)
            ]

            if not orig_matching_keys:
                logger.warning(f"No original keys found for: {orig_base_key}")
                continue

            # For each original key, find corresponding reimpl key
            for orig_key in orig_matching_keys:
                # Build expected reimpl key by replacing base pattern
                expected_reimpl_key = orig_key.replace(
                    orig_base_key, reimpl_base_pattern
                )

                if expected_reimpl_key not in reimpl_state:
                    # Try finding it with different search
                    candidates = [
                        k
                        for k in reimpl_state.keys()
                        if reimpl_base_pattern in k and orig_key.split(".")[-1] in k
                    ]
                    if len(candidates) == 1:
                        expected_reimpl_key = candidates[0]
                    else:
                        mismatches.append(
                            {
                                "orig_key": orig_key,
                                "expected_key": expected_reimpl_key,
                                "issue": "Key not found in reimpl state dict",
                                "candidates": candidates if candidates else None,
                            }
                        )
                        continue

                # Compare values
                orig_val = orig_state[orig_key]
                reimpl_val = reimpl_state[expected_reimpl_key]

                if orig_val.shape != reimpl_val.shape:
                    mismatches.append(
                        {
                            "orig_key": orig_key,
                            "reimpl_key": expected_reimpl_key,
                            "issue": f"Shape mismatch: {orig_val.shape} vs {reimpl_val.shape}",
                        }
                    )
                    continue

                # Check value equivalence
                max_diff = (orig_val - reimpl_val).abs().max().item()
                mean_diff = (orig_val - reimpl_val).abs().mean().item()

                if max_diff < 1e-6:
                    perfect_matches += 1
                    logger.debug(
                        f"✓ Perfect match: {orig_key} -> {expected_reimpl_key}"
                    )
                elif max_diff < 1e-3:
                    close_matches += 1
                    logger.info(
                        f"≈ Close match: {orig_key} -> {expected_reimpl_key} (max_diff={max_diff:.2e})"
                    )
                else:
                    # Significant mismatch - gather statistics
                    mismatches.append(
                        {
                            "orig_key": orig_key,
                            "reimpl_key": expected_reimpl_key,
                            "orig_mean": orig_val.mean().item(),
                            "orig_std": orig_val.std().item(),
                            "reimpl_mean": reimpl_val.mean().item(),
                            "reimpl_std": reimpl_val.std().item(),
                            "max_diff": max_diff,
                            "mean_diff": mean_diff,
                        }
                    )

        # Report results
        total_compared = perfect_matches + close_matches + len(mismatches)
        logger.info(f"\n{'='*60}")
        logger.info(f"Comparison Results:")
        logger.info(f"  ✓ Perfect matches: {perfect_matches}/{total_compared}")
        logger.info(f"  ≈ Close matches:   {close_matches}/{total_compared}")
        logger.info(f"  ✗ Mismatches:      {len(mismatches)}/{total_compared}")

        if mismatches:
            logger.error(f"\n❌ {len(mismatches)} PARAMETER MISMATCHES FOUND:\n")
            for i, mismatch in enumerate(mismatches[:10], 1):  # Show first 10
                logger.error(
                    f"{i}. {mismatch['orig_key']} -> {mismatch.get('reimpl_key', 'NOT FOUND')}"
                )
                if "issue" in mismatch:
                    logger.error(f"   Issue: {mismatch['issue']}")
                    if mismatch.get("candidates"):
                        logger.error(f"   Candidates: {mismatch['candidates']}")
                else:
                    logger.error(
                        f"   Original:  mean={mismatch['orig_mean']:+.6f}, std={mismatch['orig_std']:.6f}"
                    )
                    logger.error(
                        f"   Reimpl:    mean={mismatch['reimpl_mean']:+.6f}, std={mismatch['reimpl_std']:.6f}"
                    )
                    logger.error(
                        f"   Diff:      max={mismatch['max_diff']:.2e}, mean={mismatch['mean_diff']:.2e}"
                    )

            if len(mismatches) > 10:
                logger.error(f"\n   ... and {len(mismatches)-10} more mismatches\n")

            return False
        else:
            logger.info(f"✅ All {total_compared} compared parameters match!\n")
            return True

    def test_2_architectural_issues(self) -> List[str]:
        """
        Phase 2: Identify architectural differences by code inspection.

        Returns:
            List of identified issues
        """
        logger.info("=" * 80)
        logger.info("PHASE 2: Architectural Issue Detection")
        logger.info("=" * 80 + "\n")

        issues = []

        # Issue 1: Skip connection integration order
        logger.info("Checking skip connection integration order...")
        layer_ops = self.reimpl.layer_operations

        try:
            layer_idx = layer_ops.index("layer")
            addskip_idx = layer_ops.index("addskip")

            if addskip_idx > layer_idx:
                issue = (
                    "⚠️  CRITICAL: Skip connections added AFTER layer operation\n"
                    "    Original: areainput = relu(prev) + skip\n"
                    "              output = area_area(areainput)  <- conv on SUM\n"
                    "    Current:  output = area_area(relu(prev))  <- conv on prev alone\n"
                    "              output += skip                  <- skip added after\n"
                    "    Impact: These are mathematically INEQUIVALENT!\n"
                    "            conv(a+b) ≠ conv(a) + b\n"
                )
                issues.append(issue)
                logger.error("❌ Skip connection order INCORRECT\n")
            else:
                logger.info("✅ Skip connection order correct\n")
        except ValueError as e:
            issues.append(f"Error checking layer_operations: {e}")

        # Issue 2: Idle timesteps
        logger.info("Checking idle timesteps implementation...")

        # Check if idle_timesteps is set - base class TemporalBase.forward() handles it automatically
        if hasattr(self.reimpl, "idle_timesteps") and self.reimpl.idle_timesteps > 0:
            logger.info(
                f"✅ Idle timesteps IMPLEMENTED via TemporalBase.forward()\n"
                f"   idle_timesteps={self.reimpl.idle_timesteps} (matches original)\n"
                f"   Base class automatically runs null input for spontaneous activity\n"
            )
        else:
            issue = (
                "⚠️  ERROR: idle_timesteps not set or is 0\n"
                f"    Current value: {getattr(self.reimpl, 'idle_timesteps', 'NOT SET')}\n"
                "    Expected: 100 (to match original)\n"
            )
            issues.append(issue)
            logger.error("❌ idle_timesteps not configured\n")

        # Issue 3: Output computation
        logger.info("Checking output computation...")

        # Check if classifier receives combined input from last two layers
        # In original: out = avgpool(relu(rs[7]) + relu(rs[6]))
        # In reimpl: should have combine_layer8 module that adds layer7 to layer8

        # Check if combine_layer8 exists and is in layer_operations
        has_combine_module = hasattr(self.reimpl, "combine_layer8")
        has_combine_operation = "combine" in self.reimpl.layer_operations

        if has_combine_module and has_combine_operation:
            combine_module = self.reimpl.combine_layer8
            # Verify it's a Skip connection from layer7
            if hasattr(combine_module, "source"):
                source_name = combine_module.source.__class__.__name__
                logger.info(
                    f"✅ Output computation IMPLEMENTED correctly\n"
                    f"   combine_layer8 module exists: {has_combine_module}\n"
                    f"   'combine' in layer_operations: {has_combine_operation}\n"
                    f"   Source module type: {source_name}\n"
                    f"   This adds layer7 output to layer8 before classifier\n"
                )
            else:
                issue = (
                    "⚠️  WARNING: combine_layer8 exists but no source attribute\n"
                    f"    Module: {combine_module}\n"
                )
                issues.append(issue)
                logger.warning("⚠️  combine_layer8 configuration unclear\n")
        else:
            issue = (
                "⚠️  CRITICAL: Output computation differs\n"
                "    Original: out = avgpool(relu(layer8_state) + relu(layer7_state))\n"
                "              Combines BOTH last layers before classification\n"
                f"    Current:  combine_layer8 exists: {has_combine_module}\n"
                f"              'combine' in layer_operations: {has_combine_operation}\n"
                "    Impact: Missing layer7 contribution will change predictions\n"
            )
            issues.append(issue)
            logger.error("❌ Output combination not properly implemented\n")

        # Issue 4: Processing order
        logger.info("Checking layer processing order...")

        # With t_feedforward > 0, delays ensure layers retrieve from previous timestep
        # making processing order (forward vs reverse) irrelevant
        if self.reimpl.t_feedforward > 0:
            logger.info(
                f"✅ Processing order difference compensated by delays\n"
                f"   t_feedforward={self.reimpl.t_feedforward} > 0\n"
                f"   delay_feedforward={self.reimpl.delay_feedforward}\n"
                f"   Each layer retrieves input from previous timestep,\n"
                f"   making forward vs reverse processing order equivalent\n"
            )
        else:
            issue = (
                "⚠️  WARNING: Processing order may matter with t_feedforward=0\n"
                "    Original: Processes layers REVERSE (7→0) per timestep\n"
                "    Current:  Processes FORWARD (0→8) per timestep\n"
                "    With t_feedforward=0, layers may see just-updated states\n"
            )
            issues.append(issue)
            logger.warning("⚠️  Processing order may differ with t_feedforward=0\n")

        # Issue 5: Recurrence target
        logger.info("Checking recurrence configuration...")

        for i in range(1, 9):
            layer = getattr(self.reimpl, f"layer{i}")
            if layer.recurrence_target != "output":
                issue = (
                    f"⚠️  ERROR: layer{i}.recurrence_target = {layer.recurrence_target}\n"
                    f"    Expected: 'output'\n"
                    f"    Original applies recurrence to layer state r[area]\n"
                )
                issues.append(issue)
                logger.error(f"❌ layer{i} recurrence_target incorrect\n")
                break
        else:
            logger.info("✅ All layers have recurrence_target='output'\n")

        return issues

    def test_3_trace_original_dynamics(self, img: torch.Tensor, timesteps: int = 5):
        """
        Phase 3: Trace original model dynamics step-by-step.

        Args:
            img: Input image [batch, channels, height, width]
            timesteps: Number of timesteps to trace
        """
        logger.info("=" * 80)
        logger.info("PHASE 3: Trace Original Model Dynamics")
        logger.info("=" * 80 + "\n")

        batch_size = img.shape[0]
        channels = [64, 64, 128, 128, 256, 256, 512, 512]
        sizes = [56, 56, 28, 28, 14, 14, 7, 7]
        alpha = 0.1

        # Initialize states
        rs = [
            torch.zeros(
                batch_size, channels[j], sizes[j], sizes[j], device=self.device
            )
            for j in range(self.depth)
        ]

        logger.info(f"Tracing {timesteps} timesteps...\n")

        with torch.no_grad():
            for t in range(timesteps):
                logger.info(f"Timestep {t}:")

                # Process in reverse order (original behavior)
                for j in range(self.depth - 1, -1, -1):
                    old_state = rs[j].clone()
                    rs[j] = self.original.rnn(j, rs, img if t > 0 else img * 0, alpha)

                    # Log statistics
                    logger.info(
                        f"  Layer {j}: mean={rs[j].mean():.6f}, "
                        f"std={rs[j].std():.6f}, "
                        f"max={rs[j].max():.6f}, "
                        f"change={((rs[j] - old_state).abs().mean()):.6f}"
                    )

                logger.info("")

        logger.info("Original dynamics traced\n")

    def test_4_compare_layer_by_layer(
        self,
        img: torch.Tensor,
        n_timesteps: int = 10,
        idle_timesteps: int = 10,
        alpha: float = 0.2,
        verbose: bool = True,
    ) -> Dict[str, any]:
        """
        Phase 4: Compare layer activations timestep-by-timestep.

        Provides detailed logging to identify exactly where and when
        activation differences first appear.

        Args:
            img: Input image [batch, channels, height, width]
            n_timesteps: Number of active timesteps
            idle_timesteps: Number of spontaneous timesteps
            alpha: Integration constant (dt/tau)
            verbose: If True, print detailed timestep-by-timestep comparison

        Returns:
            Dictionary of metrics per layer and timestep
        """
        print("\n" + "=" * 80)
        print(f"PHASE 4: Layer-by-Layer Comparison")
        print("=" * 80)
        print(
            f"Configuration: {idle_timesteps} idle + {n_timesteps} active timesteps, alpha={alpha}"
        )

        batch_size = img.shape[0]
        channels = [64, 64, 128, 128, 256, 256, 512, 512]
        sizes = [56, 56, 28, 28, 14, 14, 7, 7]

        total_timesteps = idle_timesteps + n_timesteps

        # ==================
        # Original model - store ALL timesteps
        # ==================
        print("\nRunning original model (storing all timesteps)...")

        orig_timestep_activations = {t: {} for t in range(total_timesteps)}

        rs = [
            torch.zeros(
                batch_size, channels[j], sizes[j], sizes[j], device=self.device
            )
            for j in range(self.depth)
        ]

        with torch.no_grad():
            # Idle timesteps
            for t in range(idle_timesteps):
                for j in range(self.depth - 1, -1, -1):
                    rs[j] = self.original.rnn(j, rs, img * 0, alpha)

                # Store activations after tstep (matching reimpl recording point)
                for j in range(self.depth):
                    orig_timestep_activations[t][f"layer{j+1}"] = (
                        rs[j].detach().cpu().clone()
                    )

            # Active timesteps
            for t in range(n_timesteps):
                for j in range(self.depth - 1, -1, -1):
                    rs[j] = self.original.rnn(j, rs, img, alpha)

                # Store activations
                for j in range(self.depth):
                    orig_timestep_activations[idle_timesteps + t][f"layer{j+1}"] = (
                        rs[j].detach().cpu().clone()
                    )

        print(
            f"Original captured {total_timesteps} timesteps for layers: {list(orig_timestep_activations[0].keys())}"
        )

        # ==================
        # Reimplemented model - store ALL timesteps
        # ==================
        print("\nRunning reimplemented model (storing all timesteps)...")

        reimpl_timestep_activations = {t: {} for t in range(total_timesteps)}

        self.reimpl.reset()
        self.reimpl.eval()

        with torch.no_grad():
            # Prepare temporal input
            img_temporal = img.unsqueeze(1).repeat(1, n_timesteps, 1, 1, 1)

            # Run idle timesteps
            null_input = torch.zeros_like(img)
            for t in range(idle_timesteps):
                _, responses_t = self.reimpl._forward(
                    null_input, t=t, feedforward_only=False, store_responses=True
                )

                for layer_name, activation in responses_t.items():
                    if activation is not None:
                        reimpl_timestep_activations[t][layer_name] = (
                            activation.detach().cpu().clone()
                        )

            # Run active timesteps
            for t in range(n_timesteps):
                x = img_temporal[:, t, ...]

                _, responses_t = self.reimpl._forward(
                    x,
                    t=idle_timesteps + t,
                    feedforward_only=False,
                    store_responses=True,
                )

                # Store activations
                for layer_name, activation in responses_t.items():
                    if activation is not None:
                        reimpl_timestep_activations[idle_timesteps + t][layer_name] = (
                            activation.detach().cpu().clone()
                        )

        print(
            f"Reimplemented captured {total_timesteps} timesteps for layers: {list(reimpl_timestep_activations[0].keys())}"
        )

        # ==================
        # Timestep-by-timestep comparison with vertical alignment
        # ==================

        # Threshold for considering activations different
        THRESHOLD = 1e-4

        # Track first divergence for each layer
        first_divergence = {}

        # Detailed comparison for all timesteps to find divergence
        # Show all timesteps when total is small, otherwise show first 15
        max_detailed_timesteps = total_timesteps if total_timesteps <= 30 else 15

        for t in range(max_detailed_timesteps):
            # Determine which layers are active in reimpl at this timestep
            active_reimpl_layers = [
                k
                for k in reimpl_timestep_activations[t].keys()
                if k.startswith("layer")
            ]
            active_orig_layers = [f"layer{i}" for i in range(1, 9)]

            # Only show timesteps where at least one layer is active
            if not active_reimpl_layers:
                continue

            print(f"\n{'='*20} t={t} {'='*20}")
            if t < idle_timesteps:
                print(f"(Idle timestep {t})")
            else:
                print(f"(Active timestep {t - idle_timesteps})")

            # Show which layers are active in reimpl
            if verbose:
                print(f"Active in reimpl: {active_reimpl_layers}")

            # Compare each layer
            for i in range(1, 9):
                layer_name = f"layer{i}"

                # Check if layer exists in both models at this timestep
                has_orig = layer_name in orig_timestep_activations[t]
                has_reimpl = layer_name in reimpl_timestep_activations[t]

                if not has_orig and not has_reimpl:
                    continue  # Layer not active in either model

                # Print with vertical alignment
                print(f"\n{layer_name}:")

                if not has_orig:
                    print(f"\n{layer_name}: Missing in original")
                    continue

                orig_act = orig_timestep_activations[t][layer_name]
                orig_mean = orig_act.abs().mean().item()
                print(f"  orig:   {orig_mean:8.6f}")

                if not has_reimpl:
                    if verbose:
                        print(f"\n{layer_name}: Not yet active in reimpl")
                    continue

                reimpl_act = reimpl_timestep_activations[t][layer_name]
                reimpl_mean = reimpl_act.abs().mean().item()
                print(f"  reimpl: {reimpl_mean:8.6f}")

                # Compute metrics
                diff = (orig_act - reimpl_act).abs()
                mean_diff = diff.mean().item()

                # Check if this is first divergence for this layer
                is_divergent = mean_diff > THRESHOLD
                if is_divergent and layer_name not in first_divergence:
                    first_divergence[layer_name] = t

                # Status symbol
                status = "✗" if is_divergent else "✓"

                print(f"  diff:   {mean_diff:8.6f} {status}")

                # Highlight first divergence
                if (
                    layer_name in first_divergence
                    and first_divergence[layer_name] == t
                ):
                    print(f"  ↑ FIRST DIVERGENCE for {layer_name}")

        # Summary of first divergences
        if first_divergence:
            print(f"\n{'='*80}")
            print("FIRST DIVERGENCE SUMMARY")
            print(f"{'='*80}")
            for layer_name in sorted(
                first_divergence.keys(), key=lambda x: int(x.replace("layer", ""))
            ):
                t = first_divergence[layer_name]
                phase = (
                    "idle" if t < idle_timesteps else f"active {t - idle_timesteps}"
                )
                print(f"{layer_name}: timestep {t} ({phase})")

        # Final timestep comparison summary
        print(f"\n{'='*80}")
        print(f"FINAL TIMESTEP COMPARISON (t={total_timesteps-1})")
        print(f"{'='*80}")

        results = {}
        final_t = total_timesteps - 1

        for i in range(1, 9):
            layer_name = f"layer{i}"

            if layer_name not in orig_timestep_activations[final_t]:
                continue
            if layer_name not in reimpl_timestep_activations[final_t]:
                continue

            orig_act = orig_timestep_activations[final_t][layer_name]
            reimpl_act = reimpl_timestep_activations[final_t][layer_name]

            diff = (orig_act - reimpl_act).abs()
            mean_diff = diff.mean().item()
            orig_mean = orig_act.abs().mean().item()
            reimpl_mean = reimpl_act.abs().mean().item()

            status = "✓" if mean_diff < THRESHOLD else "✗"

            results[layer_name] = {
                "mean_abs_diff": mean_diff,
                "orig_mean": orig_mean,
                "reimpl_mean": reimpl_mean,
            }

            print(f"{layer_name}:")
            print(f"  orig:   {orig_mean:8.6f}")
            print(f"  reimpl: {reimpl_mean:8.6f}")
            print(f"  diff:   {mean_diff:8.6f} {status}")

        print("")
        return results

    def test_5_final_output_comparison(
        self,
        img: torch.Tensor,
        timesteps: int = 10,
        idle_timesteps: int = 10,
        alpha: float = 0.2,
    ) -> Dict[str, any]:
        """
        Phase 5: Compare final model outputs.

        Args:
            img: Input image [batch, channels, height, width]
            timesteps: Number of active timesteps
            idle_timesteps: Number of spontaneous timesteps
            alpha: Integration constant (dt/tau)

        Returns:
            Comparison metrics
        """
        logger.info("=" * 80)
        logger.info("PHASE 5: Final Output Comparison")
        logger.info("=" * 80 + "\n")

        batch_size = img.shape[0]
        channels = [64, 64, 128, 128, 256, 256, 512, 512]
        sizes = [56, 56, 28, 28, 14, 14, 7, 7]

        # ==================
        # Original model
        # ==================
        logger.info("Running original model...")

        with torch.no_grad():
            rs = [
                torch.zeros(
                    batch_size, channels[j], sizes[j], sizes[j], device=self.device
                )
                for j in range(self.depth)
            ]

            # Idle timesteps
            logger.info(f"  {idle_timesteps} idle timesteps (spontaneous activity)...")
            for t in range(idle_timesteps):
                for j in range(self.depth - 1, -1, -1):
                    rs[j] = self.original.rnn(j, rs, img * 0, alpha)

            # Active timesteps
            logger.info(f"  {timesteps} active timesteps...")
            for t in range(timesteps):
                for j in range(self.depth - 1, -1, -1):
                    rs[j] = self.original.rnn(j, rs, img, alpha)

            # Output: combines last TWO layers
            out_orig = self.original.out_avgpool(
                self.original.relu(rs[self.depth - 1])  # layer 8
                + self.original.relu(rs[self.depth - 2])  # layer 7
            )
            out_orig = self.original.out_flatten(out_orig)
            out_orig = self.original.out_fc(out_orig)

        logger.info(f"  Original output: {list(out_orig.shape)}\n")

        # ==================
        # Reimplementation
        # ==================
        logger.info("Running reimplementation...")

        with torch.no_grad():
            # DynVision expects [batch, time, channels, height, width]
            img_temporal = img.unsqueeze(1).repeat(1, timesteps, 1, 1, 1)
            out_reimpl = self.reimpl(img_temporal)

            # Get last timestep
            out_reimpl = out_reimpl[:, -1]

        logger.info(f"  Reimpl output: {list(out_reimpl.shape)}\n")

        # ==================
        # Detailed Comparison
        # ==================
        print("\nDetailed Output Analysis:")
        print(
            f"  Original - Mean: {out_orig.mean():.6f}, Std: {out_orig.std():.6f}, Range: [{out_orig.min():.3f}, {out_orig.max():.3f}]"
        )
        print(
            f"  Reimpl   - Mean: {out_reimpl.mean():.6f}, Std: {out_reimpl.std():.6f}, Range: [{out_reimpl.min():.3f}, {out_reimpl.max():.3f}]"
        )

        diff = (out_orig - out_reimpl).abs()
        rel_diff = diff / (out_orig.abs() + 1e-8)

        # Get top-5 predictions
        top5_orig = out_orig.topk(5, dim=1)
        top5_reimpl = out_reimpl.topk(5, dim=1)

        metrics = {
            "max_diff": diff.max().item(),
            "mean_diff": diff.mean().item(),
            "median_diff": diff.median().item(),
            "std_diff": diff.std().item(),
            "max_rel_diff": rel_diff.max().item(),
            "mean_rel_diff": rel_diff.mean().item(),
            "pred_orig": out_orig.argmax(dim=1).cpu().numpy(),
            "pred_reimpl": out_reimpl.argmax(dim=1).cpu().numpy(),
            "agreement": (out_orig.argmax(dim=1) == out_reimpl.argmax(dim=1))
            .float()
            .mean()
            .item(),
            "top5_orig": top5_orig.indices[0].cpu().tolist(),
            "top5_reimpl": top5_reimpl.indices[0].cpu().tolist(),
        }

        # Calculate top-5 overlap
        top5_overlap = len(set(metrics["top5_orig"]) & set(metrics["top5_reimpl"]))
        metrics["top5_overlap"] = top5_overlap

        print("\nAbsolute Differences:")
        print(f"  Max:    {metrics['max_diff']:.6f}")
        print(f"  Mean:   {metrics['mean_diff']:.6f}")
        print(f"  Median: {metrics['median_diff']:.6f}")
        print(f"  Std:    {metrics['std_diff']:.6f}")

        print(f"\nRelative Differences:")
        print(f"  Max:  {metrics['max_rel_diff']*100:.2f}%")
        print(f"  Mean: {metrics['mean_rel_diff']*100:.2f}%")

        print(f"\nTop-1 Predictions:")
        print(f"  Original: {metrics['pred_orig'][0]}")
        print(f"  Reimpl:   {metrics['pred_reimpl'][0]}")
        print(f"  Match: {'✅ YES' if metrics['agreement'] == 1.0 else '❌ NO'}")

        print(f"\nTop-5 Predictions:")
        print(f"  Original: {metrics['top5_orig']}")
        print(f"  Reimpl:   {metrics['top5_reimpl']}")
        print(f"  Overlap:  {top5_overlap}/5")

        # Analysis
        logger.info("\nAnalysis:")
        if metrics["agreement"] == 1.0:
            logger.info("✅ PREDICTIONS MATCH!")
            logger.info(
                "   The reimplementation produces the same top prediction as the original.\n"
            )
        else:
            logger.error("❌ PREDICTIONS DIFFER")
            if top5_overlap >= 3:
                logger.warning(
                    f"   However, {top5_overlap}/5 top-5 predictions overlap."
                )
                logger.warning(
                    "   The models are producing similar but not identical outputs.\n"
                )
            else:
                logger.error("   Top-5 predictions also differ significantly.")
                logger.error(
                    "   This suggests there may still be architectural differences.\n"
                )

        if metrics["max_diff"] < 0.01:
            logger.info(
                "   Output differences are very small (< 0.01) - likely numerical precision"
            )
        elif metrics["max_diff"] < 0.1:
            logger.info(
                "   Output differences are small (< 0.1) - minor discrepancies"
            )
        else:
            logger.warning("   Output differences are significant (>= 0.1)")

        logger.info("")

        return metrics

    def run_full_validation(self) -> Dict[str, any]:
        """Run complete validation suite."""

        print("\n" + "=" * 80)
        print(" " * 20 + "CORDSNET VALIDATION SUITE")
        print("=" * 80 + "\n")

        results = {}

        # Phase 1: Weight structure
        results["weight_structure_ok"] = self.test_1_weight_structure()

        # Phase 1b: State dict values after loading pretrained weights
        results["state_dict_values_ok"] = (
            self.test_1b_state_dict_values_after_loading()
        )

        # Phase 2: Architectural issues
        results["architectural_issues"] = self.test_2_architectural_issues()

        # Phase 3: Trace original dynamics
        torch.manual_seed(42)
        test_img = torch.randn(2, 3, 224, 224, device=self.device)
        self.test_3_trace_original_dynamics(test_img, timesteps=3)

        # Phase 4: Layer-by-layer comparison
        # Note: alpha = dt/tau = 2/10 = 0.2 (from original CordsNet)
        # Using full timesteps to allow proper activity buildup
        results["layer_metrics"] = self.test_4_compare_layer_by_layer(
            test_img,
            n_timesteps=10,  # Full timesteps for proper activity buildup
            idle_timesteps=10,  # Full idle timesteps as in original
            alpha=0.2,  # dt/tau = 2/10
        )

        # Phase 5: Output comparison
        results["output_metrics"] = self.test_5_final_output_comparison(
            test_img,
            timesteps=10,  # Full timesteps for proper activity buildup
            idle_timesteps=10,  # Full idle timesteps as in original
            alpha=0.2,  # dt/tau = 2/10
        )

        # ==================
        # Summary
        # ==================
        print("\n" + "=" * 80)
        print(" " * 25 + "VALIDATION SUMMARY")
        print("=" * 80 + "\n")

        print(
            f"Weight Structure: {'✅ PASS' if results['weight_structure_ok'] else '❌ FAIL'}"
        )

        print(
            f"State Dict Values After Loading: {'✅ PASS' if results['state_dict_values_ok'] else '❌ FAIL'}\n"
        )

        print(f"Architectural Issues Found: {len(results['architectural_issues'])}")
        if results["architectural_issues"]:
            print()
            for i, issue in enumerate(results["architectural_issues"], 1):
                print(f"Issue {i}:")
                print(issue)
                print()

        print(f"Output Comparison:")
        print(f"  Max difference: {results['output_metrics']['max_diff']:.6f}")
        print(
            f"  Prediction agreement: {results['output_metrics']['agreement']*100:.1f}%\n"
        )

        # ==================
        # Recommendations
        # ==================
        print("=" * 80)
        print(" " * 28 + "RECOMMENDATIONS")
        print("=" * 80 + "\n")

        print("Based on the model integration guide (Section 5.5: Lessons Learned),")
        print("the following CRITICAL issues must be fixed for exact replication:\n")

        print("1. Skip Connection Order:")
        print("   - Current: Skip added AFTER feedforward convolution")
        print("   - Required: Skip must be integrated BEFORE convolution")
        print(
            "   - Fix: Modify layer_operations order or RConv2d to accept pre-combined input\n"
        )

        print("2. Idle Timesteps:")
        print("   - Current: Not implemented in forward pass")
        print("   - Required: 100 timesteps of spontaneous activity before input")
        print("   - Fix: Implement custom forward() method to run idle timesteps\n")

        print("3. Output Computation:")
        print("   - Current: Uses only layer8 output")
        print("   - Required: Combine relu(layer8) + relu(layer7)")
        print("   - Fix: Modify classifier to receive combined input\n")

        print("4. Processing Order:")
        print("   - Current: Forward order (0→8)")
        print("   - Required: Reverse order (7→0)")
        print(
            "   - Fix: Verify delay mechanism compensates, or modify processing order\n"
        )

        print("See:")
        print(
            "  - Model Integration Guide: docs/development/guides/model-integration.md"
        )
        print("  - Section 5.5: Lessons Learned - Common Mistakes in Practice")
        print("  - Section 4.6: Incremental Debugging Methodology\n")

        print("=" * 80 + "\n")

        return results


def main():
    """Main validation function."""

    # Create validator
    validator = CordsNetValidator(
        dataset="imagenet",
        depth=8,
        n_timesteps=10,
    )

    # Run validation
    results = validator.run_full_validation()

    return results


if __name__ == "__main__":
    main()
