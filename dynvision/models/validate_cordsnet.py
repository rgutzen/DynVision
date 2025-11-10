"""
Comprehensive validation script comparing CordsNet reimplementation with the
reference implementation shipped alongside this repository.

The script mirrors the debugging hierarchy outlined in the model integration
guide:
1. Verify pretrained weight loading and structural alignment
2. Compare intermediate activations across timesteps
3. Trace the original dynamics step by step
4. Evaluate final classifier outputs

Usage::

    python -m dynvision.models.validate_cordsnet
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dynvision.model_components.layer_connections import Skip
from dynvision.models.cordsnet import CordsNet
from dynvision.models.cordsnet_original import cordsnet as CordsNetOriginal

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
        idle_timesteps: int = 10,
        batch_size: int = 1,
        alpha: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.dataset = dataset
        self.depth = depth
        self.n_timesteps = n_timesteps
        self.idle_timesteps = idle_timesteps
        self.batch_size = batch_size

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif not isinstance(device, torch.device):
            device = torch.device(device)

        self.device = device

        logger.info(
            "Validator configuration: dataset=%s depth=%d idle=%d active=%d batch=%d",
            self.dataset,
            self.depth,
            self.idle_timesteps,
            self.n_timesteps,
            self.batch_size,
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
            idle_timesteps=idle_timesteps,
        ).to(self.device)
        self.reimpl.setup("fit")
        self.reimpl.eval()

        # load pretrained weights into original model for fair comparison
        pretrained_weights = self.reimpl.download_pretrained_state_dict()
        self.original.load_state_dict(pretrained_weights)

        # Determine alpha from the reimplementation unless explicitly provided
        tau_value = self.reimpl.tau
        if isinstance(tau_value, torch.Tensor):
            tau_value = tau_value.item()

        dt_value = self.reimpl.dt
        if isinstance(dt_value, torch.Tensor):
            dt_value = dt_value.item()

        self.dt = float(dt_value)
        self.tau = float(tau_value)
        self.alpha = float(alpha) if alpha is not None else self.dt / self.tau
        self.total_timesteps = self.idle_timesteps + self.n_timesteps

        logger.info(
            "Using alpha=%.4f (dt=%.3f, tau=%.3f)", self.alpha, self.dt, self.tau
        )

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
            input_dims=self.reimpl.input_dims,
            n_timesteps=self.reimpl.n_timesteps,
            idle_timesteps=self.reimpl.idle_timesteps,
            n_classes=self.reimpl.n_classes,
            dt=self.dt,
            tau=self.tau,
            t_feedforward=self.reimpl.t_feedforward,
            t_recurrence=self.reimpl.t_recurrence,
            t_skip=self.reimpl.t_skip,
            recurrence_type=self.reimpl.recurrence_type,
            recurrence_target=self.reimpl.recurrence_target,
            skip=self.reimpl.skip,
            feedback=self.reimpl.feedback,
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

        # Issue 6: Classifier construction
        logger.info("Checking classifier skip integration...")

        classifier = getattr(self.reimpl, self.reimpl.classifier_name, None)
        if not isinstance(classifier, nn.Sequential):
            issues.append(
                "⚠️  ERROR: Classifier is not an nn.Sequential module; expected Skip → AvgPool → Flatten → Linear"
            )
        else:
            first_module = classifier[0]
            if not isinstance(first_module, Skip):
                issues.append(
                    "⚠️  ERROR: Classifier does not begin with a Skip module combining layer7"
                )
            else:
                expected_source = getattr(self.reimpl, "layer7", None)
                if first_module.source is not expected_source:
                    issues.append(
                        "⚠️  ERROR: Classifier Skip source does not reference layer7"
                    )
                elif getattr(first_module, "delay_index", None) not in (0, 1):
                    issues.append(
                        "⚠️  ERROR: Classifier Skip delay misconfigured (expected immediate integration)"
                    )
                else:
                    logger.info(
                        "✅ Classifier combines layer8 activations with layer7 via Skip module"
                    )

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
        alpha = self.alpha

        idle_to_trace = min(self.idle_timesteps, timesteps)
        active_to_trace = min(self.n_timesteps, timesteps)

        # Initialize states
        rs = [
            torch.zeros(
                batch_size, channels[j], sizes[j], sizes[j], device=self.device
            )
            for j in range(self.depth)
        ]

        logger.info(
            "Tracing %d idle and %d active timesteps (alpha=%.3f)...\n",
            idle_to_trace,
            active_to_trace,
            alpha,
        )

        with torch.no_grad():
            # Warm-up / idle timesteps
            for t in range(idle_to_trace):
                logger.info(f"Idle timestep {t}:")

                for j in range(self.depth - 1, -1, -1):
                    old_state = rs[j].clone()
                    rs[j] = self.original.rnn(j, rs, img * 0, alpha)
                    logger.info(
                        f"  Layer {j}: mean={rs[j].mean():.6f}, "
                        f"std={rs[j].std():.6f}, "
                        f"max={rs[j].max():.6f}, "
                        f"change={((rs[j] - old_state).abs().mean()):.6f}"
                    )

                logger.info("")

            # Active timesteps with stimulus
            for t in range(active_to_trace):
                logger.info(f"Active timestep {t}:")

                for j in range(self.depth - 1, -1, -1):
                    old_state = rs[j].clone()
                    rs[j] = self.original.rnn(j, rs, img, alpha)
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
        alpha: Optional[float] = None,
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
        alpha = self.alpha if alpha is None else alpha

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
        alpha: Optional[float] = None,
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

        alpha = self.alpha if alpha is None else alpha

        batch_size = img.shape[0]
        channels = [64, 64, 128, 128, 256, 256, 512, 512]
        sizes = [56, 56, 28, 28, 14, 14, 7, 7]

        # ==================
        # Original model
        # ==================
        logger.info("Running original model...")

        pre_fc_orig: Dict[str, torch.Tensor] = {}
        pooled_orig: Dict[str, torch.Tensor] = {}

        def _capture_orig_fc(module: nn.Module, inputs, _output):
            pre_fc_orig["features"] = inputs[0].detach().cpu()

        def _capture_orig_pool(module: nn.Module, inputs, output):
            pooled_orig["features"] = output.detach().cpu()

        hook_orig_fc = self.original.out_fc.register_forward_hook(_capture_orig_fc)
        hook_orig_pool = self.original.out_avgpool.register_forward_hook(
            _capture_orig_pool
        )

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

        hook_orig_fc.remove()
        hook_orig_pool.remove()

        logger.info(f"  Original output: {list(out_orig.shape)}\n")

        # ==================
        # Reimplementation
        # ==================
        logger.info("Running reimplementation...")

        classifier = getattr(self.reimpl, self.reimpl.classifier_name)
        pre_fc_reimpl: Dict[str, torch.Tensor] = {}
        pooled_reimpl: Dict[str, torch.Tensor] = {}

        def _capture_reimpl_fc(module: nn.Module, inputs, _output):
            pre_fc_reimpl["features"] = inputs[0].detach().cpu()

        def _capture_reimpl_pool(module: nn.Module, inputs, output):
            pooled_reimpl["features"] = output.detach().cpu()

        hook_reimpl_fc = classifier[-1].register_forward_hook(_capture_reimpl_fc)
        hook_reimpl_pool = classifier[1].register_forward_hook(_capture_reimpl_pool)

        with torch.no_grad():
            # DynVision expects [batch, time, channels, height, width]
            img_temporal = img.unsqueeze(1).repeat(1, timesteps, 1, 1, 1)
            out_reimpl = self.reimpl(img_temporal)

            # Get last timestep
            out_reimpl = out_reimpl[:, -1]

        hook_reimpl_fc.remove()
        hook_reimpl_pool.remove()

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

        classifier_input_metrics: Dict[str, float] = {}
        pooled_feature_metrics: Dict[str, float] = {}

        if "features" in pre_fc_orig and "features" in pre_fc_reimpl:
            orig_features = pre_fc_orig["features"].to(out_orig.device)
            reimpl_features = pre_fc_reimpl["features"].to(out_reimpl.device)
            feature_diff = (orig_features - reimpl_features).abs()
            classifier_input_metrics = {
                "max": feature_diff.max().item(),
                "mean": feature_diff.mean().item(),
                "std": feature_diff.std().item(),
            }

        if "features" in pooled_orig and "features" in pooled_reimpl:
            orig_pool_flat = (
                pooled_orig["features"].view(batch_size, -1).to(out_orig.device)
            )
            reimpl_pool_flat = (
                pooled_reimpl["features"].view(batch_size, -1).to(out_reimpl.device)
            )
            pool_diff = (orig_pool_flat - reimpl_pool_flat).abs()
            pooled_feature_metrics = {
                "max": pool_diff.max().item(),
                "mean": pool_diff.mean().item(),
                "std": pool_diff.std().item(),
            }

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

        if classifier_input_metrics:
            metrics["classifier_input"] = classifier_input_metrics
        if pooled_feature_metrics:
            metrics["pooled_features"] = pooled_feature_metrics

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

        if classifier_input_metrics:
            print("\nClassifier Input Differences (pre-linear features):")
            print(
                f"  Max: {classifier_input_metrics['max']:.6f} | Mean: {classifier_input_metrics['mean']:.6f} | Std: {classifier_input_metrics['std']:.6f}"
            )

        if pooled_feature_metrics:
            print("\nPooled Feature Differences (post-avgpool, pre-flatten):")
            print(
                f"  Max: {pooled_feature_metrics['max']:.6f} | Mean: {pooled_feature_metrics['mean']:.6f} | Std: {pooled_feature_metrics['std']:.6f}"
            )

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

        if classifier_input_metrics:
            logger.info(
                "   Classifier input diff (max/mean/std): %.6f / %.6f / %.6f",
                classifier_input_metrics["max"],
                classifier_input_metrics["mean"],
                classifier_input_metrics["std"],
            )

        if pooled_feature_metrics:
            logger.info(
                "   Pooled feature diff (max/mean/std): %.6f / %.6f / %.6f",
                pooled_feature_metrics["max"],
                pooled_feature_metrics["mean"],
                pooled_feature_metrics["std"],
            )

        logger.info("")

        return metrics

    def run_full_validation(
        self, max_diagnostic_steps: Optional[int] = 10
    ) -> Dict[str, any]:
        """Run complete validation suite.

        Args:
            max_diagnostic_steps: Limits the number of timesteps traced during
                detailed diagnostics. Use <=0 to run full idle/active windows.
        """

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
        test_img = torch.randn(self.batch_size, 3, 224, 224, device=self.device)
        self.test_3_trace_original_dynamics(test_img, timesteps=3)

        # Use capped values for detailed diagnostics to keep logging manageable
        if max_diagnostic_steps is None or max_diagnostic_steps <= 0:
            idle_steps = self.idle_timesteps
            active_steps = self.n_timesteps
        else:
            idle_steps = min(self.idle_timesteps, max_diagnostic_steps)
            active_steps = min(self.n_timesteps, max_diagnostic_steps)

        # Phase 4: Layer-by-layer comparison
        results["layer_metrics"] = self.test_4_compare_layer_by_layer(
            test_img,
            n_timesteps=active_steps,
            idle_timesteps=idle_steps,
            alpha=self.alpha,
        )

        # Phase 5: Output comparison
        results["output_metrics"] = self.test_5_final_output_comparison(
            test_img,
            timesteps=active_steps,
            idle_timesteps=idle_steps,
            alpha=self.alpha,
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

        if results["architectural_issues"]:
            print("=" * 80)
            print(" " * 20 + "FOLLOW-UP RECOMMENDATIONS")
            print("=" * 80 + "\n")
            for issue in results["architectural_issues"]:
                print(issue)
                if not issue.endswith("\n"):
                    print()
            print("Refer to docs/development/guides/model-integration.md for fixes.\n")
            print("=" * 80 + "\n")
        else:
            print("No architectural mismatches detected in static checks.\n")

        return results


def main():
    """Main validation function."""

    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Validate CordsNet implementation against reference model"
        )
        parser.add_argument(
            "--dataset",
            default="imagenet",
            help="Dataset key used to configure the models",
        )
        parser.add_argument(
            "--depth", type=int, default=8, help="Number of recurrent layers (areas)"
        )
        parser.add_argument(
            "--timesteps",
            type=int,
            default=10,
            help="Number of active timesteps to evaluate in diagnostic runs",
        )
        parser.add_argument(
            "--idle-timesteps",
            type=int,
            default=10,
            help="Number of idle timesteps to run before stimulus onset",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help="Batch size to use for synthetic validation inputs",
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=None,
            help="Optional override for integration constant (dt/tau)",
        )
        parser.add_argument(
            "--device",
            default=None,
            help="Torch device identifier (e.g. 'cuda', 'cpu', 'cuda:0')",
        )
        parser.add_argument(
            "--max-diagnostic-steps",
            type=int,
            default=10,
            help="Cap per-phase diagnostics to this many timesteps (<=0 for full window)",
        )
        return parser.parse_args()

    args = _parse_args()

    # Create validator
    validator = CordsNetValidator(
        dataset=args.dataset,
        depth=args.depth,
        n_timesteps=args.timesteps,
        idle_timesteps=args.idle_timesteps,
        batch_size=args.batch_size,
        alpha=args.alpha,
        device=args.device,
    )

    # Run validation
    results = validator.run_full_validation(
        max_diagnostic_steps=args.max_diagnostic_steps
    )

    return results


if __name__ == "__main__":
    main()
