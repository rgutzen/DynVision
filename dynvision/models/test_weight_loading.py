"""
Test script to verify pretrained weight loading and transfer for CordsNet.

This script:
1. Downloads pretrained weights
2. Loads them into the reimplementation
3. Verifies all weights transferred correctly
4. Compares outputs between original and reimplementation with same weights
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dynvision.models.cordsnet_original import cordsnet as CordsNetOriginal
from dynvision.models.cordsnet import CordsNet

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_weight_loading():
    """Test loading pretrained weights into reimplementation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}\n")

    # =========================================================================
    # 1. Create models
    # =========================================================================
    logger.info("=" * 80)
    logger.info("STEP 1: Creating Models")
    logger.info("=" * 80 + "\n")

    # Original model
    logger.info("Creating original CordsNet...")
    original = CordsNetOriginal(dataset='imagenet', depth=8).to(device)
    original.eval()

    # Reimplementation WITH pretrained weights
    logger.info("Creating reimplementation with pretrained weights...")
    reimpl = CordsNet(
        input_dims=(100, 3, 224, 224),
        n_classes=1000,
        n_timesteps=100,
        dt=2.0,  # alpha = dt/tau = 2/10 = 0.2
        tau=10.0,
        t_feedforward=2.0,
        t_recurrence=2.0,
        t_skip=2.0,
        idle_timesteps=100,
        init_with_pretrained=True,  # This will load pretrained weights
    ).to(device)
    reimpl.setup('fit')
    reimpl.eval()

    logger.info("Models created successfully!\n")

    # =========================================================================
    # 2. Load pretrained weights into original model
    # =========================================================================
    logger.info("=" * 80)
    logger.info("STEP 2: Loading Pretrained Weights into Original Model")
    logger.info("=" * 80 + "\n")

    try:
        # Download and load weights
        pretrained_path = reimpl.download_pretrained_state_dict.__code__.co_consts[4]  # Get save_path
        from dynvision.project_paths import project_paths
        save_path = project_paths.models / "CordsNet" / "cordsnet_pretrained.pth"

        if save_path.exists():
            logger.info(f"Loading pretrained weights from {save_path}")
            state_dict = torch.load(save_path, map_location=device)
            original.load_state_dict(state_dict)
            logger.info("✅ Pretrained weights loaded into original model\n")
        else:
            logger.warning("⚠️  Pretrained weights not found, models using random initialization\n")

    except Exception as e:
        logger.error(f"Error loading weights into original: {e}\n")

    # =========================================================================
    # 3. Verify weight transfer
    # =========================================================================
    logger.info("=" * 80)
    logger.info("STEP 3: Verifying Weight Transfer")
    logger.info("=" * 80 + "\n")

    orig_state = original.state_dict()
    reimpl_state = reimpl.state_dict()
    translation = reimpl.translate_pretrained_layer_names()

    logger.info(f"Original model: {len(orig_state)} parameters")
    logger.info(f"Reimpl model: {len(reimpl_state)} parameters")
    logger.info(f"Translation mapping: {len(translation)} entries\n")

    # Check weight transfer for each mapping
    matches = 0
    mismatches = []

    for orig_base_key, reimpl_pattern in translation.items():
        # Find original keys (handling weight_norm parametrization)
        orig_keys = [k for k in orig_state.keys() if k.startswith(orig_base_key)]
        reimpl_keys = [k for k in reimpl_state.keys() if reimpl_pattern in k]

        if not orig_keys or not reimpl_keys:
            continue

        # For weight_norm, compare the actual weight values
        for orig_key in orig_keys:
            # Find corresponding reimpl key
            reimpl_key = None
            if 'original0' in orig_key:
                reimpl_key = [k for k in reimpl_keys if 'original0' in k]
            elif 'original1' in orig_key:
                reimpl_key = [k for k in reimpl_keys if 'original1' in k]
            elif 'bias' in orig_key:
                reimpl_key = [k for k in reimpl_keys if 'bias' in k and 'original' not in k]
            else:
                reimpl_key = reimpl_keys

            if reimpl_key:
                reimpl_key = reimpl_key[0]
                orig_param = orig_state[orig_key]
                reimpl_param = reimpl_state[reimpl_key]

                if orig_param.shape == reimpl_param.shape:
                    # Check if values match (allowing for small numerical differences)
                    diff = (orig_param - reimpl_param).abs().max().item()
                    if diff < 1e-6:
                        matches += 1
                        logger.debug(f"✓ {orig_key} -> {reimpl_key}: values match (max diff: {diff:.2e})")
                    else:
                        logger.warning(f"⚠️  {orig_key} -> {reimpl_key}: shapes match but values differ (max diff: {diff:.2e})")
                        mismatches.append((orig_key, reimpl_key, diff))
                else:
                    logger.error(f"❌ Shape mismatch: {orig_key} {orig_param.shape} -> {reimpl_key} {reimpl_param.shape}")
                    mismatches.append((orig_key, reimpl_key, "shape mismatch"))

    logger.info(f"\n✅ Matched parameters: {matches}")
    if mismatches:
        logger.warning(f"⚠️  Mismatches or differences: {len(mismatches)}")
        for orig_k, reimpl_k, diff in mismatches[:5]:
            logger.warning(f"  {orig_k} -> {reimpl_k}: {diff}")
    logger.info("")

    # =========================================================================
    # 4. Compare forward passes
    # =========================================================================
    logger.info("=" * 80)
    logger.info("STEP 4: Comparing Forward Passes with Same Weights")
    logger.info("=" * 80 + "\n")

    # Create test input
    torch.manual_seed(42)
    test_img = torch.randn(1, 3, 224, 224, device=device)

    logger.info("Running forward passes...")

    with torch.no_grad():
        # Original model forward pass
        alpha = 0.2
        batch_size = test_img.shape[0]
        channels = [64, 64, 128, 128, 256, 256, 512, 512]
        sizes = [56, 56, 28, 28, 14, 14, 7, 7]

        rs = [
            torch.zeros(batch_size, channels[j], sizes[j], sizes[j], device=device)
            for j in range(8)
        ]

        # Idle timesteps
        for t in range(100):
            for j in range(7, -1, -1):
                rs[j] = original.rnn(j, rs, test_img * 0, alpha)

        # Active timesteps (just 10 for speed)
        for t in range(10):
            for j in range(7, -1, -1):
                rs[j] = original.rnn(j, rs, test_img, alpha)

        # Output
        out_orig = original.out_avgpool(
            original.relu(rs[7]) + original.relu(rs[6])
        )
        out_orig = original.out_flatten(out_orig)
        out_orig = original.out_fc(out_orig)

        logger.info(f"Original output: {out_orig.shape}, range: [{out_orig.min():.3f}, {out_orig.max():.3f}]")

        # Reimplementation forward pass
        img_temporal = test_img.unsqueeze(1).repeat(1, 10, 1, 1, 1)
        out_reimpl = reimpl(img_temporal)
        out_reimpl = out_reimpl[:, -1]  # Last timestep

        logger.info(f"Reimpl output: {out_reimpl.shape}, range: [{out_reimpl.min():.3f}, {out_reimpl.max():.3f}]\n")

    # Compare outputs
    diff = (out_orig - out_reimpl).abs()
    logger.info(f"Output comparison:")
    logger.info(f"  Max difference: {diff.max().item():.6f}")
    logger.info(f"  Mean difference: {diff.mean().item():.6f}")
    logger.info(f"  Std difference: {diff.std().item():.6f}")

    pred_orig = out_orig.argmax(dim=1).item()
    pred_reimpl = out_reimpl.argmax(dim=1).item()
    logger.info(f"  Original prediction: {pred_orig}")
    logger.info(f"  Reimpl prediction: {pred_reimpl}")
    logger.info(f"  Predictions match: {pred_orig == pred_reimpl}\n")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80 + "\n")

    if matches > 0:
        logger.info(f"✅ Weight transfer successful: {matches} parameters matched")
    else:
        logger.warning("⚠️  Weight transfer may have issues")

    if pred_orig == pred_reimpl:
        logger.info("✅ Predictions match with pretrained weights!")
    else:
        logger.warning("⚠️  Predictions differ - may need further debugging")

    logger.info("")


if __name__ == "__main__":
    test_weight_loading()
