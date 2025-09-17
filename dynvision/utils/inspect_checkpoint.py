#!/usr/bin/env python3
"""
Checkpoint Inspector for PyTorch Lightning files
Usage: python inspect_checkpoint.py path/to/checkpoint.ckpt
"""

import torch
import sys
from pathlib import Path
import json
from datetime import datetime


def inspect_checkpoint(checkpoint_path):
    """Extract and display key information from a Lightning checkpoint."""
    try:
        # Load checkpoint (map to CPU to avoid GPU memory issues)
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        print(f"\n{'='*60}")
        print(f"CHECKPOINT INSPECTION: {Path(checkpoint_path).name}")
        print(f"{'='*60}")

        # Basic training info
        print(f"🔹 Epoch: {ckpt.get('epoch', 'Unknown')}")
        print(f"🔹 Global Step: {ckpt.get('global_step', 'Unknown')}")
        print(
            f"🔹 PyTorch Lightning Version: {ckpt.get('pytorch-lightning_version', 'Unknown')}"
        )

        # File info
        file_size = Path(checkpoint_path).stat().st_size / (1024**2)  # MB
        mod_time = datetime.fromtimestamp(Path(checkpoint_path).stat().st_mtime)
        print(f"🔹 File Size: {file_size:.1f} MB")
        print(f"🔹 Last Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Hyperparameters (this is where DynVision stores model config)
        if "hyper_parameters" in ckpt:
            hparams = ckpt["hyper_parameters"]
            print(f"\n📊 MODEL CONFIGURATION:")
            print(f"   Model Type: {hparams.get('__class__', 'Unknown')}")

            # DynVision specific parameters
            key_params = [
                "n_classes",
                "n_timesteps",
                "learning_rate",
                "dt",
                "tau",
                "t_feedforward",
                "t_recurrence",
                "recurrence_type",
                "input_dims",
                "optimizer",
                "loss",
                "batch_size",
            ]

            for param in key_params:
                if param in hparams:
                    print(f"   {param}: {hparams[param]}")

            # Check for custom model parameters
            model_specific = [
                k
                for k in hparams.keys()
                if not k.startswith("_") and k not in key_params
            ]
            if model_specific:
                print(
                    f"   Other params: {', '.join(model_specific[:5])}{'...' if len(model_specific) > 5 else ''}"
                )

        # Training metrics (if available)
        if "lr_schedulers" in ckpt:
            print(f"\n📈 TRAINING STATE:")
            print(f"   Learning Rate Schedulers: {len(ckpt['lr_schedulers'])}")

        if "optimizer_states" in ckpt:
            print(f"   Optimizers: {len(ckpt['optimizer_states'])}")

        # Model architecture info
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            total_params = sum(
                p.numel() for p in state_dict.values() if hasattr(p, "numel")
            )
            layer_count = len([k for k in state_dict.keys() if "weight" in k])
            print(f"\n🧠 MODEL ARCHITECTURE:")
            print(f"   Total Parameters: {total_params:,}")
            print(f"   Weight Layers: {layer_count}")

            # Show some key layer names to identify architecture
            key_layers = [
                k
                for k in state_dict.keys()
                if any(
                    layer in k.lower() for layer in ["conv", "linear", "classifier"]
                )
            ][:5]
            if key_layers:
                print(
                    f"   Key Layers: {', '.join([k.split('.')[0] for k in key_layers])}"
                )

        # Look for custom DynVision fields
        custom_fields = [
            k
            for k in ckpt.keys()
            if k
            not in [
                "state_dict",
                "optimizer_states",
                "lr_schedulers",
                "epoch",
                "global_step",
                "pytorch-lightning_version",
                "hyper_parameters",
            ]
        ]
        if custom_fields:
            print(f"\n🔧 ADDITIONAL DATA:")
            for field in custom_fields[:5]:
                print(f"   {field}: {type(ckpt[field])}")

        print(f"\n{'='*60}\n")

    except Exception as e:
        print(f"❌ Error inspecting {checkpoint_path}: {e}")


def inspect_multiple_checkpoints(checkpoint_dir):
    """Inspect all checkpoint files in a directory."""
    checkpoint_dir = Path(checkpoint_dir)

    # Find all checkpoint files
    ckpt_files = list(checkpoint_dir.glob("*.ckpt")) + list(
        checkpoint_dir.glob("**/last*.ckpt")
    )

    if not ckpt_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return

    print(f"Found {len(ckpt_files)} checkpoint files:")

    # Sort by modification time (newest first)
    ckpt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    for ckpt_file in ckpt_files:
        inspect_checkpoint(ckpt_file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_checkpoint.py <checkpoint_path_or_directory>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_file():
        inspect_checkpoint(path)
    elif path.is_dir():
        inspect_multiple_checkpoints(path)
    else:
        print(f"Path {path} does not exist")
