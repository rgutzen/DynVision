"""Checkpoint conversion utility for DynVision models.

This module provides functionality to:
- Convert PyTorch Lightning checkpoints to state dictionaries
- Select best checkpoint based on validation loss
- Handle checkpoint conversion and saving
- Validate checkpoint compatibility

Usage:
    python checkpoint_to_statedict.py \
        --checkpoint_dir checkpoints/ \
        --output models/model.pt
"""

import logging
import re
from pathlib import Path
from typing import Optional, Union

import torch
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Convert PyTorch Lightning checkpoint to state_dict .pt file",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--checkpoint_dir", type=Path, help="Path to checkpoint directory", required=True
)
parser.add_argument(
    "--output", type=Path, help="Path to output .pt file", required=True
)


def get_best_checkpoint(
    checkpoint_dir: Path, model_identifier: str, pattern: Optional[str] = None
) -> Path:
    """Find best checkpoint based on validation loss.

    Args:
        checkpoint_dir: Directory containing checkpoints
        model_identifier: Model name to match
        pattern: Custom regex pattern for matching

    Returns:
        Path to best checkpoint

    Raises:
        FileNotFoundError: If no matching checkpoints found
        ValueError: If checkpoint directory does not exist
    """
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    logger.info(f"Searching for checkpoints in: {checkpoint_dir}")
    logger.info(f"Model identifier: {model_identifier}")

    # Updated pattern: model_identifier-<epch>-<loss>.ckpt
    # Example: mymodel-12-0.34.ckpt
    regex = pattern or rf"{re.escape(model_identifier)}-(\d+)-(\d+\.\d{{2}})\.ckpt"
    compiled_pattern = re.compile(regex)
    best_loss = float("inf")
    best_checkpoint = None

    # List all .ckpt files for debugging
    ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
    logger.info(f"Found {len(ckpt_files)} .ckpt files")

    for checkpoint_path in ckpt_files:
        logger.debug(f"Checking: {checkpoint_path.name}")
        match = compiled_pattern.fullmatch(checkpoint_path.name)
        if match:
            epch = int(match.group(1))
            loss = float(match.group(2))
            logger.info(
                f"Matched checkpoint: {checkpoint_path.name} (epoch {epch}, loss {loss:.4f})"
            )
            if loss < best_loss:
                best_loss = loss
                best_checkpoint = checkpoint_path
                logger.info(f"New best checkpoint found with loss {loss:.4f}")
        else:
            logger.debug(f"No match for: {checkpoint_path.name}")

    if best_checkpoint is None:
        available_files = [f.name for f in ckpt_files]
        raise FileNotFoundError(
            f"No checkpoint files found for model identifier: {model_identifier}\n"
            f"Available files: {available_files}\n"
            f"Pattern used: {regex}"
        )

    logger.info(f"Selected best checkpoint: {best_checkpoint}")
    logger.info(f"Best loss: {best_loss:.4f}")
    return best_checkpoint


def validate_checkpoint_file(checkpoint_path: Path) -> bool:
    """Validate that checkpoint file is not empty and can be loaded.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        True if file is valid, False otherwise
    """
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file does not exist: {checkpoint_path}")
        return False

    if checkpoint_path.stat().st_size == 0:
        logger.error(f"Checkpoint file is empty: {checkpoint_path}")
        return False

    try:
        # Try to load just the keys to verify file integrity
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint, dict):
            logger.error(f"Checkpoint is not a dictionary: {type(checkpoint)}")
            return False
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        return True
    except Exception as e:
        logger.error(f"Failed to validate checkpoint file: {e}")
        return False


def save_state_dict(
    checkpoint_path: Path, output_path: Path, device: Union[str, torch.device] = "cpu"
) -> None:
    """Save checkpoint state dictionary.

    Args:
        checkpoint_path: Path to checkpoint file
        output_path: Path to save state dictionary
        device: Device to load checkpoint on

    Raises:
        FileNotFoundError: If checkpoint file does not exist
        KeyError: If state dict not found in checkpoint
        RuntimeError: If checkpoint loading fails
    """
    if not validate_checkpoint_file(checkpoint_path):
        raise RuntimeError(f"Checkpoint validation failed: {checkpoint_path}")

    try:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    if "state_dict" not in checkpoint:
        available_keys = list(checkpoint.keys())
        raise KeyError(
            f"No state_dict found in checkpoint. Available keys: {available_keys}"
        )

    state_dict = checkpoint["state_dict"]
    logger.info(f"State dict contains {len(state_dict)} parameters")

    # Validate state dict is not empty
    if not state_dict:
        raise ValueError("State dict is empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with error handling
    try:
        torch.save(state_dict, output_path)
        logger.info(f"Model state saved to {output_path}")

        # Verify the saved file
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Output file was not created properly: {output_path}")

        # Try to load it back to verify integrity
        torch.load(output_path, map_location="cpu", weights_only=True)
        logger.info(f"Verified saved file integrity: {output_path}")

    except Exception as e:
        # Clean up failed output
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"Failed to save state dict: {e}")


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    model_identifier = args.output.stem
    output_path = args.output.with_suffix(".pt")

    try:
        logger.info(f"Converting checkpoint to state dict")
        logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
        logger.info(f"Model identifier: {model_identifier}")
        logger.info(f"Output path: {output_path}")

        # Find best checkpoint
        best_checkpoint = get_best_checkpoint(
            args.checkpoint_dir, model_identifier=model_identifier
        )

        # Save state dictionary if output doesn't exist or is invalid
        if output_path.exists():
            logger.warning(f"Output file already exists: {output_path}")
            # Check if existing file is valid
            try:
                torch.load(output_path, map_location="cpu", weights_only=True)
                logger.info("Existing output file is valid, skipping conversion")
            except Exception as e:
                logger.warning(f"Existing output file is invalid ({e}), recreating...")
                output_path.unlink()
                save_state_dict(best_checkpoint, output_path)
        else:
            save_state_dict(best_checkpoint, output_path)

        # Only touch the marker file if conversion succeeded
        if output_path.exists() and output_path.stat().st_size > 0:
            args.output.touch()
            logger.info(f"Conversion completed successfully")
        else:
            raise RuntimeError("Conversion failed - output file is missing or empty")

    except Exception as e:
        logger.error(f"Failed to convert checkpoint: {e}")
        # Clean up any partial files
        if output_path.exists():
            output_path.unlink()
        if args.output.exists():
            args.output.unlink()
        raise
