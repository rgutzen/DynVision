"""Checkpoint conversion utility for DynVision models.

This module provides functionality to:
- Convert PyTorch Lightning checkpoints to state dictionaries
- Select best checkpoint based on validation loss
- Batch-convert checkpoint files or wildcard patterns to state dictionaries
- Handle checkpoint conversion and saving
- Validate checkpoint compatibility

Usage examples:
    # Convert best checkpoint in a directory
    python checkpoint_to_statedict.py --checkpoint_dir checkpoints/ --output model.pt

    # Convert multiple checkpoints into an output directory
    python checkpoint_to_statedict.py --checkpoint_dir checkpoints/ \
        --checkpoint_globs "*.ckpt" --output_dir exported_state_dicts/
"""

import logging
import re
import glob
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

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
    "--checkpoint_dir",
    type=Path,
    default=Path.cwd(),
    help="Directory containing checkpoints or base directory for wildcard expansion",
)
parser.add_argument(
    "--output",
    type=Path,
    help="Path to single output .pt file (required when converting best checkpoint)",
)
parser.add_argument(
    "--checkpoint_globs",
    nargs="+",
    help="One or more wildcard patterns (relative to checkpoint_dir) to convert",
)
parser.add_argument(
    "--checkpoint_files",
    nargs="+",
    type=Path,
    help="Explicit checkpoint file paths to convert",
)
parser.add_argument(
    "--output_dir",
    type=Path,
    help="Directory to store converted state dicts when processing multiple checkpoints",
)
parser.add_argument(
    "--continue_on_error",
    action="store_true",
    help="Continue converting remaining checkpoints even if one conversion fails",
)


def get_best_checkpoint(
    checkpoint_dir: Path,
    model_identifier: str,
    pattern: Optional[str] = None,
    raise_error=True,
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

    # Updated pattern: model_identifier-best-<epch>-<loss>.ckpt
    # Example: mymodel-best-12-0.34.ckpt
    regex = (
        pattern or rf"{re.escape(model_identifier)}-best-(\d+)-(\d+\.\d{{2}})\.ckpt"
    )

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

    if best_checkpoint is None and raise_error:
        available_files = [f.name for f in ckpt_files]
        raise FileNotFoundError(
            f"No checkpoint files found for model identifier: {model_identifier}\n"
            f"Available files: {available_files}\n"
            f"Pattern used: {regex}"
        )

    logger.info(f"Selected best checkpoint: {best_checkpoint}")
    logger.info(f"Best loss: {best_loss:.4f}")
    return best_checkpoint


def extract_epoch_from_filename(filename: str) -> Optional[int]:
    """Attempt to extract epoch information from a checkpoint filename."""

    patterns = [
        re.compile(r"-(\d+)-(?:nan|inf)\.ckpt$", re.IGNORECASE),
        re.compile(r"-(?:best|epoch|last)-(\d+)-"),
        re.compile(r"epoch=(\d+)", re.IGNORECASE),
        re.compile(r"-(\d+)\.ckpt$"),
        re.compile(r"-(\d+)-"),
    ]

    for pattern in patterns:
        match = pattern.search(filename)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def derive_output_basename(stem: str) -> str:
    """Derive a clean base name for the exported state dict."""

    lower_stem = stem.lower()
    for token in ("-best-", "-epoch-", "-last-"):
        idx = lower_stem.find(token)
        if idx != -1:
            cleaned = stem[:idx]
            return cleaned.rstrip("-_") or cleaned

    stripped = re.sub(r"-\d+(?:-(?:nan|inf))?$", "", stem, flags=re.IGNORECASE)
    stripped = stripped.rstrip("-_")
    return stripped if stripped else stem


def _expand_pattern(base_dir: Path, pattern: str) -> List[Path]:
    if Path(pattern).is_absolute():
        matches = glob.glob(pattern)
    else:
        matches = glob.glob(str(base_dir / pattern))
    return [Path(match) for match in matches]


def resolve_checkpoint_paths(
    checkpoint_dir: Path,
    patterns: Optional[Sequence[str]] = None,
    files: Optional[Sequence[Path]] = None,
) -> List[Path]:
    """Resolve checkpoint paths from explicit files and glob patterns."""

    resolved: List[Path] = []
    seen = set()
    base_dir = checkpoint_dir if checkpoint_dir is not None else Path.cwd()

    if patterns:
        for pattern in patterns:
            for candidate in _expand_pattern(base_dir, pattern):
                if candidate.exists() and candidate.suffix == ".ckpt":
                    key = candidate.resolve()
                    if key not in seen:
                        resolved.append(key)
                        seen.add(key)

    if files:
        for file_path in files:
            candidate = file_path
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            if candidate.exists() and candidate.suffix == ".ckpt":
                key = candidate.resolve()
                if key not in seen:
                    resolved.append(key)
                    seen.add(key)
            else:
                logger.warning(
                    "Checkpoint file not found or invalid suffix: %s", candidate
                )

    return sorted(resolved)


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


def convert_checkpoints_to_directory(
    checkpoints: Sequence[Path],
    output_dir: Path,
    device: Union[str, torch.device] = "cpu",
    continue_on_error: bool = False,
) -> Tuple[int, List[Tuple[Path, Exception]]]:
    """Convert a sequence of checkpoints into state dicts within an output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    failures: List[Tuple[Path, Exception]] = []
    converted = 0

    for checkpoint_path in checkpoints:
        epoch = extract_epoch_from_filename(checkpoint_path.name)

        if epoch == 0:
            logger.info(
                "Skipping checkpoint with epoch 0 (likely incomplete): %s",
                checkpoint_path,
            )
            continue

        epoch_str = str(epoch) if epoch is not None else "unknown"
        base_name = derive_output_basename(checkpoint_path.stem)
        output_path = output_dir / f"{base_name}-epoch={epoch_str}.pt"

        if output_path.exists():
            try:
                torch.load(output_path, map_location="cpu", weights_only=True)
                logger.info(
                    "Existing state dict is valid, skipping conversion: %s",
                    output_path,
                )
                converted += 1
                continue
            except Exception:
                logger.warning(
                    "Existing state dict is invalid, regenerating: %s", output_path
                )
                output_path.unlink(missing_ok=True)

        try:
            save_state_dict(checkpoint_path, output_path, device=device)
            converted += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to convert %s: %s", checkpoint_path, exc)
            failures.append((checkpoint_path, exc))
            if not continue_on_error:
                raise

    return converted, failures


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    if unknown:
        logger.warning("Ignoring unknown arguments: %s", unknown)

    is_multi_mode = any(
        [args.checkpoint_globs, args.checkpoint_files, args.output_dir]
    )

    if is_multi_mode:
        if args.output is not None:
            parser.error(
                "--output cannot be used together with multi-checkpoint options"
            )
        if args.output_dir is None:
            parser.error(
                "--output_dir is required when converting multiple checkpoints"
            )

        checkpoints = resolve_checkpoint_paths(
            args.checkpoint_dir,
            patterns=args.checkpoint_globs,
            files=args.checkpoint_files,
        )

        if not checkpoints:
            raise FileNotFoundError(
                "No checkpoints matched the provided patterns/files"
            )

        logger.info("Found %d checkpoints to convert", len(checkpoints))
        converted, failures = convert_checkpoints_to_directory(
            checkpoints,
            args.output_dir,
            continue_on_error=args.continue_on_error,
        )

        if failures and not args.continue_on_error:
            raise RuntimeError("Conversion aborted due to previous error")

        if failures:
            logger.warning(
                "Completed conversion with %d failures. See logs for details.",
                len(failures),
            )
        logger.info("Successfully converted %d checkpoints", converted)
    else:
        if args.output is None:
            parser.error(
                "--output is required when not using multi-checkpoint options"
            )

        model_identifier = args.output.name.split("#", 1)[0]
        output_path = (
            args.output if args.output.suffix else args.output.with_suffix(".pt")
        )

        checkpoint_dir = args.checkpoint_dir
        if checkpoint_dir is None:
            parser.error(
                "--checkpoint_dir must be provided for single conversion mode"
            )

        try:
            logger.info("Converting best checkpoint to state dict")
            logger.info("Checkpoint dir: %s", checkpoint_dir)
            logger.info("Model identifier: %s", model_identifier)
            logger.info("Output path: %s", output_path)

            best_checkpoint = get_best_checkpoint(
                checkpoint_dir, model_identifier=model_identifier
            )

            if output_path.exists():
                logger.warning("Output file already exists: %s", output_path)
                try:
                    torch.load(output_path, map_location="cpu", weights_only=True)
                    logger.info("Existing output file is valid, skipping conversion")
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Existing output file is invalid (%s), recreating...", e
                    )
                    output_path.unlink()
                    save_state_dict(best_checkpoint, output_path)
            else:
                save_state_dict(best_checkpoint, output_path)

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError(
                    "Conversion failed - output file is missing or empty"
                )

            logger.info("Conversion completed successfully")

        except Exception as e:  # noqa: BLE001
            logger.error("Failed to convert checkpoint: %s", e)
            if "output_path" in locals() and output_path.exists():
                output_path.unlink()
            raise
