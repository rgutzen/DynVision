import torch
import argparse
from pathlib import Path
import re

parser = argparse.ArgumentParser(
    description="Convert PyTorch Lightning checkpoint to state_dict .pt file"
)
parser.add_argument(
    "--checkpoint_dir", type=Path, help="Path to checkpoint directory", required=True
)
parser.add_argument(
    "--output", type=Path, help="Path to output .pt file", required=True
)


def get_best_checkpoint(checkpoint_dir, model_identifier):
    pattern = re.compile(rf"{re.escape(model_identifier)}.*val_loss=(\d+\.\d+)")
    best_loss = float("inf")
    best_checkpoint = None

    for checkpoint_path in checkpoint_dir.glob("*.ckpt"):
        match = pattern.search(checkpoint_path.name)
        if match:
            val_loss = float(match.group(1))
            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint = checkpoint_path

    if best_checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint files found for model identifier: {model_identifier}"
        )

    return best_checkpoint


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    model_identifier = args.output.stem

    best_checkpoint = get_best_checkpoint(
        args.checkpoint_dir, model_identifier=model_identifier
    )

    checkpoint = torch.load(
        best_checkpoint, map_location=torch.device("cpu"), weights_only=True
    )

    output_path = args.output.with_suffix(".pt")

    if output_path.exists():
        print(f"The file {output_path} already exists.")
    else:
        torch.save(checkpoint["state_dict"], output_path)
        print(f"Model state saved to {output_path}")

    args.output.touch()
