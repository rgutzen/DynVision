#!/usr/bin/env python3
"""Count trainable parameters for each model variation in the benchmarking table.

Instantiates DyRCNNx8 on CPU with each parameter variation and counts
trainable parameters.  Includes skip/feedback 1×1-convs created during setup().
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

import torch


# -- parameter variations (matching the benchmarking table) -------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "n_timesteps": 20,
    "dt": 2,
    "tau": 5,
    "t_feedforward": 0,
    "t_recurrence": 6,
    "t_skip": 0,
    "t_feedback": 30,
    "recurrence_type": "full",
    "recurrence_target": "output",
    "skip": True,
    "feedback": False,
    "feedforward_only": False,
    "data_presentation_pattern": "1",
    "n_classes": 10,
    "n_channels": 3,
    "dim_y": 224,
    "dim_x": 224,
    "use_retina": False,
    "train_tau": False,
    "bias": True,
    "max_weight_init": 0.001,
}

VARIATIONS: List[Tuple[str, str, List[Any]]] = [
    ("Recurrence Type", "recurrence_type", ["self", "full", "depthpointwise", "pointdepthwise", "local", "localdepthwise", "none"]),
    ("Recurrence Target", "recurrence_target", ["input", "middle", "output"]),
    ("Skip", "skip", [True, False]),
    ("Feedback", "feedback", ["false", "additive", "multiplicative"]),
    ("Timesteps", "n_timesteps", [8, 14, 20, 26]),
    ("Presentation Pattern", "data_presentation_pattern", ["1", "1011", "1001", "1000"]),
    ("Loss Reaction Time", "loss_reaction_time", [0, 4, 8, 18, 28]),
    ("EnergyLoss Weight", "activity_loss_weight", [0, 0.0002, 0.02, 0.2, 1.0]),
    ("Idle Timesteps", "idle_timesteps", [0, 1, 5, 10, 20]),
]

# Parameters that do NOT affect model architecture (delta always 0).
ARCH_NEUTRAL_KEYS = {
    "n_timesteps", "loss_reaction_time", "activity_loss_weight",
    "idle_timesteps", "data_presentation_pattern",
}


def _instantiate(overrides: Dict[str, Any]):
    """Create DyRCNNx8 with given overrides."""
    from dynvision.models.dyrcnn import DyRCNNx8

    config = dict(DEFAULT_CONFIG)
    config.update(overrides)

    model_kwargs = dict(
        recurrence_type=config["recurrence_type"],
        recurrence_target=config["recurrence_target"],
        n_timesteps=config["n_timesteps"],
        skip=config["skip"],
        feedback=config["feedback"],
        feedback_mode=config.get("feedback_mode", "additive"),
        dt=config["dt"],
        tau=config["tau"],
        t_feedforward=config["t_feedforward"],
        t_recurrence=config["t_recurrence"],
        t_skip=config["t_skip"],
        t_feedback=config.get("t_feedback", 30),
        feedforward_only=config["feedforward_only"],
        data_presentation_pattern=config["data_presentation_pattern"],
        n_classes=config["n_classes"],
        n_channels=config["n_channels"],
        dim_y=config["dim_y"],
        dim_x=config["dim_x"],
        use_retina=config["use_retina"],
        train_tau=config["train_tau"],
        bias=config["bias"],
        max_weight_init=config["max_weight_init"],
    )

    return DyRCNNx8(**model_kwargs)


def count_trainable_params(overrides: Dict[str, Any]) -> int:
    """Instantiate DyRCNNx8 and count trainable params, including
    skip/feedback 1×1-convs created during setup()."""
    model = _instantiate(overrides)

    # If the model has skip or feedback, call _initialize_connections()
    # to create the auto-adapt 1×1 conv layers.
    if model.feedback or model.skip:
        model.eval()
        with torch.no_grad():
            x = torch.randn((1, model.n_timesteps, model.n_channels,
                             model.dim_y, model.dim_x))
            _ = model.forward(x, store_responses=False)
        model.reset()

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> int:
    # Default
    default_count = count_trainable_params({})
    print(f"Default model: {default_count:,} trainable parameters")
    print()
    print(f"{'Category':25s} {'Value':16s} {'# Params':>12s}  {'Δ':>10s}")
    print("-" * 68)

    for category, config_key, valid_values in VARIATIONS:
        for val in valid_values:
            if category == "Feedback":
                if val == "false":
                    overrides = {"feedback": False}
                elif val == "additive":
                    overrides = {"feedback": True, "feedback_mode": "additive"}
                elif val == "multiplicative":
                    overrides = {"feedback": True, "feedback_mode": "multiplicative"}
                else:
                    overrides = {}
            else:
                overrides = {config_key: val}

            if config_key in ARCH_NEUTRAL_KEYS:
                delta = 0
            else:
                count = count_trainable_params(overrides)
                delta = count - default_count

            label = str(val)
            if category == "Presentation Pattern" and val == "1011":
                label = "1011 (shuffled)"

            delta_str = f"{delta:+,}" if delta != 0 else "0"
            count_str = "—" if config_key in ARCH_NEUTRAL_KEYS else f"{count:>12,}"
            print(f"{category:25s} {label:16s} {count_str}  {delta_str:>10}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
