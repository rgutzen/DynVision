import argparse
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from dynvision.utils.visualization_utils import tensor_to_numpy

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True, help="Path to pt file")
parser.add_argument("--output", type=Path, required=True, help="Path to directory")


def get_layer_names_from_state_dict(state_dict, key_marker="tstep") -> list:
    layer_names = set()
    for key in state_dict.keys():
        if key_marker in key:
            layer_name = key.split(".")[0].split("_")[-1]
            layer_names.add(layer_name)
    return list(layer_names)


def plot_weight_distributions(state_dict):

    sns.set_context("talk")
    sns.set_style("ticks")

    layer_names = get_layer_names_from_state_dict(state_dict, key_marker="tstep")

    df = pd.DataFrame()

    for key, params in state_dict.items():
        module_name = key.split(".")[0]
        if module_name in layer_names and "weight" in key:
            weights = params
            layer_name = module_name
            flattened_weights = tensor_to_numpy(weights.flatten())

            temp_df = pd.DataFrame(
                {
                    "weight": flattened_weights,
                    "layer": layer_name,
                    "type": "recurrence" if "recurrence" in key else "feedforward",
                }
            )
            df = pd.concat([df, temp_df], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    if len(df):
        split = "recurrence" in df["type"].values
        sns.violinplot(
            data=df,
            x="layer",
            y="weight",
            hue="type",
            split=split,
            ax=ax,
            inner=None,
            density_norm="width",
        )

    ax.legend(loc="upper right", title=None, frameon=False)

    sns.despine(left=True, ax=ax)
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file '{args.input}' not found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    data_identifier = args.input.stem

    state_dict = torch.load(args.input)

    plot_weight_distributions(state_dict)

    plt.savefig(args.output, bbox_inches="tight")
