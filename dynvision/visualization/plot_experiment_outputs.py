import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dynvision.utils import extract_param_from_string, load_df
from dynvision.visualization.plot_classifier_responses import (
    calculate_accuracy,
    get_label_changepoints,
)
from dynvision.utils.visualization_utils import save_plot

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_outputs", nargs="+", type=Path, required=True, help="Path to csv files"
)
parser.add_argument("--output", type=Path, required=True, help="Path to directory")
parser.add_argument("--parameter", type=str, required=True, help="Parameter to plot")
parser.add_argument("--category", type=str, required=True, help="Category to plot")


def calculate_confidence(df):
    confidence = None
    return confidence


def load_test_outputs(csv_files, data_arg_key="contrast"):
    dfs = []

    for csv_file in csv_files:
        arg_value = extract_param_from_string(
            csv_file.name, key=data_arg_key, value_type=float
        )
        rctype = extract_param_from_string(csv_file.name, key="rctype", value_type=str)

        df = load_df(csv_file)
        df[data_arg_key] = arg_value
        df["recurrence_type"] = rctype

        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    return df


def plot_classifier_response(df, ax):
    sns.lineplot(
        data=df,
        x="times_index",
        y="response",
        hue="class_index",
        errorbar=("ci", 75),
        palette="tab10",
        ax=ax,
        marker=".",
    )
    # ax.set_xlabel('time step')
    ax.set_xlabel("")
    ax.set_ylabel("")
    # ax.set_ylabel('classifier response')
    # ax.set_title(f"{label_set}")
    ax.text(0.05, 0.8, f"{calculate_accuracy(df)*100:.0f}%", transform=ax.transAxes)
    ax.legend().remove()
    sns.despine(ax=ax)
    return ax


def plot_classifier_responses(df, data_arg_key="contrast", label_target="8"):
    df_label_set = df[df.label_set.str.contains(str(label_target))]
    recurrence_types = df_label_set.recurrence_type.unique()
    data_arg_values = df_label_set[data_arg_key].unique()

    fig, axes = plt.subplots(
        nrows=len(recurrence_types),
        ncols=len(data_arg_values),
        sharex=True,
        sharey=True,
        figsize=(len(data_arg_values) * 3.5, len(recurrence_types) * 3.5 + 2),
    )

    for i, rctype in enumerate(recurrence_types):
        for j, value in enumerate(np.sort(data_arg_values)):
            ax = axes[i, j]
            plot_df = df_label_set[
                (df_label_set[data_arg_key] == value)
                & (df_label_set.recurrence_type == rctype)
            ]

            plot_classifier_response(plot_df, ax)

            label_set = plot_df.label_set.unique()[0]
            for changepoint in get_label_changepoints(plot_df, label_set):
                ax.axvline(changepoint, color="0.6", linestyle="--", linewidth=1)

            axes[-1, j].set_xlabel("time step")
            axes[i, 0].set_ylabel(rctype, weight="bold")
            axes[0, j].set_title(f"{data_arg_key}={value}")

    axes[0, -1].legend(
        title="class index", bbox_to_anchor=(0.93, 1), loc="upper left", frameon=False
    )
    return fig, axes


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    df = load_test_outputs(
        args.test_outputs,
        data_arg_key=args.parameter,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_identifier = args.output.parent.name

    label_targets = df.label_index.unique()
    label_targets = label_targets[label_targets != -1]

    for label_target in label_targets:
        fig, axes = plot_classifier_responses(
            df, data_arg_key=args.parameter, label_target=label_target
        )

        fig.suptitle(f"label={label_target}\n {data_identifier}")

        save_plot(args.output.parent / f"experiment_outputs_label{label_target}.png")
