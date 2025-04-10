import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dynvision.utils import load_df

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True, help="Path to CSV file")
parser.add_argument("--output", type=Path, required=True, help="Path to directory")


def plot_classifier_responses(df, n_lines=10) -> plt.Axes:

    sns.set_context("talk")
    sns.set_style("ticks")

    label_set = df.label_set.unique()[0]

    # if n_lines:
    #     class_indices = df.class_index.unique()[:n_lines]
    #     df = df[df.class_index.isin(class_indices)]

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.lineplot(
        data=df,
        x="times_index",
        y="response",
        hue="class_index",
        errorbar=("ci", 75),
        palette="tab10",
        ax=ax,
        marker=".",
        # legend=False,
    )

    for changepoint in get_label_changepoints(df, label_set):
        ax.axvline(changepoint, color="0.6", linestyle="--", linewidth=1)

    # # For DoubleStimulus:
    # ax.set_xlim(9, 29)
    # ax.set_ylim(0, 20)  # full 18, self 24
    # ax.set_yticks([0, 5, 10, 15, 20])
    # ax.fill_between([9, 29], 0, 8, color="0.8", zorder=-5)  # remove again
    # ax.set_xlabel("Time (ms)")
    # ax.set_ylabel("")
    # # ax.legend(title="Class", bbox_to_anchor=(0.93, 1), loc="upper left", frameon=False)
    # ax.set_xticklabels(
    #     [f"{int(label.get_text().strip('−')) * 2}" for label in ax.get_xticklabels()]
    # )

    ax.set_xlabel("time step")
    ax.set_ylabel("classifier response")
    ax.set_title(f"{label_set}")
    ax.text(
        0.05,
        0.85,
        f"accuracy: {calculate_accuracy(df):.2f}",
        transform=ax.transAxes,
    )

    ax.legend(
        title="class index", bbox_to_anchor=(0.93, 1), loc="upper left", frameon=False
    )
    sns.despine()
    plt.tight_layout()

    return fig, ax


def get_label_changepoints(df, label_set):
    label_set_df = df[df.label_set == label_set]
    sample_index = label_set_df.sample_index.unique()[0]
    class_index = label_set_df.class_index.unique()[0]
    label_set_df = label_set_df[
        (label_set_df.sample_index == sample_index)
        & (label_set_df.class_index == class_index)
    ]
    label_indices = label_set_df.label_index.values

    changepoints = np.where(np.diff(label_indices) != 0)[0]
    return changepoints


def calculate_accuracy(df):
    dfi = df[df.label_index != -1]
    n_correct = (dfi.guess_index == dfi.label_index).sum()
    accuracy = n_correct / len(dfi)
    return accuracy


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file '{args.input}' not found.")

    args.output.mkdir(parents=True, exist_ok=True)

    data_identifier = args.input.stem

    df = load_df(args.input)

    for label_set in df.label_set.unique():
        plot_df = df[(df.label_set == label_set)]

        # plot_df = plot_df[(plot_df.class_index.isin([8, 9]))]  # remove again

        plot_classifier_responses(plot_df)

        plt.savefig(
            args.output / f"{data_identifier}_label{label_set}.png",
            bbox_inches="tight",
        )
