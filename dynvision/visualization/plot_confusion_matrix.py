import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dynvision.data.dataset import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True, help="Path to CSV file")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--dataset", type=Path, required=True, help="Name of dataset")
parser.add_argument("--palette", type=str, default="viridis")


def plot_confusion_matrix(
    cm: np.ndarray,
    palette: str = "viridis",
    index_to_class: dict[int, str] = None,
) -> plt.Axes:

    sns.set_context("talk")
    sns.set_style("ticks")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Reverse the order of the confusion matrix to achieve origin='lower'
    cm = cm[::-1]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=palette,
        ax=ax,
        cbar=True,
        square=True,
    )

    ax.set_xlabel("guess")
    ax.set_ylabel("label")

    if index_to_class is not None:
        ax.set_xticks(np.arange(cm.shape[1]) + 0.5)
        ax.set_yticks(np.arange(cm.shape[0])[::-1] + 0.5)
        ax.set_xticklabels([index_to_class[i] for i in range(cm.shape[1])])
        ax.set_yticklabels([index_to_class[i] for i in range(cm.shape[0])])

    return ax


if __name__ == "__main__":
    args = parser.parse_args()

    # Load the testing results
    if not args.input.exists():
        raise FileNotFoundError(f"Input file '{args.input}' not found.")

    df = pd.read_csv(args.input)

    # Calculate the confusion matrix
    cm = confusion_matrix(df["label_index"], df["guess_index"])

    # Load the dataset
    dataset = get_dataset(args.dataset)

    class_to_index = defaultdict(lambda: -1, dataset.class_to_idx)
    index_to_class = {v: k for k, v in class_to_index.items()}

    # Plot the confusion matrix
    ax = plot_confusion_matrix(
        cm,
        palette=args.palette,
        index_to_class=index_to_class,
    )

    # Save the plot
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)
