import argparse
import re
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch

from dynvision.utils import replace_param_in_string

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_outputs", nargs="+", type=Path, required=True, help="Path to csv files"
)
parser.add_argument(
    "--output", type=Path, required=True, help="Path to output csv file"
)
parser.add_argument(
    "--parameter", type=str, required=True, help="Parameter to extract from filename"
)
parser.add_argument(
    "--category", type=str, required=True, help="Category to extract from filename"
)
parser.add_argument(
    "--topk",
    nargs="+",
    type=int,
    default=[5],
    help="Top-k values for calculating top-k accuracy",
)


def extract_category_value(filename, category):
    """Extract category value from filename string."""
    pattern = rf"{category}=([^_+]+)"
    match = re.search(pattern, str(filename))
    return match.group(1) if match else None


def extract_parameter_value(filename, parameter):
    """Extract parameter value from filename string."""
    pattern = rf"{parameter}=([^_+]+)"
    match = re.search(pattern, str(filename))
    return match.group(1) if match else None


def calculate_confidence_from_responses(group):
    """Calculate confidence as softmax probability for the true label from response values."""
    # Get all response values for each class
    responses = torch.tensor(group["response"].values)

    # Apply softmax
    probabilities = F.softmax(responses, dim=0)

    # Get confidence for the true label (first row should have the label)
    label_idx = int(group.iloc[0]["label_index"])

    if label_idx >= 0 and label_idx < len(probabilities):
        # Find the response for the true label class
        class_indices = group["class_index"].values
        if label_idx in class_indices:
            true_label_position = np.where(class_indices == label_idx)[0][0]
            return probabilities[true_label_position].item()

    return np.nan


def calculate_topk_accuracy(group, k):
    """Calculate top-k accuracy for a group of responses."""
    # Get all response values and corresponding class indices
    responses = group["response"].values
    class_indices = group["class_index"].values

    # Get the true label
    label_idx = int(group.iloc[0]["label_index"])

    if label_idx < 0:
        return np.nan

    # Get top-k class indices based on response values
    top_k_indices = np.argsort(responses)[-k:]  # Get indices of top-k responses
    top_k_classes = class_indices[top_k_indices]  # Get corresponding class indices

    # Check if true label is in top-k
    return float(label_idx in top_k_classes)


def get_presentation_label(label_series):
    """Get the presentation label for a group (sample across time steps)."""
    # Find the first non-negative label_index in the series
    valid_labels = label_series[label_series >= 0]
    if len(valid_labels) > 0:
        return valid_labels.iloc[0]
    else:
        return -1


def process_csv_file(csv_path, parameter, category, topk_values):
    """Process a single CSV file."""
    print(f"Processing: {csv_path}")

    # Load the CSV file
    df = pd.read_csv(csv_path)
    print(f"  Loaded dataframe with {len(df)} rows, {len(df.columns)} columns")

    # Sanity check: verify required columns exist
    required_cols = [
        "sample_index",
        "times_index",
        "label_index",
        "guess_index",
        "class_index",
        "response",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(
        f"  Time steps range: {df['times_index'].min()} to {df['times_index'].max()}"
    )
    print(f"  Unique samples: {df['sample_index'].nunique()}")
    print(f"  Classes range: {df['class_index'].min()} to {df['class_index'].max()}")

    # Calculate confidence and top-k accuracy for each (sample, times_index) group
    print(f"  Calculating confidence and top-k accuracy for k={topk_values}...")

    # Group by sample and times_index to calculate metrics from responses
    metrics_data = []

    for (sample_idx, times_index), group in df.groupby(
        ["sample_index", "times_index"]
    ):
        confidence = calculate_confidence_from_responses(group)

        # Calculate top-k accuracy for each k value
        topk_accuracies = {}
        for k in topk_values:
            topk_accuracies[f"topk_accuracy_{k}"] = calculate_topk_accuracy(group, k)

        # Add metrics to each row in the group
        for idx in group.index:
            row_data = {
                "index": idx,
                "confidence": confidence,
                **topk_accuracies,
            }
            metrics_data.append(row_data)

    # Create metrics dataframe and merge
    metrics_df = pd.DataFrame(metrics_data).set_index("index")
    df = df.merge(metrics_df, left_index=True, right_index=True)

    # Sanity check: metrics calculation
    valid_confidence = df["confidence"].notna().sum()
    total_rows = len(df)
    print(
        f"  Valid confidence values: {valid_confidence}/{total_rows} ({valid_confidence/total_rows*100:.1f}%)"
    )

    for k in topk_values:
        valid_topk = df[f"topk_accuracy_{k}"].notna().sum()
        print(
            f"  Valid top-{k} accuracy values: {valid_topk}/{total_rows} ({valid_topk/total_rows*100:.1f}%)"
        )

    # Remove label_set columns (we'll compress across classes)
    columns_to_drop = []
    if "label_set" in df.columns:
        columns_to_drop.append("label_set")

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"  Dropped columns: {columns_to_drop}")

    # Add presentation_label column
    print("  Adding presentation labels...")
    df["presentation_label"] = df.groupby(["sample_index"])["label_index"].transform(
        get_presentation_label
    )

    # Sanity check: presentation labels
    unique_presentation_labels = df["presentation_label"].unique()
    print(
        f"  Unique presentation labels: {len(unique_presentation_labels)} (range: {unique_presentation_labels.min()} to {unique_presentation_labels.max()})"
    )

    # Group by presentation_label and times_index to compress data (keeping times_index dimension)
    print("  Compressing data...")

    # Since we have multiple rows per (sample, times_index) due to class_index,
    # we need to compress by taking one row per (sample, times_index) first
    sample_time_df = df.groupby(["sample_index", "times_index"]).first().reset_index()

    print(
        f"  Compressed from {len(df)} to {len(sample_time_df)} rows (one per sample-time)"
    )

    # Now group by presentation_label and times_index
    grouped = sample_time_df.groupby(["presentation_label", "times_index"])

    # Create compressed dataframe
    compressed_stats = []

    for (pres_label, times_index), group in grouped:
        # Calculate accuracy
        accuracy = (group["guess_index"] == pres_label).mean()

        # Calculate top-k accuracy for each k value
        topk_accuracy_avgs = {}
        for k in topk_values:
            topk_accuracy_avgs[f"accuracy_top{k}"] = group[f"topk_accuracy_{k}"].mean()

        # Calculate confidence stats
        conf_avg = group["confidence"].mean()
        conf_std = group["confidence"].std()

        # Get other columns (should be identical within group)
        # KEEP label_index since it's different from presentation_label (contains -1 for some timesteps)
        other_data = {}
        exclude_cols = [
            "sample_index",
            "guess_index",
            "confidence",
            "class_index",
            "response",
        ] + [
            f"topk_accuracy_{k}" for k in topk_values
        ]  # Exclude all topk_accuracy columns

        for col in sample_time_df.columns:
            if col not in exclude_cols:
                # Take first value (should be identical within group)
                other_data[col] = group[col].iloc[0]

        # Combine all data
        row_data = {
            "presentation_label": pres_label,
            "times_index": times_index,
            "accuracy": accuracy,
            **topk_accuracy_avgs,  # Add all top-k accuracy columns
            "confidence_avg": conf_avg,
            "confidence_std": conf_std,
            **other_data,
        }

        compressed_stats.append(row_data)

    compressed_df = pd.DataFrame(compressed_stats)

    print(f"  Compressed to {len(compressed_df)} rows")
    print(f"  Time steps preserved: {compressed_df['times_index'].nunique()}")

    # Sanity check: verify compression maintained time steps
    original_time_steps = sample_time_df["times_index"].nunique()
    compressed_time_steps = compressed_df["times_index"].nunique()
    if original_time_steps != compressed_time_steps:
        print(
            f"  WARNING: Time steps changed from {original_time_steps} to {compressed_time_steps}"
        )

    # Extract category and parameter values from filename
    category_value = extract_category_value(csv_path.name, category)
    parameter_value = extract_parameter_value(csv_path.name, parameter)

    # Add category and parameter columns
    compressed_df[category] = category_value
    compressed_df[parameter] = parameter_value

    print(f"  Added {category}={category_value}, {parameter}={parameter_value}")

    # Sanity check: final dataframe
    print(f"  Final shape: {compressed_df.shape}")
    print(f"  Columns: {list(compressed_df.columns)}")
    print(
        f"  Label index range: {compressed_df['label_index'].min()} to {compressed_df['label_index'].max()}"
    )
    print(
        f"  Accuracy range: {compressed_df['accuracy'].min():.3f} to {compressed_df['accuracy'].max():.3f}"
    )

    for k in topk_values:
        topk_col = f"accuracy_top{k}"
        print(
            f"  Top-{k} accuracy range: {compressed_df[topk_col].min():.3f} to {compressed_df[topk_col].max():.3f}"
        )

    print(
        f"  Confidence avg range: {compressed_df['confidence_avg'].min():.3f} to {compressed_df['confidence_avg'].max():.3f}"
    )

    return compressed_df


if __name__ == "__main__":
    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Process each CSV file
    all_dataframes = []
    total_rows = 0

    print(
        f"Processing {len(args.test_outputs)} CSV files with top-k accuracy for k={args.topk}..."
    )

    for csv_path in tqdm(args.test_outputs, desc="Processing CSV files"):
        # Sanity check: file exists
        if not csv_path.exists():
            print(f"  WARNING: File does not exist: {csv_path}")
            continue

        df = process_csv_file(csv_path, args.parameter, args.category, args.topk)
        all_dataframes.append(df)
        total_rows += len(df)
        print(f"  Running total: {total_rows} rows")
        print("-" * 50)

    # Sanity check: we have data to concatenate
    if not all_dataframes:
        raise ValueError("No dataframes to concatenate!")

    # Concatenate all dataframes
    print("Concatenating all dataframes...")
    final_df = pd.concat(all_dataframes, ignore_index=True)

    # Final sanity checks
    print(f"Final dataframe shape: {final_df.shape}")
    print(f"Final dataframe columns: {list(final_df.columns)}")
    print(f"Time steps in final dataframe: {final_df['times_index'].nunique()}")
    print(f"Unique {args.category} values: {final_df[args.category].nunique()}")
    print(f"Unique {args.parameter} values: {final_df[args.parameter].nunique()}")
    print(
        f"Accuracy range: {final_df['accuracy'].min():.3f} to {final_df['accuracy'].max():.3f}"
    )

    for k in args.topk:
        topk_col = f"accuracy_top{k}"
        print(
            f"Top-{k} accuracy range: {final_df[topk_col].min():.3f} to {final_df[topk_col].max():.3f}"
        )
        print(f"Missing values in top-{k} accuracy: {final_df[topk_col].isna().sum()}")

    print(f"Missing values in accuracy: {final_df['accuracy'].isna().sum()}")

    # Save the result
    print(f"Saving processed data to: {args.output}")
    final_df.to_csv(args.output, index=False)
    print(f"Processed data saved with {len(final_df)} total rows")
