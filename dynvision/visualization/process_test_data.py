"""Process test data by combining layer responses and test performance metrics.

This script unifies the functionality of process_plotting_data.py and 
process_test_performance.py to create a single comprehensive dataset
containing both layer metrics and test performance metrics.
"""

import argparse
import re
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dynvision.utils import replace_param_in_string
from dynvision.utils.visualization_utils import (
    layer_response_avg,
    layer_response_std,
    spatial_variance,
    feature_variance,
    tensor_to_numpy,
    extract_param_from_string,
)
from dynvision.utils.data_utils import load_df


def chunk_lists(lst1, lst2, chunk_size):
    """Split two lists into chunks of specified size."""
    for i in range(0, len(lst1), chunk_size):
        yield lst1[i : i + chunk_size], lst2[i : i + chunk_size]


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


def process_layer_responses(responses, measures):
    """Process layer responses to calculate various metrics.

    Args:
        responses: Dict of layer_name -> tensor mappings
        measures: List of measure names to calculate

    Returns:
        Dict of metric_name -> value mappings
    """
    print("  Processing layer response metrics...")

    layer_metrics = {}

    for layer_name, response_tensor in responses.items():
        # Always calculate response_avg for all layers
        if "response_avg" in measures:
            layer_metrics[f"{layer_name}_response_avg"] = layer_response_avg(
                response_tensor
            )

        # Also calculate response_std for all layers
        if "response_std" in measures:
            layer_metrics[f"{layer_name}_response_std"] = layer_response_std(
                response_tensor
            )

        # Only calculate spatial/feature variance for non-classifier layers (5D tensors)
        is_classifier = "classifier" in layer_name.lower()

        if "spatial_variance" in measures and not is_classifier:
            layer_metrics[f"{layer_name}_spatial_variance"] = spatial_variance(
                response_tensor
            )

        if "feature_variance" in measures and not is_classifier:
            layer_metrics[f"{layer_name}_feature_variance"] = feature_variance(
                response_tensor
            )

    return layer_metrics


def process_test_performance(df, topk_values):
    """Calculate accuracy, confidence, and topk metrics from test outputs.

    Args:
        df: DataFrame with test outputs
        topk_values: List of k values for top-k accuracy

    Returns:
        DataFrame with processed performance metrics
    """
    print("  Processing test performance metrics...")

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
        f"    Time steps range: {df['times_index'].min()} to {df['times_index'].max()}"
    )
    print(f"    Unique samples: {df['sample_index'].nunique()}")
    print(f"    Classes range: {df['class_index'].min()} to {df['class_index'].max()}")

    # Calculate confidence and top-k accuracy for each (sample, times_index) group
    print(f"    Calculating confidence and top-k accuracy for k={topk_values}...")

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

    # Remove label_set columns (we'll compress across classes)
    columns_to_drop = []
    if "label_set" in df.columns:
        columns_to_drop.append("label_set")

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"    Dropped columns: {columns_to_drop}")

    # Add presentation_label column
    print("    Adding presentation labels...")
    df["presentation_label"] = df.groupby(["sample_index"])["label_index"].transform(
        get_presentation_label
    )

    # Group by presentation_label and times_index to compress data (keeping times_index dimension)
    print("    Compressing data...")

    # Since we have multiple rows per (sample, times_index) due to class_index,
    # we need to compress by taking one row per (sample, times_index) first
    sample_time_df = df.groupby(["sample_index", "times_index"]).first().reset_index()

    print(
        f"    Compressed from {len(df)} to {len(sample_time_df)} rows (one per sample-time)"
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
    print(f"    Compressed to {len(compressed_df)} rows")

    return compressed_df


def process_single_batch(
    response_files, test_output_files, data_arg_key, measures, category, topk_values
):
    """Process a single batch of response and test output files.

    Args:
        response_files: List of response .pt files
        test_output_files: List of test output .csv files
        data_arg_key: Parameter key for extraction
        measures: List of layer measures to compute
        category: Category key for extraction
        topk_values: List of k values for top-k accuracy

    Returns:
        DataFrame with combined layer and performance metrics
    """
    batch_dfs = []

    for pt_file, csv_file in zip(response_files, test_output_files):
        print(f"  Processing: {pt_file.name} + {csv_file.name}")

        # Extract parameter and category values
        arg_value = extract_param_from_string(
            pt_file.stem, key=data_arg_key, value_type=float
        )
        cat_value = extract_param_from_string(
            pt_file.stem, key=category, value_type=None
        )

        # Verify matching values
        csv_arg_value = extract_param_from_string(
            csv_file.stem, key=data_arg_key, value_type=float
        )
        csv_cat_value = extract_param_from_string(
            csv_file.stem, key=category, value_type=None
        )

        if not arg_value == csv_arg_value:
            raise ValueError(
                f"{data_arg_key} values do not match: {arg_value} vs {csv_arg_value}"
            )
        if not cat_value == csv_cat_value:
            raise ValueError(
                f"{category} values do not match: {cat_value} vs {csv_cat_value}"
            )

        # Load test outputs
        test_df = load_df(csv_file)

        # Process test performance metrics
        performance_df = process_test_performance(test_df, topk_values)

        # Load layer responses
        responses = torch.load(pt_file, map_location=torch.device("cpu"))

        # Pad responses to same number of timesteps
        max_timesteps = max(tensor.shape[1] for tensor in responses.values())
        for layer_name, tensor in responses.items():
            pad_len = max_timesteps - tensor.shape[1]
            if pad_len > 0:
                # Pad at the start along axis 1 (time steps) with zeros
                pad = (0, 0, 0, 0, 0, 0, pad_len, 0)
                tensor = torch.nn.functional.pad(tensor, pad, mode="constant", value=0)
                responses[layer_name] = tensor

        # Process layer metrics
        layer_metrics = process_layer_responses(responses, measures)

        # Convert layer metrics to dataframe format
        layer_names = list(responses.keys())
        n_samples, n_timesteps, *_ = responses[layer_names[0]].shape

        # Create rows for each presentation_label and times_index combination
        layer_rows = []
        unique_presentations = performance_df["presentation_label"].unique()
        unique_times = performance_df["times_index"].unique()

        for pres_label in unique_presentations:
            for times_idx in unique_times:
                # Find the sample_index for this presentation_label
                sample_indices = test_df[
                    test_df.groupby("sample_index")["label_index"].transform(
                        get_presentation_label
                    )
                    == pres_label
                ]["sample_index"].unique()

                if len(sample_indices) == 0:
                    continue

                # Use first sample for this presentation (they should be the same)
                sample_idx = sample_indices[0]

                row_data = {
                    "presentation_label": pres_label,
                    "times_index": times_idx,
                }

                # Add layer metrics for this sample and time
                for layer_name in layer_names:
                    for measure in measures:
                        metric_name = f"{layer_name}_{measure}"
                        if metric_name in layer_metrics:
                            metric_tensor = layer_metrics[metric_name]
                            row_data[metric_name] = tensor_to_numpy(
                                metric_tensor[sample_idx, times_idx]
                            )

                layer_rows.append(row_data)

        layer_df = pd.DataFrame(layer_rows)

        # Merge layer metrics with performance metrics
        merged_df = performance_df.merge(
            layer_df, on=["presentation_label", "times_index"], how="left"
        )

        # Add parameter and category information
        merged_df[data_arg_key] = arg_value
        merged_df[category] = cat_value

        batch_dfs.append(merged_df)
        print(f"    Created merged dataframe with {len(merged_df)} rows")

    # Combine all files in this batch
    if batch_dfs:
        batch_combined = pd.concat(batch_dfs, ignore_index=True)
        return batch_combined
    else:
        return pd.DataFrame()


def process_test_data_in_batches(
    response_files,
    test_output_files,
    data_arg_key,
    measures,
    category,
    topk_values,
    batch_size=1,
):
    """Process test data in batches to manage memory efficiently.

    Args:
        response_files: List of response .pt files
        test_output_files: List of test output .csv files
        data_arg_key: Parameter key for extraction
        measures: List of layer measures to compute
        category: Category key for extraction
        topk_values: List of k values for top-k accuracy
        batch_size: Number of files to process per batch

    Returns:
        Combined DataFrame with all processed data
    """
    if len(response_files) != len(test_output_files):
        raise ValueError(
            "Number of response files must match number of test output files"
        )

    print(f"Processing {len(response_files)} file pairs in batches of {batch_size}")

    all_dataframes = []

    # Process files in batches
    batch_count = 0
    for response_batch, output_batch in chunk_lists(
        response_files, test_output_files, batch_size
    ):
        batch_count += 1
        print(
            f"Processing batch {batch_count}/{(len(response_files) + batch_size - 1) // batch_size}"
        )
        print(
            f"  Files: {len(response_batch)} response files, {len(output_batch)} output files"
        )

        try:
            # Process this batch
            batch_df = process_single_batch(
                response_batch,
                output_batch,
                data_arg_key=data_arg_key,
                measures=measures,
                category=category,
                topk_values=topk_values,
            )

            if not batch_df.empty:
                all_dataframes.append(batch_df)
                print(f"  Batch {batch_count} processed: {len(batch_df)} rows")
            else:
                print(f"  Batch {batch_count}: No data processed")

            # Clear memory
            del batch_df

        except Exception as e:
            print(f"Error processing batch {batch_count}: {e}")
            print(f"  Response files: {response_batch}")
            print(f"  Output files: {output_batch}")
            continue

    if not all_dataframes:
        raise ValueError("No data was successfully loaded from any batch")

    # Combine all dataframes
    print(f"Combining {len(all_dataframes)} batches...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Clear memory
    del all_dataframes

    print(f"Total combined data: {len(combined_df)} rows")
    return combined_df


# Command line interface
parser = argparse.ArgumentParser()
parser.add_argument(
    "--responses",
    nargs="+",
    type=Path,
    required=True,
    help="Path to response .pt files",
)
parser.add_argument(
    "--test_outputs",
    nargs="+",
    type=Path,
    required=True,
    help="Path to test output .csv files",
)
parser.add_argument(
    "--output", type=Path, required=True, help="Path to output CSV file"
)
parser.add_argument(
    "--parameter", type=str, required=True, help="Parameter to extract from filenames"
)
parser.add_argument(
    "--category", type=str, required=True, help="Category to extract from filenames"
)
parser.add_argument(
    "--measures",
    nargs="+",
    type=str,
    default=["response_avg", "response_std", "spatial_variance", "feature_variance"],
    help="Layer measures to compute",
)
parser.add_argument(
    "--topk",
    nargs="+",
    type=int,
    default=[3, 5],
    help="Top-k values for calculating top-k accuracy",
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="Batch size for processing files"
)


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    # Handle interval parameter measure adjustment (from original process_plotting_data logic)
    if "interval" in args.parameter:
        # For interval experiments, we might want different default measures
        print(f"Detected interval parameter: {args.parameter}")

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Process data in batches
    print(f"Processing with measures: {args.measures}")
    print(f"Top-k values: {args.topk}")

    df = process_test_data_in_batches(
        args.responses,
        args.test_outputs,
        data_arg_key=args.parameter,
        measures=args.measures,
        category=args.category,
        topk_values=args.topk,
        batch_size=args.batch_size,
    )

    # Save processed data
    print(f"Saving processed data to: {args.output}")
    df.to_csv(args.output, index=False)
    print(f"Processed data saved with {len(df)} total rows")

    # Print summary statistics
    print("\nDataset Summary:")
    print(f"  Shape: {df.shape}")
    print(f"  Unique {args.category} values: {df[args.category].nunique()}")
    print(f"  Unique {args.parameter} values: {df[args.parameter].nunique()}")
    print(f"  Time steps: {df['times_index'].nunique()}")
    print(f"  Presentation labels: {df['presentation_label'].nunique()}")

    # Layer metrics summary
    layer_cols = [
        col for col in df.columns if any(measure in col for measure in args.measures)
    ]
    print(f"  Layer metric columns: {len(layer_cols)}")

    # Performance metrics summary
    if "accuracy" in df.columns:
        print(
            f"  Accuracy range: {df['accuracy'].min():.3f} to {df['accuracy'].max():.3f}"
        )

    for k in args.topk:
        topk_col = f"accuracy_top{k}"
        if topk_col in df.columns:
            print(
                f"  Top-{k} accuracy range: {df[topk_col].min():.3f} to {df[topk_col].max():.3f}"
            )
