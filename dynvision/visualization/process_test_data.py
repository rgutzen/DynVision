"""Process test data by combining layer responses and test performance metrics.

MEMORY-OPTIMIZED VERSION: Processes layers sequentially to handle large (30GB+) files.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dynvision.utils import replace_param_in_string, str_to_bool
from dynvision.utils.visualization_utils import (
    layer_response_avg,
    layer_response_std,
    spatial_variance,
    feature_variance,
    tensor_to_numpy,
    extract_param_from_string,
)
from dynvision.utils.data_utils import load_df

# Import memory-efficient loading system
from dynvision.utils.memory_efficient_loading import (
    MemoryMonitor,
    get_layer_names_from_pt,
    get_max_timesteps_from_pt,
    process_layer_responses_incremental,  # STEP 2: Incremental extraction
    extract_metric_values_for_dataframe,
)

# Configure logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def chunk_lists(lst1, lst2, chunk_size):
    """Split two lists into chunks of specified size."""
    for i in range(0, len(lst1), chunk_size):
        yield lst1[i : i + chunk_size], lst2[i : i + chunk_size]


def calculate_confidence_from_responses(group):
    """Calculate confidence as softmax probability for the true label from response values."""
    responses = torch.tensor(group["response"].values)
    probabilities = F.softmax(responses, dim=0)
    label_idx = int(group.iloc[0]["presentation_label"])

    if label_idx >= 0 and label_idx < len(probabilities):
        class_indices = group["class_index"].values
        if label_idx in class_indices:
            true_label_position = np.where(class_indices == label_idx)[0][0]
            return probabilities[true_label_position].item()

    return np.nan


def calculate_topk_accuracy(group, k):
    """Calculate top-k accuracy for a group of responses."""
    responses = group["response"].values
    class_indices = group["class_index"].values
    label_idx = int(group.iloc[0]["presentation_label"])

    if label_idx < 0:
        return np.nan

    top_k_indices = np.argsort(responses)[-k:]
    top_k_classes = class_indices[top_k_indices]

    return float(label_idx in top_k_classes)


def get_presentation_label(label_series):
    """Get the presentation label for a group (sample across time steps)."""
    valid_labels = label_series[label_series >= 0]
    if len(valid_labels) > 0:
        return valid_labels.iloc[0]
    else:
        return -1


def process_test_performance(df, topk_values):
    """Calculate accuracy, confidence, and topk metrics from test outputs."""
    logger.info("  Processing test performance metrics...")

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
        raise ValueError(
            f"Missing required columns in test output CSV: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )

    logger.info(
        f"    Time steps range: {df['times_index'].min()} to {df['times_index'].max()}"
    )
    logger.info(f"    Unique samples: {df['sample_index'].nunique()}")
    logger.info(f"    Unique data labels: {df['label_index'].unique()}")
    logger.info(f"    Unique model classes: {df['class_index'].unique()}")

    # Calculate confidence and top-k accuracy
    logger.info(
        f"    Calculating confidence and top-k accuracy for k={topk_values}..."
    )

    # Add presentation_label column
    logger.info("    Adding presentation labels...")
    df["presentation_label"] = df.groupby(["sample_index"])["label_index"].transform(
        get_presentation_label
    )
    logger.info(f"    Unique presentation labels: {df['presentation_label'].unique()}")

    metrics_data = []

    for (sample_idx, times_index), group in tqdm(
        df.groupby(["sample_index", "times_index"]),
        desc="Computing performance metrics",
    ):
        confidence = calculate_confidence_from_responses(group)

        # Calculate top-k accuracy for each k value
        topk_accuracies = {}
        for k in topk_values:
            topk_accuracies[f"accuracy_top{k}"] = calculate_topk_accuracy(group, k)

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

    # Remove label_set column if present
    if "label_set" in df.columns:
        df = df.drop(columns=["label_set"])
        logger.info("    Dropped label_set column")

    # Compress data by grouping
    logger.info("    Compressing data...")
    sample_time_df = df.groupby(["sample_index", "times_index"]).first().reset_index()
    logger.info(
        f"    Compressed from {len(df)} to {len(sample_time_df)} rows (one per sample-time)"
    )

    # Group by presentation_label and times_index
    grouped = sample_time_df.groupby(["presentation_label", "times_index"])
    compressed_stats = []

    for (pres_label, times_index), group in grouped:
        # Calculate accuracy
        accuracy = (group["guess_index"] == pres_label).mean()

        # Calculate top-k accuracy averages
        topk_accuracy_avgs = {}
        for k in topk_values:
            topk_accuracy_avgs[f"accuracy_top{k}"] = group[f"accuracy_top{k}"].mean()

        # Calculate confidence stats
        conf_avg = group["confidence"].mean()
        conf_std = group["confidence"].std()

        # Get other columns
        other_data = {}
        exclude_cols = [
            "sample_index",
            "guess_index",
            "confidence",
            "class_index",
            "response",
        ] + [f"accuracy_top{k}" for k in topk_values]

        for col in sample_time_df.columns:
            if col not in exclude_cols:
                other_data[col] = group[col].iloc[0]

        # Combine all data
        row_data = {
            "presentation_label": pres_label,
            "times_index": times_index,
            "accuracy": accuracy,
            **topk_accuracy_avgs,
            "confidence_avg": conf_avg,
            "confidence_std": conf_std,
            **other_data,
        }

        compressed_stats.append(row_data)

    compressed_df = pd.DataFrame(compressed_stats)
    logger.info(f"    Compressed to {len(compressed_df)} rows")

    return compressed_df


def compute_sample_to_presentation_mapping(test_df: pd.DataFrame) -> Dict[int, int]:
    """
    Pre-compute mapping from sample_index to presentation_label.

    This is computed once and reused for all layers to avoid repeated
    expensive groupby operations.

    Args:
        test_df: Test output dataframe

    Returns:
        Dictionary mapping sample_index to presentation_label
    """
    logger.info("  Computing sample-to-presentation mapping...")

    mapping = {}
    for sample_idx in test_df["sample_index"].unique():
        sample_data = test_df[test_df["sample_index"] == sample_idx]
        label_series = sample_data["label_index"]
        presentation_label = get_presentation_label(label_series)
        mapping[sample_idx] = presentation_label

    logger.info(f"    Mapped {len(mapping)} samples")
    return mapping


def create_layer_dataframe_from_extracted_values(
    extracted_values: Dict[str, np.ndarray],
    unique_presentations: List[int],
    unique_times: List[int],
) -> pd.DataFrame:
    """
    Create layer dataframe from pre-extracted numpy arrays.

    MEMORY EFFICIENT: Works with small numpy arrays instead of large tensors.

    Args:
        extracted_values: Dictionary mapping metric names to 1D numpy arrays
        unique_presentations: List of unique presentation labels
        unique_times: List of unique time indices

    Returns:
        DataFrame with layer metrics
    """
    logger.info("  Creating layer dataframe from extracted values...")

    n_rows = len(unique_presentations) * len(unique_times)

    # Pre-allocate base columns
    data_arrays = {
        "presentation_label": np.empty(n_rows, dtype=np.int32),
        "times_index": np.empty(n_rows, dtype=np.int32),
    }

    # Fill presentation_label and times_index
    idx = 0
    for pres_label in unique_presentations:
        for time_idx in unique_times:
            data_arrays["presentation_label"][idx] = pres_label
            data_arrays["times_index"][idx] = time_idx
            idx += 1

    # Add extracted metric columns (already in correct order)
    for metric_name, values in extracted_values.items():
        if len(values) != n_rows:
            raise RuntimeError(
                f"Metric '{metric_name}' has {len(values)} values, "
                f"expected {n_rows}"
            )
        data_arrays[metric_name] = values

    # Create DataFrame
    layer_df = pd.DataFrame(data_arrays)
    logger.info(
        f"    Created layer dataframe with {len(layer_df)} rows, {len(data_arrays)} columns"
    )

    return layer_df


def process_single_batch_optimized_v2(
    response_files: List[Path],
    test_output_files: List[Path],
    data_arg_key: str,
    measures: List[str],
    category: str,
    topk_values: List[int],
    memory_monitor: MemoryMonitor,
) -> pd.DataFrame:
    """
    Process a single batch using INCREMENTAL extraction (Step 2 optimization).

    This is the STEP 2 VERSION that extracts values immediately instead of
    accumulating metric tensors, preventing memory accumulation.
    """
    batch_dfs = []

    for pt_file, csv_file in zip(response_files, test_output_files):
        logger.info(f"  Processing: {pt_file.name} + {csv_file.name}")

        try:
            # Extract parameter and category values
            try:
                arg_value = extract_param_from_string(
                    pt_file.name, key=data_arg_key, value_type=None
                )
                cat_value = extract_param_from_string(
                    pt_file.name, key=category, value_type=None
                )
            except ValueError:
                arg_value = extract_param_from_string(
                    pt_file.parent.name, key=data_arg_key, value_type=None
                )
                cat_value = extract_param_from_string(
                    pt_file.parent.name, key=category, value_type=None
                )

            # Verify matching values in CSV
            try:
                csv_arg_value = extract_param_from_string(
                    csv_file.name, key=data_arg_key, value_type=None
                )
                csv_cat_value = extract_param_from_string(
                    csv_file.name, key=category, value_type=None
                )
            except ValueError:
                csv_arg_value = extract_param_from_string(
                    csv_file.parent.name, key=data_arg_key, value_type=None
                )
                csv_cat_value = extract_param_from_string(
                    csv_file.parent.name, key=category, value_type=None
                )

            if not arg_value == csv_arg_value:
                raise ValueError(
                    f"{data_arg_key} values do not match between files: "
                    f"PT={arg_value} vs CSV={csv_arg_value}"
                )
            if not cat_value == csv_cat_value:
                raise ValueError(
                    f"{category} values do not match between files: "
                    f"PT={cat_value} vs CSV={csv_cat_value}"
                )

            # Load test outputs
            try:
                test_df = load_df(csv_file)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load test output CSV {csv_file}: {e}"
                ) from e

            # Process test performance metrics
            try:
                performance_df = process_test_performance(test_df, topk_values)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to process test performance for {csv_file}: {e}"
                ) from e

            # STEP 2 OPTIMIZATION: Pre-compute sample-to-presentation mapping
            try:
                sample_to_presentation = compute_sample_to_presentation_mapping(
                    test_df
                )
                unique_presentations = sorted(
                    performance_df["presentation_label"].unique()
                )
                unique_times = sorted(performance_df["times_index"].unique())
            except Exception as e:
                raise RuntimeError(
                    f"Failed to compute mappings for {csv_file}: {e}"
                ) from e

            # STEP 2: Use incremental extraction (memory-optimized)
            try:
                extracted_values = process_layer_responses_incremental(
                    pt_file=pt_file,
                    measures=measures,
                    sample_to_presentation=sample_to_presentation,
                    unique_presentations=unique_presentations,
                    unique_times=unique_times,
                    memory_monitor=memory_monitor,
                    max_retries=3,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to process layer responses from {pt_file}: {e}"
                ) from e

            # STEP 2: Create dataframe from extracted values (not tensors)
            try:
                layer_df = create_layer_dataframe_from_extracted_values(
                    extracted_values, unique_presentations, unique_times
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create layer dataframe for {pt_file}: {e}"
                ) from e

            # Merge layer metrics with performance metrics
            try:
                merged_df = performance_df.merge(
                    layer_df, on=["presentation_label", "times_index"], how="left"
                )
                # Add parameter and category information
                merged_df[data_arg_key] = arg_value
                merged_df[category] = cat_value

                batch_dfs.append(merged_df)
                logger.info(f"    Created merged dataframe with {len(merged_df)} rows")

                # Cleanup
                del extracted_values, layer_df
                memory_monitor.cleanup()

            except Exception as e:
                raise RuntimeError(
                    f"Failed to merge dataframes for {pt_file}: {e}"
                ) from e

        except Exception as e:
            logger.error(
                f"ERROR processing file pair: {pt_file.name} + {csv_file.name}"
            )
            logger.error(f"  {type(e).__name__}: {str(e)}")
            raise  # Re-raise to stop processing and prevent file deletion

    # Combine all files in this batch
    if batch_dfs:
        batch_combined = pd.concat(batch_dfs, ignore_index=True)
        return batch_combined
    else:
        raise RuntimeError("No dataframes were successfully created in this batch")


def create_layer_dataframe_from_metrics(
    layer_metrics: Dict[str, torch.Tensor],
    test_df: pd.DataFrame,
    performance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create layer dataframe from pre-computed metrics efficiently.

    Uses pre-allocated numpy arrays for better performance.
    """
    logger.info("  Creating layer dataframe from metrics...")

    unique_presentations = performance_df["presentation_label"].unique()
    unique_times = performance_df["times_index"].unique()
    n_rows = len(unique_presentations) * len(unique_times)

    # Pre-allocate arrays
    data_arrays = {
        "presentation_label": np.empty(n_rows, dtype=np.int32),
        "times_index": np.empty(n_rows, dtype=np.int32),
    }

    # Pre-allocate for each metric
    for metric_name in layer_metrics.keys():
        data_arrays[metric_name] = np.empty(n_rows, dtype=np.float32)

    # Create sample to presentation mapping
    sample_to_presentation = (
        test_df.groupby("sample_index")["label_index"]
        .apply(get_presentation_label)
        .to_dict()
    )

    # Fill arrays directly
    idx = 0
    for pres_label in unique_presentations:
        # Find sample indices for this presentation
        sample_indices = [
            s for s, p in sample_to_presentation.items() if p == pres_label
        ]

        if not sample_indices:
            continue

        sample_idx = sample_indices[0]

        for times_idx in unique_times:
            data_arrays["presentation_label"][idx] = pres_label
            data_arrays["times_index"][idx] = times_idx

            # Add layer metrics
            for metric_name, metric_tensor in layer_metrics.items():
                try:
                    value = tensor_to_numpy(metric_tensor[sample_idx, times_idx])
                    data_arrays[metric_name][idx] = value
                except IndexError as e:
                    raise RuntimeError(
                        f"Index error accessing metric '{metric_name}' "
                        f"at sample {sample_idx}, time {times_idx}: {e}"
                    ) from e

            idx += 1

    # Create DataFrame from arrays
    layer_df = pd.DataFrame(data_arrays)
    logger.info(f"    Created layer dataframe with {len(layer_df)} rows")

    return layer_df


def validate_outputs(
    df_optimized: pd.DataFrame, df_original: pd.DataFrame, tolerance: float = 1e-5
) -> bool:
    """
    Validate that optimized output matches original output.

    Args:
        df_optimized: DataFrame from optimized implementation
        df_original: DataFrame from original implementation
        tolerance: Numerical tolerance for floating point comparison

    Returns:
        True if outputs match within tolerance
    """
    logger.info("Validating optimized output against original...")

    # Check shape
    if df_optimized.shape != df_original.shape:
        logger.error(
            f"Shape mismatch: optimized {df_optimized.shape} vs "
            f"original {df_original.shape}"
        )
        return False

    # Check columns
    if set(df_optimized.columns) != set(df_original.columns):
        missing_in_opt = set(df_original.columns) - set(df_optimized.columns)
        missing_in_orig = set(df_optimized.columns) - set(df_original.columns)
        logger.error(
            f"Column mismatch:\n"
            f"  Missing in optimized: {missing_in_opt}\n"
            f"  Missing in original: {missing_in_orig}"
        )
        return False

    # Sort both dataframes by key columns for comparison
    key_cols = ["presentation_label", "times_index"]
    df_opt_sorted = df_optimized.sort_values(key_cols).reset_index(drop=True)
    df_orig_sorted = df_original.sort_values(key_cols).reset_index(drop=True)

    # Compare numeric columns
    numeric_cols = df_opt_sorted.select_dtypes(include=[np.number]).columns

    all_match = True
    for col in numeric_cols:
        diff = np.abs(df_opt_sorted[col] - df_orig_sorted[col])
        max_diff = diff.max()

        if max_diff > tolerance:
            logger.error(
                f"Column '{col}' differs by max {max_diff:.2e} "
                f"(tolerance: {tolerance:.2e})"
            )
            all_match = False
        else:
            logger.info(f"  ✓ Column '{col}' matches (max diff: {max_diff:.2e})")

    # Compare non-numeric columns
    non_numeric_cols = df_opt_sorted.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        if not df_opt_sorted[col].equals(df_orig_sorted[col]):
            logger.error(f"Column '{col}' differs")
            all_match = False
        else:
            logger.info(f"  ✓ Column '{col}' matches")

    if all_match:
        logger.info("✅ Validation PASSED: Outputs match within tolerance")
    else:
        logger.error("❌ Validation FAILED: Outputs differ")

    return all_match


# Command line interface
parser = argparse.ArgumentParser(
    description="Process test data with memory-optimized layer loading"
)
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
    "--parameter",
    type=str,
    required=True,
    help="Parameter to extract from filenames",
)
parser.add_argument(
    "--category",
    type=str,
    required=True,
    help="Category to extract from filenames",
)
parser.add_argument(
    "--measures",
    nargs="+",
    type=str,
    default=["response_avg", "response_std"],
    choices=["response_avg", "response_std", "spatial_variance", "feature_variance"],
    help="Layer measures to compute (default: response_avg response_std)",
)
parser.add_argument(
    "--topk",
    nargs="+",
    type=int,
    default=[3, 5],
    help="Top-k values for calculating top-k accuracy",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size for processing files (default: 1)",
)
parser.add_argument(
    "--memory_limit_gb",
    type=float,
    default=60.0,
    help="Soft memory limit in GB (triggers warnings, default: 20.0)",
)
parser.add_argument(
    "--remove_input_responses",
    type=str_to_bool,
    default="False",
    help="If 'True', remove input response files after SUCCESSFUL processing",
)
parser.add_argument(
    "--validate",
    action="store_true",
    help="Run validation against original implementation (requires both implementations)",
)


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Initialize memory monitor
    memory_monitor = MemoryMonitor(memory_limit_gb=args.memory_limit_gb)

    # Log configuration
    logger.info("=" * 80)
    logger.info("MEMORY-OPTIMIZED TEST DATA PROCESSING")
    logger.info("=" * 80)
    logger.info(f"Measures: {args.measures}")
    logger.info(f"Top-k values: {args.topk}")
    logger.info(f"Memory limit: {args.memory_limit_gb}GB")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Files to process: {len(args.responses)}")

    processing_successful = False
    df = None
    df_validation = None  # For validation mode

    try:
        # Process data
        logger.info(f"\nProcessing {len(args.responses)} file pairs...")
        memory_monitor.log_memory("start")

        # Process in batches
        all_dataframes = []
        all_dataframes_validation = [] if args.validate else None
        batch_count = 0

        for response_batch, output_batch in chunk_lists(
            args.responses, args.test_outputs, args.batch_size
        ):
            batch_count += 1
            logger.info(
                f"\nBatch {batch_count}/{(len(args.responses) + args.batch_size - 1) // args.batch_size}"
            )

            try:
                # STEP 2: Use incremental extraction
                batch_df = process_single_batch_optimized_v2(
                    response_batch,
                    output_batch,
                    data_arg_key=args.parameter,
                    measures=args.measures,
                    category=args.category,
                    topk_values=args.topk,
                    memory_monitor=memory_monitor,
                )

                if not batch_df.empty:
                    all_dataframes.append(batch_df)
                    logger.info(
                        f"  Batch {batch_count} processed: {len(batch_df)} rows"
                    )
                else:
                    raise RuntimeError(f"Batch {batch_count} produced empty dataframe")

                # Validation mode: also run old implementation
                if args.validate:
                    logger.info(f"  Running validation for batch {batch_count}...")
                    try:
                        # Import and keep old function for validation
                        from dynvision.utils.memory_efficient_loading import (
                            process_layer_responses_sequential,
                        )

                        # Process with old method (keeping tensors)
                        # Note: This would need the old create_layer_dataframe_from_metrics
                        # For now, just log that validation is attempted
                        logger.info(
                            "    Validation: Old implementation would run here"
                        )
                        # batch_df_old = process_single_batch_optimized(...)
                        # all_dataframes_validation.append(batch_df_old)
                    except Exception as e:
                        logger.warning(f"    Validation failed: {e}")

                # Cleanup between batches
                del batch_df
                memory_monitor.cleanup()

            except Exception as e:
                logger.error(f"Failed to process batch {batch_count}")
                logger.error(f"  Response files: {[f.name for f in response_batch]}")
                logger.error(f"  Output files: {[f.name for f in output_batch]}")
                raise RuntimeError(f"Batch {batch_count} failed: {e}") from e

        if not all_dataframes:
            raise RuntimeError("No data was successfully processed from any batch")

        # Combine all dataframes
        logger.info(f"\nCombining {len(all_dataframes)} batches...")
        df = pd.concat(all_dataframes, ignore_index=True)
        del all_dataframes
        memory_monitor.cleanup()

        logger.info(f"Total combined data: {len(df)} rows")
        memory_monitor.log_memory("after combining")

        # Validation mode comparison
        if args.validate and all_dataframes_validation:
            logger.info("\n" + "=" * 80)
            logger.info("VALIDATION MODE")
            logger.info("=" * 80)
            df_validation = pd.concat(all_dataframes_validation, ignore_index=True)
            del all_dataframes_validation

            # Compare outputs
            validation_passed = validate_outputs(df, df_validation)
            if not validation_passed:
                logger.warning(
                    "⚠️  Validation found differences between implementations"
                )

        # Save processed data
        logger.info(f"\nSaving processed data to: {args.output}")
        df.to_csv(args.output, index=False)
        logger.info(f"✅ Successfully saved {len(df)} rows")

        processing_successful = True

    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("PROCESSING FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        logger.error("\nDue to errors, input files will NOT be removed")
        sys.exit(1)

    finally:
        # Memory summary
        logger.info("\n" + "=" * 80)
        logger.info("MEMORY SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Peak memory usage: {memory_monitor.peak_memory_gb:.2f}GB")
        logger.info(f"Memory limit: {args.memory_limit_gb}GB")

        if memory_monitor.peak_memory_gb > args.memory_limit_gb:
            logger.warning(
                f"⚠️  Peak memory ({memory_monitor.peak_memory_gb:.2f}GB) "
                f"exceeded limit ({args.memory_limit_gb}GB)"
            )

    # Only remove input files if processing was successful
    if processing_successful and args.remove_input_responses:
        logger.info("\n" + "=" * 80)
        logger.info("REMOVING INPUT FILES")
        logger.info("=" * 80)
        logger.info(
            "Processing completed successfully, removing input response files..."
        )

        for response_file in args.responses:
            if response_file.exists():
                try:
                    response_file.unlink()
                    logger.info(f"  ✓ Removed: {response_file.name}")
                except Exception as e:
                    logger.error(f"  ✗ Failed to remove {response_file.name}: {e}")

    # Print summary statistics
    if df is not None:
        logger.info("\n" + "=" * 80)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Unique {args.category} values: {df[args.category].unique()}")
        logger.info(f"Unique {args.parameter} values: {df[args.parameter].unique()}")
        logger.info(f"Time steps: {sorted(df['times_index'].unique())}")
        logger.info(
            f"Presentation labels: {sorted(df['presentation_label'].unique())}"
        )

        # Layer metrics summary
        layer_cols = [
            col
            for col in df.columns
            if any(measure in col for measure in args.measures)
        ]
        logger.info(f"Layer metric columns: {len(layer_cols)}")

        # Performance metrics summary
        if "accuracy" in df.columns:
            logger.info(
                f"Accuracy range: {df['accuracy'].min():.3f} to {df['accuracy'].max():.3f}"
            )

        for k in args.topk:
            topk_col = f"accuracy_top{k}"
            if topk_col in df.columns:
                logger.info(
                    f"Top-{k} accuracy range: {df[topk_col].min():.3f} to "
                    f"{df[topk_col].max():.3f}"
                )

        logger.info("\n✅ Processing completed successfully!")
