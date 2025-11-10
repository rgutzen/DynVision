"""Process test data by combining layer responses and test performance metrics.

MEMORY-OPTIMIZED VERSION: Processes layers sequentially to handle large (30GB+) files.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dynvision.utils import str_to_bool
from dynvision.utils.visualization_utils import (
    layer_response_avg,
    layer_response_std,
    spatial_variance,
    feature_variance,
    tensor_to_numpy,
    extract_param_from_string,
)
from dynvision.utils.data_utils import load_df
from dynvision.utils.performance_measures import (
    calculate_topk_accuracy,
    calculate_confidence,
)

# Import memory-efficient loading system
from dynvision.utils.memory_efficient_loading import (
    MemoryMonitor,
    process_layer_responses_incremental,  # STEP 2: Incremental extraction
)

# Configure logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class MeasureConfig:
    layer_measures: List[str]
    confidence_measures: List[str]
    accuracy_topk: List[int]
    classifier_topk: List[int]


def build_measure_config(measures: List[str]) -> MeasureConfig:
    """Categorize measure strings into layer, confidence, top-k accuracy, and classifier groups."""

    layer_measure_names = {
        "response_avg",
        "response_std",
        "spatial_variance",
        "feature_variance",
    }
    confidence_measure_names = {
        "guess_confidence",
        "label_confidence",
        "first_label_confidence",
    }

    layer_measures: List[str] = []
    confidence_measures: List[str] = []
    accuracy_topk: List[int] = []
    classifier_topk: List[int] = []

    for measure in measures:
        if measure in layer_measure_names:
            layer_measures.append(measure)
        elif measure in confidence_measure_names:
            confidence_measures.append(measure)
        elif measure.startswith("accuracy_top"):
            value_str = measure.replace("accuracy_top", "", 1)
            if not value_str.isdigit():
                raise ValueError(f"Invalid accuracy measure '{measure}'")
            value = int(value_str)
            if value <= 0:
                raise ValueError(f"Top-k accuracy must be positive, got '{measure}'")
            accuracy_topk.append(value)
        elif measure.startswith("classifier_top"):
            value_str = measure.replace("classifier_top", "", 1)
            if not value_str.isdigit():
                raise ValueError(f"Invalid classifier measure '{measure}'")
            value = int(value_str)
            if value <= 0:
                raise ValueError(f"Classifier top-k must be positive, got '{measure}'")
            classifier_topk.append(value)
        else:
            raise ValueError(f"Unsupported measure '{measure}'")

    # Deduplicate while preserving natural ordering for deterministic output
    layer_measures = list(dict.fromkeys(layer_measures))
    confidence_measures = list(dict.fromkeys(confidence_measures))
    accuracy_topk = sorted(set(accuracy_topk))
    classifier_topk = sorted(set(classifier_topk))

    return MeasureConfig(
        layer_measures=layer_measures,
        confidence_measures=confidence_measures,
        accuracy_topk=accuracy_topk,
        classifier_topk=classifier_topk,
    )


def chunk_lists(lst1, lst2, chunk_size):
    """Split two lists into chunks of specified size."""
    # Ensure both lists are the same length
    for i in range(0, len(lst1), chunk_size):
        yield lst1[i : i + chunk_size], lst2[i : i + chunk_size]


def calculate_confidence_and_topk_from_classifier(
    classifier_responses: torch.Tensor,
    label_index: torch.Tensor,
    sample_index: torch.Tensor = None,
    times_index: torch.Tensor = None,
    guess_index: torch.Tensor = None,
    first_label_index: torch.Tensor = None,
    topk_values: Optional[List[int]] = None,
    confidence_measures: List[str] = [
        "guess_confidence",
        "label_confidence",
        "first_label_confidence",
    ],
) -> pd.DataFrame:
    """
    Calculate all confidence measures and top-k accuracies from classifier responses.

    Returns a DataFrame with (sample_index, times_index) as index and confidence/topk columns.

    Args:
        classifier_responses: Tensor of shape (n_samples, n_timesteps, n_classes)
        label_index: Ground truth labels (n_samples, n_timesteps)
        guess_index: Model predictions (n_samples, n_timesteps)
        first_label_index: First valid label per sample (n_samples,)
        topk_values: List of k values for top-k accuracy

    Returns:
        DataFrame with columns: sample_index, times_index, guess_confidence,
        label_confidence, first_label_confidence, accuracy_topK (for each K)
    """
    n_samples, n_timesteps, n_classes = classifier_responses.shape
    if sample_index is None:
        sample_index = np.repeat(np.arange(n_samples), n_timesteps)
    if times_index is None:
        times_index = np.tile(np.arange(n_timesteps), n_samples)

    topk_values = topk_values or []

    # Apply softmax once to get probabilities when needed
    probabilities = (
        torch.softmax(classifier_responses, dim=-1) if topk_values else None
    )

    # ===== CONFIDENCE CALCULATIONS =====
    confidence_results = {}

    if "guess_confidence" in confidence_measures and guess_index is not None:
        confidence_results["guess_confidence"] = (
            calculate_confidence(classifier_responses, guess_index).cpu().numpy()
        )

    if "label_confidence" in confidence_measures:
        confidence_results["label_confidence"] = (
            calculate_confidence(classifier_responses, label_index).cpu().numpy()
        )

    if (
        "first_label_confidence" in confidence_measures
        and first_label_index is not None
    ):
        confidence_results["first_label_confidence"] = (
            calculate_confidence(classifier_responses, first_label_index).cpu().numpy()
        )

    # ===== TOP-K ACCURACY =====
    topk_results = {}
    if topk_values:
        accuracy_label = (
            label_index if first_label_index is None else first_label_index
        )
        topk_accuracy = calculate_topk_accuracy(
            probabilities, accuracy_label, k=topk_values
        )
        topk_results = {
            f"accuracy_top{k}": acc.cpu().numpy()
            for k, acc in zip(topk_values, topk_accuracy)
        }

    # ===== BUILD DATAFRAME =====
    # Create multi-index for sample and time
    data = {
        "sample_index": sample_index,
        "times_index": times_index,
    }
    # Add confidence columns
    for measure, conf_array in confidence_results.items():
        data[measure] = conf_array.ravel()
    # Add top-k accuracy columns
    for k, acc_array in topk_results.items():
        data[k] = acc_array.ravel()

    return pd.DataFrame(data)


def compress_to_presentation_level(
    df: pd.DataFrame,
    id_cols: List[str] = ["first_label_index", "times_index"],
    drop_cols: List[str] = ["sample_index", "image_index"],
) -> pd.DataFrame:
    """
    Compress dataframe from sample-level to presentation-level resolution.

    For each (first_label_index, times_index) group:
    - If all values identical: keep single value
    - If values vary: create <col>_avg and <col>_std

    Args:
        df: Input dataframe at sample-level
        id_cols: Grouping columns (default: first_label_index, times_index)
        drop_cols: Columns to discard

    Returns:
        Compressed dataframe at presentation-level
    """
    logger.info(f"    Compressing from {len(df)} rows to presentation-level...")

    aggregated = {}

    for col in df.columns:
        if col in id_cols:
            continue
        if col in drop_cols:
            continue

        grouped = df.groupby(id_cols)[col]

        # Check if all non-NaN values are identical within each group
        # For each group, check if nunique (excluding NaN) is <= 1
        nunique_per_group = grouped.nunique()
        all_identical = (nunique_per_group <= 1).all()

        if all_identical:
            # Keep single value
            aggregated[col] = grouped.first()
            logger.debug(f"      {col}: identical values, keeping single column")
        else:
            # Compute mean and std
            aggregated[f"{col}_avg"] = grouped.mean()
            aggregated[f"{col}_std"] = grouped.std()
            logger.debug(
                f"      {col}: varying values, creating _avg and _std columns"
            )

    result = pd.DataFrame(aggregated).reset_index()
    logger.info(
        f"    Compressed to {len(result)} rows with {len(result.columns)} columns"
    )

    return result


def process_test_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process test performance CSV and add first_label_index.

    Returns DataFrame at (sample_index, times_index) resolution with:
    - All original CSV columns
    - first_label_index: first valid label for each sample
    - accuracy: boolean (guess == first_label)
    """
    logger.info("  Processing test performance metrics...")

    # Verify required columns
    required_cols = ["sample_index", "times_index", "label_index", "guess_index"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available: {df.columns.tolist()}"
        )

    logger.info(f"    Input shape: {df.shape}")
    logger.info(
        f"    Samples: {df['sample_index'].nunique()}, Times: {df['times_index'].nunique()}"
    )

    # Ensure one row per (sample, time)
    df = df.groupby(["sample_index", "times_index"]).first().reset_index()

    # Add first_label_index for each sample
    first_labels = (
        df[df["label_index"] >= 0].groupby("sample_index")["label_index"].first()
    )
    df["first_label_index"] = (
        df["sample_index"].map(first_labels).fillna(-1).astype(int)
    )

    # Calculate accuracy (1 if guess matches first label, 0 otherwise, NaN if no valid label)
    df["accuracy"] = (df["guess_index"] == df["first_label_index"]).astype(float)
    df.loc[df["first_label_index"] < 0, "accuracy"] = np.nan

    logger.info(f"    Output shape: {df.shape}")
    logger.info(f"    Unique first labels: {df['first_label_index'].unique()}")

    return df


def process_single_batch_optimized(
    response_files: List[Path],
    test_output_files: List[Path],
    data_arg_key: str,
    measure_config: MeasureConfig,
    category: str,
    memory_monitor: MemoryMonitor,
    resolution: str = "sample",
) -> pd.DataFrame:
    """
    Process a single batch of files.

    Workflow:
    1. Load and process test CSV (adds first_label_index, accuracy)
    2. Load response PT file and calculate confidence/topk metrics from classifier
    3. Calculate layer response metrics
    4. Merge all metrics into single DataFrame at (sample_index, times_index) resolution

    Returns:
        DataFrame with all metrics at (sample_index, times_index, first_label_index) resolution
    """
    batch_dfs = []

    layer_measures = measure_config.layer_measures
    confidence_measures = measure_config.confidence_measures
    accuracy_topk_values = measure_config.accuracy_topk
    classifier_topk_values = measure_config.classifier_topk

    for pt_file, csv_file in zip(response_files, test_output_files):
        logger.info(f"  Processing: {pt_file.name} + {csv_file.name}")

        try:
            # Extract metadata from filenames
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

            # Verify CSV has matching metadata
            try:
                csv_arg = extract_param_from_string(
                    csv_file.name, key=data_arg_key, value_type=None
                )
                csv_cat = extract_param_from_string(
                    csv_file.name, key=category, value_type=None
                )
            except ValueError:
                csv_arg = extract_param_from_string(
                    csv_file.parent.name, key=data_arg_key, value_type=None
                )
                csv_cat = extract_param_from_string(
                    csv_file.parent.name, key=category, value_type=None
                )

            if arg_value != csv_arg or cat_value != csv_cat:
                raise ValueError(f"Metadata mismatch between PT and CSV files")

            # ===== STEP 1: Load and process test CSV =====
            test_df = load_df(csv_file)
            test_df_with_first_label = process_test_performance(test_df)

            # ===== STEP 2: Load responses and calculate classifier metrics =====
            try:
                responses = torch.load(pt_file, map_location="cpu", weights_only=True)
                has_responses = bool(responses) and len(responses) > 0
            except Exception as e:
                logger.warning(f"Failed to load responses: {e}")
                has_responses = False

            # Work at sample level first, then compress
            sample_level_df = test_df_with_first_label.copy()

            if not has_responses:
                logger.warning(f"  Empty response file, skipping response metrics")
            elif "classifier" in responses and (
                confidence_measures or accuracy_topk_values or classifier_topk_values
            ):
                logger.info(f"  Calculating classifier metrics...")

                try:
                    # Get dimensions
                    n_samples_csv = test_df_with_first_label["sample_index"].nunique()
                    n_timesteps = test_df_with_first_label["times_index"].nunique()
                    classifier_tensor = responses["classifier"]
                    n_samples_pt = classifier_tensor.shape[0]
                    n_units = classifier_tensor.shape[-1]

                    # Validate sample sizes
                    if n_samples_pt > n_samples_csv:
                        raise ValueError(
                            f"PT file has MORE samples ({n_samples_pt}) than CSV ({n_samples_csv}). "
                            f"This should not happen!"
                        )

                    # Log warning if mismatch
                    if n_samples_pt < n_samples_csv:
                        logger.warning(f"⚠️  SAMPLE SIZE MISMATCH detected!")
                        logger.warning(f"    CSV file: {n_samples_csv} samples")
                        logger.warning(f"    PT file:  {n_samples_pt} samples")
                        logger.warning(
                            f"    Confidence metrics will be NaN for {n_samples_csv - n_samples_pt} samples "
                            f"(sample_index >= {n_samples_pt})."
                        )
                        logger.warning(
                            f"    Top-k accuracies will be calculated from available samples only."
                        )

                    # Filter confidence measures that are already in CSV
                    confidence_measures_needed = [
                        m
                        for m in confidence_measures
                        if m not in test_df_with_first_label.columns
                    ]

                    # Filter accuracy top-k requests that exceed classifier dimensionality
                    valid_accuracy_topk: List[int] = []
                    if accuracy_topk_values:
                        valid_accuracy_topk = [
                            k for k in accuracy_topk_values if k <= n_units
                        ]
                        invalid_accuracy_topk = sorted(
                            set(accuracy_topk_values) - set(valid_accuracy_topk)
                        )
                        if invalid_accuracy_topk:
                            logger.warning(
                                "    Requested accuracy_topk values %s exceed available classifier units (%d)",
                                invalid_accuracy_topk,
                                n_units,
                            )

                    # Filter classifier top-k requests similarly
                    valid_classifier_topk: List[int] = []
                    if classifier_topk_values:
                        valid_classifier_topk = [
                            k for k in classifier_topk_values if k <= n_units
                        ]
                        invalid_classifier_topk = sorted(
                            set(classifier_topk_values) - set(valid_classifier_topk)
                        )
                        if invalid_classifier_topk:
                            logger.warning(
                                "    Requested classifier_topk values %s exceed available classifier units (%d)",
                                invalid_classifier_topk,
                                n_units,
                            )

                    # Prepare tensors for available samples only
                    if n_samples_pt > 0:
                        sorted_df = test_df_with_first_label.sort_values(
                            ["sample_index", "times_index"]
                        )
                        sorted_df_available = sorted_df[
                            sorted_df["sample_index"] < n_samples_pt
                        ]

                        if len(sorted_df_available) > 0:
                            guess_tensor = torch.tensor(
                                sorted_df_available["guess_index"].values.reshape(
                                    n_samples_pt, n_timesteps
                                ),
                                dtype=torch.int,
                            )
                            label_tensor = torch.tensor(
                                sorted_df_available["label_index"].values.reshape(
                                    n_samples_pt, n_timesteps
                                ),
                                dtype=torch.int,
                            )
                            first_label_tensor = torch.tensor(
                                sorted_df_available[
                                    "first_label_index"
                                ].values.reshape(n_samples_pt, n_timesteps),
                                dtype=torch.int,
                            )

                            # Calculate classifier metrics at sample level
                            classifier_df = (
                                calculate_confidence_and_topk_from_classifier(
                                    classifier_tensor,
                                    label_index=label_tensor,
                                    guess_index=guess_tensor,
                                    first_label_index=first_label_tensor,
                                    topk_values=valid_accuracy_topk,
                                    confidence_measures=confidence_measures_needed,
                                )
                            )

                            # Merge back to full sample_level_df (creates NaN for missing samples)
                            sample_level_df = sample_level_df.merge(
                                classifier_df,
                                on=["sample_index", "times_index"],
                                how="left",
                            )

                            logger.info(
                                f"    Added {len(classifier_df.columns)-2} classifier metrics"
                            )

                            # Add classifier top-k activation columns
                            if valid_classifier_topk:
                                max_classifier_top = max(valid_classifier_topk)
                                overall_activity = classifier_tensor.sum(dim=(0, 1))
                                ranked_units = torch.argsort(
                                    overall_activity, descending=True
                                )
                                ranked_units = ranked_units[:max_classifier_top]

                                if ranked_units.numel() < max_classifier_top:
                                    logger.warning(
                                        "    Fewer classifier units (%d) than requested top-k (%d)",
                                        ranked_units.numel(),
                                        max_classifier_top,
                                    )

                                reordered_responses = classifier_tensor[
                                    :, :, ranked_units
                                ]
                                flattened_responses = reordered_responses.reshape(
                                    -1, ranked_units.numel()
                                )

                                if flattened_responses.shape[0] != len(
                                    sorted_df_available
                                ):
                                    logger.warning(
                                        "    Classifier activations shape mismatch (expected %d rows, got %d)",
                                        len(sorted_df_available),
                                        flattened_responses.shape[0],
                                    )
                                else:
                                    classifier_top_data = {
                                        "sample_index": sorted_df_available[
                                            "sample_index"
                                        ].values,
                                        "times_index": sorted_df_available[
                                            "times_index"
                                        ].values,
                                    }
                                    np_responses = flattened_responses.cpu().numpy()
                                    for idx in range(ranked_units.numel()):
                                        classifier_top_label = (
                                            f"classifier_top{idx + 1}"
                                        )
                                        unit_index = int(ranked_units[idx].item())
                                        classifier_top_data[classifier_top_label] = (
                                            np_responses[:, idx]
                                        )
                                        classifier_top_data[
                                            f"{classifier_top_label}_id"
                                        ] = np.full(
                                            np_responses.shape[0],
                                            unit_index,
                                            dtype=np.int64,
                                        )

                                    classifier_top_df = pd.DataFrame(
                                        classifier_top_data
                                    )
                                    sample_level_df = sample_level_df.merge(
                                        classifier_top_df,
                                        on=["sample_index", "times_index"],
                                        how="left",
                                    )

                                    logger.info(
                                        "    Added classifier top-k activations up to k=%d",
                                        max_classifier_top,
                                    )
                        else:
                            logger.warning(
                                f"    No valid samples to process for classifier metrics"
                            )
                    else:
                        logger.warning(
                            f"    PT file has no samples, skipping classifier metrics"
                        )

                except Exception as e:
                    logger.error(f"Failed to calculate classifier metrics: {e}")
                    raise

            # ===== STEP 3: Conditionally compress based on resolution =====
            if resolution == "class":
                logger.info(
                    f"  Compressing sample-level data to presentation-level..."
                )
                result_df = compress_to_presentation_level(
                    sample_level_df,
                    id_cols=["first_label_index", "times_index"],
                    drop_cols=["sample_index", "image_index"],
                )
            else:  # resolution == "sample"
                logger.info(f"  Keeping data at sample-level (no compression)...")
                result_df = sample_level_df.copy()

            # ===== STEP 4: Calculate layer response metrics =====
            if has_responses and layer_measures:
                logger.info(
                    f"  Calculating layer metrics at '{resolution}' resolution..."
                )

                try:
                    # Build sample-to-presentation mapping
                    sample_to_presentation = (
                        test_df_with_first_label.groupby("sample_index")[
                            "first_label_index"
                        ]
                        .first()
                        .to_dict()
                    )

                    presented_classes = sorted(result_df["first_label_index"].unique())
                    unique_times = sorted(result_df["times_index"].unique())

                    # Calculate expected sample count for sample resolution mode
                    n_samples_expected = None
                    if resolution == "sample":
                        n_samples_expected = len(sample_to_presentation)
                        logger.info(
                            f"    Expected samples for layer metrics: {n_samples_expected}"
                        )

                    # Use incremental layer processing with resolution parameter
                    layer_metrics = process_layer_responses_incremental(
                        pt_file=pt_file,
                        measures=layer_measures,
                        sample_to_presentation=sample_to_presentation,
                        presented_classes=presented_classes,
                        unique_times=unique_times,
                        memory_monitor=memory_monitor,
                        max_retries=3,
                        resolution=resolution,
                        n_samples_expected=n_samples_expected,
                    )

                    # Convert layer metrics to DataFrame based on resolution
                    if resolution == "class":
                        # Class resolution: (first_label_index, times_index)
                        n_rows = len(presented_classes) * len(unique_times)
                        layer_data = {
                            "first_label_index": np.repeat(
                                presented_classes, len(unique_times)
                            ),
                            "times_index": np.tile(
                                unique_times, len(presented_classes)
                            ),
                        }
                        layer_data.update(layer_metrics)
                        layer_df = pd.DataFrame(layer_data)

                        # Merge on (first_label_index, times_index)
                        result_df = result_df.merge(
                            layer_df,
                            on=["first_label_index", "times_index"],
                            how="left",
                        )
                    else:  # resolution == "sample"
                        # Sample resolution: (sample_index, times_index)
                        n_samples = result_df["sample_index"].nunique()
                        n_times = len(unique_times)
                        layer_data = {
                            "sample_index": np.repeat(np.arange(n_samples), n_times),
                            "times_index": np.tile(unique_times, n_samples),
                        }
                        layer_data.update(layer_metrics)
                        layer_df = pd.DataFrame(layer_data)

                        # Merge on (sample_index, times_index)
                        result_df = result_df.merge(
                            layer_df, on=["sample_index", "times_index"], how="left"
                        )

                    logger.info(f"    Added {len(layer_metrics)} layer metrics")

                except Exception as e:
                    logger.error(f"Failed to calculate layer metrics: {e}")
                    raise

            # ===== STEP 5: Add metadata columns =====
            result_df[data_arg_key] = arg_value
            result_df[category] = cat_value

            batch_dfs.append(result_df)
            logger.info(
                f"    Final shape: {result_df.shape} with {len(result_df.columns)} columns"
            )

            # Cleanup
            if has_responses:
                del responses
            memory_monitor.cleanup()

        except Exception as e:
            logger.error(f"ERROR: {pt_file.name} + {csv_file.name}")
            logger.error(f"  {type(e).__name__}: {str(e)}")
            raise

    # Combine all file results
    if not batch_dfs:
        raise RuntimeError("No dataframes created")

    combined = pd.concat(batch_dfs, ignore_index=True)
    logger.info(f"  Batch combined: {combined.shape}")

    return combined


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
    help=(
        "Measures to compute. Supported values include layer metrics (response_avg, "
        "response_std, spatial_variance, feature_variance), confidence measures "
        "(guess_confidence, label_confidence, first_label_confidence), "
        "top-k accuracies (accuracy_topK), and classifier activations "
        "(classifier_topK)."
    ),
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
parser.add_argument(
    "--sample_resolution",
    type=str,
    default="sample",
    choices=["sample", "class"],
    help="Resolution for output data: 'sample' for (sample_index, times_index), "
    "'class' for (first_label_index, times_index) with aggregation",
)


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Initialize measure configuration and memory monitor
    measure_config = build_measure_config(args.measures)
    memory_monitor = MemoryMonitor(memory_limit_gb=args.memory_limit_gb)

    # Log configuration
    logger.info("=" * 80)
    logger.info("MEMORY-OPTIMIZED TEST DATA PROCESSING")
    logger.info("=" * 80)
    logger.info(f"Resolution: {args.sample_resolution}")
    logger.info(f"Measures: {args.measures}")
    if measure_config.accuracy_topk:
        logger.info(f"Accuracy top-k measures: {measure_config.accuracy_topk}")
    if measure_config.classifier_topk:
        logger.info(f"Classifier top-k measures: {measure_config.classifier_topk}")
    logger.info(f"Memory limit: {args.memory_limit_gb}GB")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Files to process: {len(args.responses)}")
    if args.sample_resolution == "sample":
        logger.info("NOTE: Output at sample-level (no compression)")
    else:
        logger.info("NOTE: Output at class-level (with compression and aggregation)")
    logger.info("NOTE: New CSV format - classifier responses no longer in CSV")
    logger.info("NOTE: Empty response files will be handled gracefully")

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
                batch_df = process_single_batch_optimized(
                    response_batch,
                    output_batch,
                    data_arg_key=args.parameter,
                    measure_config=measure_config,
                    category=args.category,
                    memory_monitor=memory_monitor,
                    resolution=args.sample_resolution,
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
        logger.info(f"Resolution: {args.sample_resolution}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Unique {args.category} values: {df[args.category].unique()}")
        logger.info(f"Unique {args.parameter} values: {df[args.parameter].unique()}")
        logger.info(f"Time steps: {sorted(df['times_index'].unique())}")

        if args.sample_resolution == "sample" and "sample_index" in df.columns:
            logger.info(f"Samples: {df['sample_index'].nunique()}")

        logger.info(f"Presentation labels: {sorted(df['first_label_index'].unique())}")

        # Layer metrics summary
        layer_cols = [
            col
            for col in df.columns
            if any(pattern in col for pattern in measure_config.layer_measures)
        ]
        if measure_config.classifier_topk:
            layer_cols.extend(
                [col for col in df.columns if col.startswith("classifier_top")]
            )
        layer_cols = sorted(set(layer_cols))
        logger.info(f"Layer metric columns: {len(layer_cols)}")

        # Performance metrics summary
        if "accuracy" in df.columns:
            logger.info(
                f"Accuracy range: {df['accuracy'].min():.3f} to {df['accuracy'].max():.3f}"
            )

        accuracy_columns = []
        for col in df.columns:
            if col.startswith("accuracy_top"):
                suffix = col.replace("accuracy_top", "", 1)
                if suffix.isdigit():
                    accuracy_columns.append((int(suffix), col))
        accuracy_columns.sort()
        for k, topk_col in accuracy_columns:
            min_val = df[topk_col].min(skipna=True)
            max_val = df[topk_col].max(skipna=True)
            logger.info(f"Top-{k} accuracy range: {min_val:.3f} to {max_val:.3f}")

        classifier_columns = []
        for col in df.columns:
            if col.startswith("classifier_top"):
                suffix = col.replace("classifier_top", "", 1)
                if suffix.isdigit():
                    classifier_columns.append((int(suffix), col))
        classifier_columns.sort()
        for rank, classifier_col in classifier_columns:
            min_val = df[classifier_col].min(skipna=True)
            max_val = df[classifier_col].max(skipna=True)
            logger.info(
                f"Classifier top{rank} activation range: {min_val:.3f} to {max_val:.3f}"
            )

        logger.info("\n✅ Processing completed successfully!")
