"""
Load and process data from Jang et al. (2021).

Jang, H., McCormack, D., & Tong, F. (2021). Noise-trained deep neural networks
effectively predict human vision in a crowded task. PNAS, 118(28), e2101427118.

This module provides functions to extract human and CNN behavioral data from the
published .mat file.
"""

import scipy.io
import numpy as np
import pandas as pd
from dynvision.project_paths import project_paths


def _extract_structured_array(mat_struct):
    """
    Convert MATLAB structured array to pandas DataFrame.

    Parameters
    ----------
    mat_struct : numpy.ndarray
        MATLAB structured array with named fields

    Returns
    -------
    pd.DataFrame
        Converted data with columns from field names
    """
    field_names = mat_struct.dtype.names
    data_dict = {}

    for field in field_names:
        field_data = mat_struct[field].flatten()

        # Handle nested arrays
        if len(field_data) > 0 and isinstance(field_data[0], np.ndarray):
            data_dict[field] = [
                item.flatten()[0] if item.size == 1 else item.flatten()
                for item in field_data
            ]
        else:
            data_dict[field] = field_data

    return pd.DataFrame(data_dict)


def _extract_cnn_data(mat_struct):
    """
    Extract CNN data from MATLAB structured array.

    The CNN data has structure:
    - SSNR: (1, 20) - 20 test levels
    - category_id: (16, 1) - 16 categories
    - image_id: (16, 50) - 50 images per category
    - accuracy: (20, 16, 50) - accuracy for each SSNR × category × image

    Returns trial-level DataFrame.
    """
    model_name = mat_struct["model_name"].flatten()[0]
    ssnr_values = mat_struct["SSNR"].flatten()
    category_ids = mat_struct["category_id"].flatten()
    image_ids = mat_struct["image_id"]
    accuracy = mat_struct["accuracy"]
    ssnr_threshold = mat_struct["ssnr_threshold"].flatten()[0]

    rows = []
    for ssnr_idx, ssnr in enumerate(ssnr_values):
        for cat_idx, cat_id in enumerate(category_ids):
            for img_idx in range(image_ids.shape[1]):
                rows.append(
                    {
                        "model_name": model_name,
                        "SSNR": ssnr,
                        "category_id": cat_id,
                        "image_id": image_ids[cat_idx, img_idx],
                        "accuracy": accuracy[ssnr_idx, cat_idx, img_idx],
                        "ssnr_threshold": ssnr_threshold,
                    }
                )

    return pd.DataFrame(rows)


def _explode_trials(df, id_cols, array_cols, recompute_accuracy=False):
    """
    Explode trial-level arrays into long format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with array columns to explode
    id_cols : list
        Columns identifying the entity (subject_id, model_name, etc.)
    array_cols : list
        Columns containing arrays to explode (SSNR, accuracy, etc.)
    recompute_accuracy : bool, default=False
        If True, recompute accuracy from category_true and category_choice

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with one row per trial
    """
    rows = []

    for idx, row in df.iterrows():
        id_values = {col: row[col] for col in id_cols}

        # Verify all arrays have same length
        array_lengths = [len(row[col]) for col in array_cols]
        if len(set(array_lengths)) > 1:
            raise ValueError(
                f"Array columns have different lengths: {dict(zip(array_cols, array_lengths))}"
            )

        n_trials = array_lengths[0]

        # Create one row per trial
        for trial_idx in range(n_trials):
            trial_row = id_values.copy()
            trial_row["trial_idx"] = trial_idx
            for col in array_cols:
                value = row[col][trial_idx]
                # Flatten nested arrays for category fields
                if col in ["category_true", "category_choice"] and isinstance(
                    value, np.ndarray
                ):
                    value = value.flatten()[0] if value.size > 0 else value
                trial_row[col] = value
            rows.append(trial_row)

    result_df = pd.DataFrame(rows)

    # Recompute accuracy if requested
    if (
        recompute_accuracy
        and "category_true" in result_df.columns
        and "category_choice" in result_df.columns
    ):
        result_df["accuracy"] = (
            result_df["category_true"] == result_df["category_choice"]
        ).astype(float)

    return result_df


def load_jang_2021_data(noise_type="gaussian"):
    """
    Load Jang et al. (2021) human and CNN data.

    Parameters
    ----------
    noise_type : str, default='gaussian'
        Type of noise condition to load. Options: 'gaussian', 'fourier'

    Returns
    -------
    human_df : pd.DataFrame
        Trial-level human data with columns:
        - subject_id: subject identifier
        - trial_idx: trial index within subject
        - SSNR: signal-to-signal+noise ratio
        - image_path: path to stimulus image
        - category_true: true category ID
        - category_choice: chosen category ID
        - accuracy: binary accuracy (0 or 1)

    cnn_df : pd.DataFrame
        Trial-level CNN data with columns:
        - model_name: CNN architecture name
        - SSNR: signal-to-signal+noise ratio
        - category_id: category identifier
        - image_id: image identifier
        - accuracy: binary accuracy (0 or 1)
        - ssnr_threshold: SSNR threshold for this model

    Notes
    -----
    Data from: Jang, H., McCormack, D., & Tong, F. (2021). Noise-trained deep
    neural networks effectively predict human vision in a crowded task.
    PNAS, 118(28), e2101427118.
    """
    if noise_type not in ["gaussian", "fourier"]:
        raise ValueError(
            f"noise_type must be 'gaussian' or 'fourier', got '{noise_type}'"
        )

    # Load .mat file: todo: make cli arg
    data_path = (
        project_paths.data.external
        / "jang-et-al_2021"
        / "Jang-at-al_2021_Figures-2+4.mat"
    )
    data = scipy.io.loadmat(data_path)

    # Extract human data
    human_key = f"human_{noise_type}"
    human_gaussian = data[human_key]
    human_df_wide = _extract_structured_array(human_gaussian[0, :])

    # Explode to trial-level format and recompute accuracy
    # Note: The accuracy field in the .mat file is incorrect, so we recompute it
    # by comparing category_true with category_choice
    human_id_cols = ["subject_id"]
    human_array_cols = [
        "SSNR",
        "image_path",
        "category_true",
        "category_choice",
        "accuracy",
    ]
    human_df = _explode_trials(
        human_df_wide, human_id_cols, human_array_cols, recompute_accuracy=True
    )

    # Extract CNN data
    cnn_key = f"cnn_{noise_type}"
    cnn_gaussian = data[cnn_key]

    cnn_dfs = []
    for model_idx in range(cnn_gaussian.shape[1]):
        model_data = cnn_gaussian[0, model_idx]
        model_df = _extract_cnn_data(model_data)
        cnn_dfs.append(model_df)

    cnn_df = pd.concat(cnn_dfs, ignore_index=True)

    return human_df, cnn_df


def compute_psychometric_curves(human_df=None, cnn_df=None):
    """
    Compute psychometric curves (accuracy vs SSNR).

    Parameters
    ----------
    human_df : pd.DataFrame, optional
        Trial-level human data from load_jang_2021_data()
    cnn_df : pd.DataFrame, optional
        Trial-level CNN data from load_jang_2021_data()

    Returns
    -------
    human_curve : pd.DataFrame or None
        Human accuracy aggregated by SSNR with columns:
        - SSNR: signal-to-signal+noise ratio
        - accuracy_avg: mean accuracy (0-1 scale)
        - accuracy_std: standard deviation of accuracy
        - count: number of trials

    cnn_curves : pd.DataFrame or None
        CNN accuracy aggregated by model and SSNR with columns:
        - model_name: CNN architecture (includes "Average" for cross-model mean)
        - SSNR: signal-to-signal+noise ratio
        - accuracy_avg: mean accuracy (0-1 scale)
        - accuracy_std: standard deviation of accuracy (within-model for individual
          models, inter-model variability for "Average")
        - count: number of trials (or number of models for "Average")
    """
    human_curve = None
    cnn_curves = None

    if human_df is not None:
        human_curve = (
            human_df.groupby("SSNR")["accuracy"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "accuracy_avg", "std": "accuracy_std"})
        )

    if cnn_df is not None:
        cnn_curves = (
            cnn_df.groupby(["model_name", "SSNR"])["accuracy"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "accuracy_avg", "std": "accuracy_std"})
        )

        # Compute average across all models for each SSNR
        # accuracy_avg = mean of model accuracy_avg values
        # accuracy_std = std of model accuracy_avg values (inter-model variability)
        model_avg = (
            cnn_curves.groupby("SSNR")["accuracy_avg"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "accuracy_avg", "std": "accuracy_std"})
        )
        model_avg.insert(0, "model_name", "Average")

        # Concatenate average with individual models
        cnn_curves = pd.concat([cnn_curves, model_avg], ignore_index=True)

    return human_curve, cnn_curves


# Example usage
if __name__ == "__main__":
    print("Loading Jang et al. (2021) data...")
    human_df, cnn_df = load_jang_2021_data(noise_type="gaussian")

    print(
        f"\nHuman data: {human_df.shape[0]} trials, {human_df['subject_id'].nunique()} subjects"
    )
    print(
        f"CNN data: {cnn_df.shape[0]} trials, {cnn_df['model_name'].nunique()} models"
    )

    print("\n" + "=" * 80)
    print("Computing psychometric curves...")
    human_curve, cnn_curves = compute_psychometric_curves(human_df, cnn_df)

    print("\nHuman accuracy by SSNR:")
    print(human_curve.to_string(index=False))

    print("\nCNN models:")
    print(cnn_df["model_name"].unique())

    print("\nExample: AlexNet accuracy by SSNR:")
    alexnet = cnn_curves[cnn_curves["model_name"] == "AlexNet"]
    print(alexnet.to_string(index=False))

    print("\n" + "=" * 80)
    print("DataFrames available for plotting:")
    print("  - human_df, cnn_df: trial-level data")
    print("  - human_curve, cnn_curves: aggregated psychometric curves")
