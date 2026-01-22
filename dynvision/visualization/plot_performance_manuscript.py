"""Plot performance traces with category-2 and Jang et al. (2021) benchmarks.

This module extends plot_performance.py to create a 4-panel manuscript figure:
- Panel A: Performance traces over time (main models)
- Panel B: Max accuracy vs parameter (main models)
- Panel C: Max accuracy vs parameter (category-2 models with plasma palette)
- Panel D: Jang et al. (2021) human and DNN benchmarks
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

# Import from plot_performance (Panel A and B functions)
from dynvision.visualization.plot_performance import (
    FORMATTING,
    ERRORBAR_CONFIG,
    _coerce_optional_dimension,
    _resolve_measure_column,
    _append_suffix_to_label,
    _filter_data_for_cell,
    _plot_accuracy_panel_with_ffonly,
    _plot_peak_height_panel,
    _add_hue_legend,
)

# Import functions from plot_responses
from dynvision.visualization.plot_responses import (
    _get_colors_for_dimension,
    _get_dimension_key,
    _extract_dimension_values,
    _normalize_dimension,
    _standardize_category_value,
    _validate_dimensions as _validate_dimension_choices,
    _format_legend_label,
)

from dynvision.utils.visualization_utils import (
    calculate_label_indicator,
    get_display_name,
    load_config_from_args,
    save_plot,
)

# Import Jang data loading - place somewhere else?
from dynvision.data.jang_2021 import load_jang_2021_data, compute_psychometric_curves

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# Layout configuration - modified from plot_performance
MANUSCRIPT_LAYOUT = {
    # Figure dimensions
    "subplot_width": 3,  # Width of each performance subplot
    "subplot_height": 7.0,  # Height of each subplot
    "subplot_spacing_x": 0.3,  # Horizontal spacing between subplots
    "subplot_spacing_y": 0.7,  # Vertical spacing between rows
    "ssnr_panel_width": 4.0,  # Width of SSNR panel (Panel B)
    "ssnr_panel_spacing": 1,  # Spacing before SSNR panel
    "category2_panel_width": 4.0,  # Width of category-2 panel (Panel C)
    "category2_panel_spacing": 1.0,  # Spacing before category-2 panel
    "jang_panel_width": 4.0,  # Width of Jang panel (Panel D)
    "jang_panel_spacing": 1,  # Spacing before Jang panel
    # Margins
    "left_margin": 0.1,
    "right_margin": 0.0,
    "top_margin": 0.1,
    "bottom_margin": 0.0,
    # Title spacing
    "title_spacing": 0.08,
    # Panel letters
    "panel_letter_offset_x": -0.01,
    "panel_letter_offset_y": 0.03,
}


# Panel C: Category-2 performance comparison
def _plot_category2_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    row_key: Optional[str],
    row_value: Optional[str],
    subplot_key: str,
    subplot_values: List[str],
    category2_key: str,
    category2_values: List[str],
    colors: Dict[str, Any],
    config: Dict,
    dt: float,
    show_ylabel: bool = True,
    show_legend: bool = True,
    accuracy_cols: Optional[List[str]] = None,
    primary_accuracy: Optional[str] = None,
) -> None:
    """Plot category-2 max performance vs parameter (Panel C).

    Similar to _plot_peak_height_panel but for category-2 data
    with different hue dimension and color palette.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    data : pd.DataFrame
        Category-2 data
    row_key : str or None
        Row dimension key for filtering
    row_value : str or None
        Row value for filtering
    subplot_key : str
        Key for x-axis values (sweep parameter)
    subplot_values : list of str
        Values for x-axis
    category2_key : str
        Key for category-2 dimension (hue)
    category2_values : list of str
        Category-2 values for hue
    colors : dict
        Color mapping for category-2 values
    config : dict
        Visualization config
    dt : float
        Time resolution (ms)
    show_ylabel : bool
        Whether to show y-axis label
    show_legend : bool
        Whether to show legend
    accuracy_cols : list of str, optional
        Accuracy column names
    primary_accuracy : str, optional
        Primary accuracy measure
    """
    # Filter by row if needed
    if row_key and row_value:
        data_filtered = data[data[row_key] == _standardize_category_value(row_value)]
    else:
        data_filtered = data

    if len(data_filtered) == 0:
        ax.text(
            0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes
        )
        return

    # Determine accuracy measure
    accuracy_measure = (
        _resolve_measure_column(data_filtered, primary_accuracy)
        if primary_accuracy is not None
        else None
    )

    if not accuracy_measure:
        # Fallback to "accuracy" if available
        if "accuracy" in data_filtered.columns:
            accuracy_measure = "accuracy"
        else:
            ax.text(
                0.5,
                0.5,
                "No Performance Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

    # Standardize data columns for comparison
    if category2_key in data_filtered.columns:
        data_filtered = data_filtered.copy()
        data_filtered[category2_key] = data_filtered[category2_key].apply(
            _standardize_category_value
        )
    if subplot_key in data_filtered.columns:
        data_filtered[subplot_key] = data_filtered[subplot_key].apply(
            _standardize_category_value
        )

    # For each category-2 value and subplot value, compute max performance
    plot_data = []
    for cat2_val in category2_values:
        cat2_val_std = _standardize_category_value(cat2_val)
        for subplot_val in subplot_values:
            subplot_val_std = _standardize_category_value(subplot_val)

            # Filter data for this combination
            mask = (
                (data_filtered[category2_key] == cat2_val_std)
                & (data_filtered[subplot_key] == subplot_val_std)
            )
            subset = data_filtered[mask]

            if len(subset) == 0:
                continue

            # Convert subplot value to numeric for x-axis (same as Panel B)
            try:
                x_val = float(subplot_val)
            except (ValueError, TypeError):
                x_val = float(subplot_values.index(subplot_val))

            # Compute max performance per seed (if multiple seeds)
            if "seed_id" in subset.columns or "source_file" in subset.columns:
                seed_col = "seed_id" if "seed_id" in subset.columns else "source_file"
                max_per_seed = (
                    subset.groupby([seed_col, "times_index"])[accuracy_measure]
                    .mean()
                    .groupby(seed_col)
                    .max()
                )
                max_perf = max_per_seed.mean()
                max_perf_std = max_per_seed.std() if len(max_per_seed) > 1 else 0
            else:
                # Single seed
                max_perf = subset.groupby("times_index")[accuracy_measure].mean().max()
                max_perf_std = 0

            plot_data.append(
                {
                    "category2": cat2_val,
                    "x_val": x_val,
                    "subplot": subplot_val,
                    "max_performance": max_perf,
                    "std": max_perf_std,
                }
            )

    if not plot_data:
        ax.text(
            0.5,
            0.5,
            "No Data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    plot_df = pd.DataFrame(plot_data)

    # Plot each category-2 value
    for cat2_val in category2_values:
        cat2_data = plot_df[plot_df["category2"] == cat2_val]
        if len(cat2_data) == 0:
            continue

        color = colors.get(cat2_val, "gray")

        # Plot with error bars (x_val is already computed)
        ax.errorbar(
            cat2_data["x_val"],
            cat2_data["max_performance"],
            yerr=cat2_data["std"],
            color=color,
            marker="o",
            markersize=6,
            linewidth=2,
            linestyle="-",
            capsize=4,
            alpha=0.8,
            label=_format_legend_label(category2_key, cat2_val, config, dt),
        )

    # Format axes
    if show_ylabel:
        ax.set_ylabel(
            "Max Performance", fontsize=FORMATTING["fontsize_axis"], fontweight="bold"
        )
    else:
        ax.set_ylabel("")

    # Set x-axis label to subplot dimension name
    ax.set_xlabel(
        get_display_name(subplot_key, config), fontsize=FORMATTING["fontsize_axis"]
    )

    # Set x-ticks to match Panel B (fixed positions)
    x_tick_positions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    x_tick_labels = [
        _format_legend_label(subplot_key, str(val), config, dt)
        for val in x_tick_positions
    ]

    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, rotation=0, ha="center")

    ax.tick_params(labelsize=FORMATTING["fontsize_tick"])

    # Add legend with bold title and lines only (no markers)
    if show_legend and len(category2_values) > 0:
        legend = ax.legend(
            loc="upper left",
            frameon=False,
            fontsize=FORMATTING["fontsize_legend"],
            title=get_display_name(category2_key, config),
            title_fontsize=FORMATTING["fontsize_legend"],
            handlelength=2.5,  # Length of legend lines
            numpoints=1,
        )
        # Make legend title bold
        legend.get_title().set_fontweight("bold")
        # Remove markers from legend handles (only for Line2D objects)
        for handle in legend.legendHandles:
            if isinstance(handle, Line2D):
                handle.set_marker("")
                handle.set_linestyle("-")
                handle.set_linewidth(2)

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)


# Panel D: Jang et al. (2021) benchmarks
def _plot_jang_panel(
    ax: plt.Axes,
    jang_noise_type: str = "gaussian",
    show_ylabel: bool = True,
) -> None:
    """Plot Jang et al. (2021) human and DNN data only.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    jang_noise_type : str, default='gaussian'
        Noise type for Jang data ('gaussian' or 'fourier')
    show_ylabel : bool
        Whether to show y-axis label
    """
    try:
        logger.info(
            f"Loading Jang et al. (2021) {jang_noise_type} noise data for Panel C..."
        )
        human_df, cnn_df = load_jang_2021_data(noise_type=jang_noise_type)
        human_curve, cnn_curves = compute_psychometric_curves(human_df, cnn_df)

        # Plot human data
        human_df_plot = human_curve
        ax.errorbar(
            human_df_plot["SSNR"],
            human_df_plot["accuracy_avg"],
            yerr=human_df_plot["accuracy_std"] / np.sqrt(human_df_plot["count"]),
            color="black",
            linewidth=2.5,
            marker="o",
            markersize=8,
            linestyle="-",
            capsize=5,
            alpha=0.9,
            zorder=10,
            label="Human",
        )

        # Plot average DNN data
        avg_cnn = cnn_curves[cnn_curves["model_name"] == "Average"]
        ax.errorbar(
            avg_cnn["SSNR"],
            avg_cnn["accuracy_avg"],
            yerr=avg_cnn["accuracy_std"],
            color="gray",
            linewidth=2.0,
            marker="s",
            markersize=6,
            linestyle="--",
            capsize=4,
            alpha=0.8,
            zorder=9,
            label="DNNs",
        )

        # Formatting
        if show_ylabel:
            ax.set_ylabel(
                "Accuracy", fontsize=FORMATTING["fontsize_axis"], fontweight="bold"
            )
        else:
            ax.set_ylabel("")

        ax.set_xlabel("SSNR", fontsize=FORMATTING["fontsize_axis"])
        ax.tick_params(labelsize=FORMATTING["fontsize_tick"])

        legend = ax.legend(
            loc="upper left",
            frameon=False,
            fontsize=FORMATTING["fontsize_legend"],
            title="Jang et al. 2021",
            title_fontsize=FORMATTING["fontsize_legend"],
            handlelength=2.5,
            numpoints=1,
        )
        # Make legend title bold and remove markers from legend handles
        legend.get_title().set_fontweight("bold")
        for handle in legend.legendHandles:
            if isinstance(handle, Line2D):
                handle.set_marker("")
                handle.set_linestyle("-")
                handle.set_linewidth(2)

        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)

        logger.info("Successfully plotted Jang et al. data in Panel C")

    except Exception as e:
        logger.warning(f"Could not plot Jang et al. data: {e}")
        ax.text(
            0.5,
            0.5,
            "Jang et al. data\nunavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )


def _add_panel_letters(
    fig: plt.Figure,
    perf_axes: List[List[plt.Axes]],
    ssnr_axes: List[plt.Axes],
    category2_axes: List[plt.Axes],
    jang_axes: List[plt.Axes],
    layout: Dict,
) -> None:
    """Add panel letters A), B), C), D) to performance, parameter comparison, category-2, and Jang panels."""
    fmt = FORMATTING

    if not perf_axes or not perf_axes[0]:
        return

    # Panel A) - Above first performance subplot
    first_perf_ax = perf_axes[0][0]
    first_perf_ax.text(
        layout["panel_letter_offset_x"],
        1 + layout["panel_letter_offset_y"],
        "A)",
        fontsize=fmt["fontsize_panel_label"],
        fontweight="bold",
        ha="center",
        va="bottom",
        transform=first_perf_ax.transAxes,
    )

    # Panel B) - Above parameter comparison panel
    if ssnr_axes:
        ssnr_ax = ssnr_axes[0]
        ssnr_ax.text(
            layout["panel_letter_offset_x"],
            1 + layout["panel_letter_offset_y"],
            "B)",
            fontsize=fmt["fontsize_panel_label"],
            fontweight="bold",
            ha="center",
            va="bottom",
            transform=ssnr_ax.transAxes,
        )

    # Panel C) - Above category-2 panel
    if category2_axes:
        category2_ax = category2_axes[0]
        category2_ax.text(
            layout["panel_letter_offset_x"],
            1 + layout["panel_letter_offset_y"],
            "C)",
            fontsize=fmt["fontsize_panel_label"],
            fontweight="bold",
            ha="center",
            va="bottom",
            transform=category2_ax.transAxes,
        )

    # Panel D) - Above Jang et al. panel
    if jang_axes:
        jang_ax = jang_axes[0]
        jang_ax.text(
            layout["panel_letter_offset_x"],
            1 + layout["panel_letter_offset_y"],
            "D)",
            fontsize=fmt["fontsize_panel_label"],
            fontweight="bold",
            ha="center",
            va="bottom",
            transform=jang_ax.transAxes,
        )


def plot_performance_manuscript(
    data_paths: List[Path],
    output: Path,
    subplot_var: str,
    row_var: Optional[str],
    hue_var: str,
    category_key: Optional[str] = None,
    parameter_key: Optional[str] = None,
    experiment_names: Optional[List[str]] = None,
    data_ffonly_paths: Optional[List[Path]] = None,
    data_feedforward_paths: Optional[List[Path]] = None,
    data_category2_paths: Optional[List[Path]] = None,
    category2_key: Optional[str] = None,
    accuracy_measure: Optional[Union[str, List[str]]] = "accuracy",
    confidence_measure: Optional[Union[str, List[str]]] = None,
    dt: float = 2.0,
    config: Optional[Dict] = None,
    subplot_filter: Optional[List[str]] = None,
    jang_noise_type: str = "gaussian",
    plot_individual_seeds: bool = False,
    **kwargs,
) -> plt.Figure:
    """Plot performance grid with category-2 and Jang et al. (2021) benchmarks.

    This function creates a figure with:
    - Panel A: Performance traces over time (multiple subplots)
    - Panel B: Max accuracy vs parameter for main models
    - Panel C: Max accuracy vs parameter for category-2 models (optional)
    - Panel D: Jang et al. (2021) human and DNN benchmarks

    Parameters
    ----------
    data_paths : list of Path
        Paths to CSV files with model performance data
    output : Path
        Output figure path
    subplot_var : str
        Variable for horizontal subplots ('category', 'parameter', 'experiment')
    row_var : str or None
        Variable for vertical rows (use 'none' to collapse)
    hue_var : str
        Variable for color coding ('category', 'parameter', 'experiment')
    category_key : str, optional
        Category column name
    parameter_key : str, optional
        Parameter column name
    experiment_names : list of str, optional
        Experiment names
    data_ffonly_paths : list of Path, optional
        Paths to feedforward-only model data (recurrent models with connections silenced)
    data_feedforward_paths : list of Path, optional
        Paths to feedforward-trained model data (models trained without recurrence)
    data_category2_paths : list of Path, optional
        Paths to category-2 model data for Panel C
    category2_key : str, optional
        Category-2 column name for Panel C hue dimension
    accuracy_measure : str or list of str, default='accuracy'
        Accuracy measure column name(s)
    confidence_measure : str or list of str, optional
        Confidence measure column name(s)
    dt : float, default=2.0
        Time resolution (ms)
    config : dict, optional
        Visualization configuration (palette, naming, ordering)
    subplot_filter : list of str, optional
        Filter subplot values to display
    jang_noise_type : str, default='gaussian'
        Noise type for Jang data ('gaussian' or 'fourier')
    plot_individual_seeds : bool, default=False
        If True and multiple seed files are provided, plot each seed as a separate
        trace with the same color and linestyle instead of averaging over seeds.
    **kwargs
        Additional formatting arguments

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    logger.info("=" * 60)
    logger.info("Starting manuscript performance plotting")
    logger.info("=" * 60)

    # Import from plot_performance for data loading logic
    from dynvision.visualization.plot_performance import (
        _coerce_measure_list,
    )

    # Normalize dimensions
    subplot_clean = _coerce_optional_dimension(subplot_var)
    row_clean = _coerce_optional_dimension(row_var)
    hue_clean = _coerce_optional_dimension(hue_var)

    subplot_var, subplot_limit = _normalize_dimension(subplot_clean)
    row_var, row_limit = _normalize_dimension(row_clean)
    hue_var, hue_limit = _normalize_dimension(hue_clean)

    if subplot_var is None or hue_var is None:
        raise ValueError("subplot and hue dimensions cannot be empty")

    _validate_dimension_choices(
        subplot_var=subplot_var, hue_var=hue_var, column_var=row_var
    )

    if config is None:
        config = {"palette": {}, "naming": {}, "ordering": {}}

    # Parse accuracy and confidence measures
    accuracy_measures = _coerce_measure_list(accuracy_measure, default="accuracy")
    confidence_measures = _coerce_measure_list(
        confidence_measure, default="first_label_confidence"
    )

    primary_accuracy_measure = accuracy_measures[0] if accuracy_measures else None
    primary_confidence_measure = (
        confidence_measures[0] if confidence_measures else None
    )

    logger.info(f"Primary accuracy measure: {primary_accuracy_measure}")
    logger.info(f"Primary confidence measure: {primary_confidence_measure}")

    # Load data (same logic as plot_performance.py)
    logger.info(f"Loading data from {len(data_paths)} files...")
    combined_data = []

    experiment_names = experiment_names or []
    use_experiment_dimension = "experiment" in {
        dim for dim in (subplot_var, row_var, hue_var) if dim is not None
    }

    combined_label = (
        experiment_names[0]
        if experiment_names
        else (data_paths[0].stem if data_paths else "combined")
    )

    for i, data_path in enumerate(data_paths):
        df = pd.read_csv(data_path)

        if use_experiment_dimension:
            experiment_name = (
                experiment_names[i]
                if experiment_names and i < len(experiment_names)
                else (experiment_names[0] if experiment_names else data_path.stem)
            )
        else:
            experiment_name = combined_label

        df["experiment"] = experiment_name
        df["model_type"] = "full"
        df["source_file"] = data_path.stem

        # Add seed identifier if plotting individual seeds
        if plot_individual_seeds:
            df["seed_id"] = i

        combined_data.append(df)
        logger.info(f"Loaded {len(df)} rows from {data_path.name}")

    # Load feedforward-only data if provided (recurrent models with connections silenced)
    if data_ffonly_paths:
        for idx, path in enumerate(data_ffonly_paths):
            if path and Path(path).exists():
                df = pd.read_csv(path)
                if use_experiment_dimension:
                    exp_name = (
                        experiment_names[idx]
                        if experiment_names and idx < len(experiment_names)
                        else (experiment_names[0] if experiment_names else path.stem)
                    )
                else:
                    exp_name = combined_label
                df["experiment"] = exp_name
                df["model_type"] = "ffonly"
                df["source_file"] = path.stem

                # Add seed identifier if plotting individual seeds
                if plot_individual_seeds:
                    df["seed_id"] = idx

                combined_data.append(df)
                logger.info(f"Loaded {len(df)} ffonly rows from {path.name}")

    # Load feedforward-trained data if provided (models trained without recurrence)
    if data_feedforward_paths:
        for idx, path in enumerate(data_feedforward_paths):
            if path and Path(path).exists():
                df = pd.read_csv(path)
                if use_experiment_dimension:
                    exp_name = (
                        experiment_names[idx]
                        if experiment_names and idx < len(experiment_names)
                        else (experiment_names[0] if experiment_names else path.stem)
                    )
                else:
                    exp_name = combined_label
                df["experiment"] = exp_name
                df["model_type"] = "feedforward"
                df["source_file"] = path.stem

                # Add seed identifier if plotting individual seeds
                if plot_individual_seeds:
                    df["seed_id"] = idx

                combined_data.append(df)
                logger.info(f"Loaded {len(df)} feedforward-trained rows from {path.name}")

    df = pd.concat(combined_data, ignore_index=True)
    logger.info(f"Combined: {len(df)} rows, {len(df.columns)} columns")

    # Note: feedforward data is kept separate from ffonly
    # - ffonly: recurrent models with connections silenced (star markers)
    # - feedforward: models trained without recurrence (full time traces)

    # Load category-2 data if provided
    df_category2 = None
    category2_values = []
    category2_colors = {}
    if data_category2_paths and category2_key:
        logger.info(f"Loading category-2 data from {len(data_category2_paths)} files...")
        combined_category2_data = []
        for i, data_path in enumerate(data_category2_paths):
            if data_path and Path(data_path).exists():
                df_cat2 = pd.read_csv(data_path)
                if use_experiment_dimension:
                    experiment_name = (
                        experiment_names[i]
                        if experiment_names and i < len(experiment_names)
                        else (experiment_names[0] if experiment_names else data_path.stem)
                    )
                else:
                    experiment_name = combined_label
                df_cat2["experiment"] = experiment_name
                df_cat2["model_type"] = "full"
                df_cat2[category_key] = "None"
                df_cat2["source_file"] = data_path.stem
                if plot_individual_seeds:
                    df_cat2["seed_id"] = i
                combined_category2_data.append(df_cat2)
                logger.info(f"Loaded {len(df_cat2)} category-2 rows from {data_path.name}")

        if combined_category2_data:
            df_category2 = pd.concat(combined_category2_data, ignore_index=True)
            logger.info(
                f"Combined category-2: {len(df_category2)} rows, {len(df_category2.columns)} columns"
            )

            # Extract category-2 dimension values
            category2_values = _extract_dimension_values(
                df=df_category2,
                dimension="category",
                dimension_key=category2_key,
                config=config,
                dimension_limit=None,
            )
            logger.info(f"Category-2 values: {category2_values}")

            # Generate plasma color palette for category-2 values
            n_category2 = len(category2_values)
            if n_category2 > 0:
                plasma_cmap = plt.cm.plasma
                plasma_colors = [
                    plasma_cmap(i / max(n_category2 - 1, 1)) for i in range(n_category2)
                ]
                category2_colors = {
                    val: plasma_colors[i] for i, val in enumerate(category2_values)
                }
                logger.info(f"Generated plasma palette for {n_category2} category-2 values")
    else:
        logger.info("No category-2 data provided, Panel C will be blank")

    # Get dimension keys and values
    subplot_key = _get_dimension_key(
        dimension=subplot_var, category_key=category_key, parameter_key=parameter_key
    )
    row_key = (
        _get_dimension_key(
            dimension=row_var, category_key=category_key, parameter_key=parameter_key
        )
        if row_var
        else None
    )
    hue_key = _get_dimension_key(
        dimension=hue_var, category_key=category_key, parameter_key=parameter_key
    )

    subplot_values_all = _extract_dimension_values(
        df=df,
        dimension=subplot_var,
        dimension_key=subplot_key,
        config=config,
        dimension_limit=subplot_limit,
    )

    if row_var is None:
        row_values = [None]
    else:
        row_values = _extract_dimension_values(
            df=df,
            dimension=row_var,
            dimension_key=row_key,
            config=config,
            dimension_limit=row_limit,
        )

    hue_values = _extract_dimension_values(
        df=df,
        dimension=hue_var,
        dimension_key=hue_key,
        config=config,
        dimension_limit=hue_limit,
    )

    # Apply subplot filter
    subplot_values_display = (
        [v for v in subplot_values_all if v in subplot_filter]
        if subplot_filter
        else subplot_values_all
    )

    logger.info(f"Subplot values (display): {subplot_values_display}")
    logger.info(f"Row values: {row_values}")
    logger.info(f"Hue values: {hue_values}")

    # Get colors
    colors = _get_colors_for_dimension(
        values=hue_values, dimension_key=hue_key, config=config
    )

    # Calculate layout
    layout = {
        **MANUSCRIPT_LAYOUT,
        **{k: v for k, v in kwargs.items() if k in MANUSCRIPT_LAYOUT},
    }

    n_subplots_display = len(subplot_values_display)
    n_rows = len(row_values)

    performance_section_width = (
        n_subplots_display * layout["subplot_width"]
        + (n_subplots_display - 1) * layout["subplot_spacing_x"]
    )

    fig_width = (
        layout["left_margin"]
        + performance_section_width
        + layout["ssnr_panel_spacing"]
        + layout["ssnr_panel_width"]
        + layout["category2_panel_spacing"]
        + layout["category2_panel_width"]
        + layout["jang_panel_spacing"]
        + layout["jang_panel_width"]
        + layout["right_margin"]
    )

    fig_height = (
        layout["top_margin"]
        + n_rows * layout["subplot_height"]
        + (n_rows - 1) * layout["subplot_spacing_y"]
        + layout["bottom_margin"]
    )

    logger.info(f'Figure size: {fig_width:.2f}" x {fig_height:.2f}"')

    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Calculate panel positions
    def calculate_panel_positions():
        positions = {"performance": [], "ssnr": [], "category2": [], "jang": []}

        perf_left_start = layout["left_margin"]
        for row_idx in range(n_rows):
            row_positions = []
            row_bottom = layout["bottom_margin"] + (n_rows - 1 - row_idx) * (
                layout["subplot_height"] + layout["subplot_spacing_y"]
            )

            for col_idx in range(n_subplots_display):
                left = perf_left_start + col_idx * (
                    layout["subplot_width"] + layout["subplot_spacing_x"]
                )
                row_positions.append(
                    [
                        left / fig_width,
                        row_bottom / fig_height,
                        layout["subplot_width"] / fig_width,
                        layout["subplot_height"] / fig_height,
                    ]
                )
            positions["performance"].append(row_positions)

        # SSNR panels (Panel B)
        ssnr_left = (
            perf_left_start + performance_section_width + layout["ssnr_panel_spacing"]
        )
        for row_idx in range(n_rows):
            row_bottom = layout["bottom_margin"] + (n_rows - 1 - row_idx) * (
                layout["subplot_height"] + layout["subplot_spacing_y"]
            )
            positions["ssnr"].append(
                [
                    ssnr_left / fig_width,
                    row_bottom / fig_height,
                    layout["ssnr_panel_width"] / fig_width,
                    layout["subplot_height"] / fig_height,
                ]
            )

        # Category-2 panels (Panel C)
        category2_left = (
            ssnr_left + layout["ssnr_panel_width"] + layout["category2_panel_spacing"]
        )
        for row_idx in range(n_rows):
            row_bottom = layout["bottom_margin"] + (n_rows - 1 - row_idx) * (
                layout["subplot_height"] + layout["subplot_spacing_y"]
            )
            positions["category2"].append(
                [
                    category2_left / fig_width,
                    row_bottom / fig_height,
                    layout["category2_panel_width"] / fig_width,
                    layout["subplot_height"] / fig_height,
                ]
            )

        # Jang panels (Panel D)
        jang_left = (
            category2_left
            + layout["category2_panel_width"]
            + layout["jang_panel_spacing"]
        )
        for row_idx in range(n_rows):
            row_bottom = layout["bottom_margin"] + (n_rows - 1 - row_idx) * (
                layout["subplot_height"] + layout["subplot_spacing_y"]
            )
            positions["jang"].append(
                [
                    jang_left / fig_width,
                    row_bottom / fig_height,
                    layout["jang_panel_width"] / fig_width,
                    layout["subplot_height"] / fig_height,
                ]
            )

        return positions

    panel_positions = calculate_panel_positions()

    # Create axes
    perf_axes = []
    ssnr_axes = []
    category2_axes = []
    jang_axes = []

    for row_idx in range(n_rows):
        # Performance axes
        perf_row = []
        for col_idx in range(n_subplots_display):
            ax = fig.add_axes(panel_positions["performance"][row_idx][col_idx])
            perf_row.append(ax)
        perf_axes.append(perf_row)

        # SSNR axis (Panel B)
        ssnr_ax = fig.add_axes(panel_positions["ssnr"][row_idx])
        ssnr_axes.append(ssnr_ax)

        # Category-2 axis (Panel C)
        category2_ax = fig.add_axes(panel_positions["category2"][row_idx])
        category2_axes.append(category2_ax)

        # Jang axis (Panel D)
        jang_ax = fig.add_axes(panel_positions["jang"][row_idx])
        jang_axes.append(jang_ax)

    # Plot performance panels
    logger.info("=" * 60)
    logger.info("PANEL A - Performance Traces Over Time")
    if plot_individual_seeds:
        logger.info(
            "Traces: Each trace shows mean over trials/samples for individual seed"
        )
        logger.info("Error bars: SEM over trials/samples (for accuracy traces only)")
    else:
        logger.info(
            "Traces: Mean over trials/samples and seeds (when multiple input files)"
        )
        logger.info(
            "Error bars: SEM over trials/samples and seeds (shaded bands for accuracy traces)"
        )
    logger.info("           No error bars for confidence traces")
    logger.info("=" * 60)

    for row_idx, row_value in enumerate(row_values):
        for col_idx, subplot_value in enumerate(subplot_values_display):
            ax = perf_axes[row_idx][col_idx]

            cell_data = _filter_data_for_cell(
                df=df,
                row_key=row_key,
                row_value=row_value,
                subplot_key=subplot_key,
                subplot_value=subplot_value,
            )

            if len(cell_data) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Add error bars as shaded area with SEM for accuracy traces
            # Note: This applies to all traces; confidence traces use dotted style
            plot_kwargs = {**kwargs, "errorbar": "se", "err_style": "band"}

            _plot_accuracy_panel_with_ffonly(
                ax=ax,
                data=cell_data,
                hue_var=hue_var,
                hue_key=hue_key,
                hue_values=hue_values,
                colors=colors,
                dt=dt,
                show_ylabel=(col_idx == 0),
                show_legend=False,  # Legend moved to category hue legend
                accuracy_cols=accuracy_measures,
                confidence_cols=confidence_measures,
                **plot_kwargs,
            )

            if col_idx == 0:
                sns.despine(ax=ax, left=False)

            # Add subplot title
            if row_idx == 0:
                title_text = get_display_name(subplot_key, config)
                title_text += " = " + _format_legend_label(
                    subplot_key, subplot_value, config, dt
                )
                ax.set_title(title_text, fontsize=FORMATTING["fontsize_axis"], pad=10)

        # Add row label
        if row_var is not None:
            if row_var == "experiment":
                experiment_name = row_value
            else:
                experiment_name = (
                    experiment_names[0]
                    if experiment_names
                    else get_display_name(row_key or "row", config)
                )

            row_label_text = get_display_name(f"{experiment_name}_experiment", config)
            if not row_label_text or row_label_text == f"{experiment_name}_experiment":
                formatted_name = str(experiment_name) if experiment_name else ""
                row_label_text = formatted_name.replace("_", " ").title()

            perf_axes[row_idx][0].text(
                -0.5,
                0.5,
                row_label_text,
                rotation=90,
                ha="center",
                va="center",
                fontsize=FORMATTING["fontsize_axis"],
                fontweight="bold",
                transform=perf_axes[row_idx][0].transAxes,
            )

    # Plot peak height panels (Panel B - max accuracy vs parameter)
    logger.info("=" * 60)
    logger.info("PANEL B - Maximum Performance vs Parameter")
    logger.info(
        "Data points: Max performance = max over time of (mean over trials/samples at each timepoint)"
    )
    if len(data_paths) > 1:
        logger.info(
            "             When multiple seeds: mean of max performance across seeds"
        )
        logger.info(
            "Error bars: STD (Standard Deviation) of max performance across seeds"
        )
    else:
        logger.info("             Single seed - no error bars")
    logger.info("=" * 60)

    for row_idx, row_value in enumerate(row_values):
        _plot_peak_height_panel(
            ax=ssnr_axes[row_idx],
            data=df,
            row_key=row_key,
            row_value=row_value,
            subplot_key=subplot_key,
            subplot_values=subplot_values_all,
            hue_key=hue_key,
            hue_values=hue_values,
            colors=colors,
            config=config,
            dt=dt,
            show_ylabel=(row_idx == 0),
            ylabel="Max Accuracy",
            show_legend=False,  # Legend moved to category hue legend
            show_seed_errorbars=True,
            accuracy_cols=accuracy_measures,
            confidence_cols=confidence_measures,
            primary_accuracy=primary_accuracy_measure,
            primary_confidence=primary_confidence_measure,
        )

    # Plot category-2 panels (Panel C)
    logger.info("=" * 60)
    logger.info("PANEL C - Category-2 Max Performance vs Parameter")
    if df_category2 is not None and len(category2_values) > 0:
        logger.info(
            "Data points: Max performance = max over time of (mean over trials/samples at each timepoint)"
        )
        if len(data_category2_paths) > 1:
            logger.info(
                "             When multiple seeds: mean of max performance across seeds"
            )
            logger.info(
                "Error bars: STD (Standard Deviation) of max performance across seeds"
            )
        else:
            logger.info("             Single seed - no error bars")
    else:
        logger.info("No category-2 data provided - panel will be blank")
    logger.info("=" * 60)

    for row_idx, row_value in enumerate(row_values):
        if df_category2 is not None and len(category2_values) > 0:
            _plot_category2_panel(
                ax=category2_axes[row_idx],
                data=df_category2,
                row_key=row_key,
                row_value=row_value,
                subplot_key=subplot_key,
                subplot_values=subplot_values_all,  # Use all values for plotting
                category2_key=category2_key,
                category2_values=category2_values,
                colors=category2_colors,
                config=config,
                dt=dt,
                show_ylabel=False,  # No y-label for Panel C
                show_legend=(row_idx == 0),  # Show legend in first row only
                accuracy_cols=accuracy_measures,
                primary_accuracy=primary_accuracy_measure,
            )
        else:
            # Show blank panel with message
            category2_axes[row_idx].text(
                0.5,
                0.5,
                "No Data",
                ha="center",
                va="center",
                transform=category2_axes[row_idx].transAxes,
            )
            category2_axes[row_idx].set_xticks([])
            category2_axes[row_idx].set_yticks([])

    # Plot Jang et al. panels (Panel D - Jang data only)
    logger.info("=" * 60)
    logger.info("PANEL D - Jang et al. (2021) Benchmark Data")
    logger.info("Human data points: Mean accuracy over human observers")
    logger.info(
        "               Error bars: STD (Standard Deviation) over human observers"
    )
    logger.info("DNN data points: Mean accuracy over DNN models")
    logger.info("             Error bars: STD (Standard Deviation) over DNN models")
    logger.info("=" * 60)

    for row_idx in range(n_rows):
        _plot_jang_panel(
            ax=jang_axes[row_idx],
            jang_noise_type=jang_noise_type,
            show_ylabel=(row_idx == 0),
        )

    # Synchronize y-limits between Panel A, B, and C (Panel D has independent scale)
    all_perf_axes = [ax for row in perf_axes for ax in row]
    panel_abc_axes = all_perf_axes + ssnr_axes + category2_axes

    try:
        global_ymin = min(ax.get_ylim()[0] for ax in panel_abc_axes)
        global_ymax = max(ax.get_ylim()[1] for ax in panel_abc_axes)
    except ValueError:
        global_ymin, global_ymax = 0.0, 1.0

    global_ymin = max(global_ymin, 0.0)
    global_ymax = min(global_ymax, 1.0)

    # Apply to Panel A, B, and C
    for ax in panel_abc_axes:
        ax.set_ylim(global_ymin, global_ymax)

    # Hide y-tick labels except leftmost
    for row_idx in range(n_rows):
        for col_idx in range(n_subplots_display):
            if col_idx > 0:
                perf_axes[row_idx][col_idx].set_yticklabels([])

    # Add label indicators to performance panels (Panel A)
    for row_idx in range(n_rows):
        for col_idx in range(n_subplots_display):
            try:
                calculate_label_indicator(perf_axes[row_idx][col_idx])
            except Exception as e:
                logger.debug(f"Could not add label indicator to Panel A: {e}")

    # Add hue legend with "without recurrence" entry for ffonly/feedforward models
    has_ffonly_or_feedforward = (
        (data_ffonly_paths and len(data_ffonly_paths) > 0)
        or (data_feedforward_paths and len(data_feedforward_paths) > 0)
    )
    _add_hue_legend(
        fig=fig,
        perf_axes=perf_axes,
        hue_key=hue_key,
        hue_values=hue_values,
        colors=colors,
        config=config,
        dt=dt,
        show_without_recurrence=has_ffonly_or_feedforward,
    )

    # Add panel letters
    _add_panel_letters(
        fig=fig,
        perf_axes=perf_axes,
        ssnr_axes=ssnr_axes,
        category2_axes=category2_axes,
        jang_axes=jang_axes,
        layout=layout,
    )

    # Save
    logger.info(f"Saving figure to: {output}")
    save_plot(output)
    logger.info("Manuscript performance plotting complete")

    return fig


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot performance with Jang et al. benchmarks"
    )
    parser.add_argument(
        "--data", type=Path, nargs="+", required=True, help="Paths to CSV files"
    )
    parser.add_argument(
        "--data-ffonly",
        type=Path,
        nargs="*",
        help="Paths to ffonly CSV files (recurrent models with connections silenced)",
    )
    parser.add_argument(
        "--data-feedforward",
        type=Path,
        nargs="*",
        help="Paths to feedforward-trained CSV files (models trained without recurrence)",
    )
    parser.add_argument(
        "--data-category2",
        type=Path,
        nargs="*",
        help="Paths to category-2 CSV files for Panel C",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output path")
    parser.add_argument(
        "--subplot",
        type=str,
        required=True,
        choices=["category", "parameter", "experiment"],
        help="Variable for subplots",
    )
    parser.add_argument(
        "--row", type=str, required=True, help="Variable for rows (or 'none')"
    )
    parser.add_argument(
        "--hue",
        type=str,
        required=True,
        choices=["category", "parameter", "experiment"],
        help="Variable for hue",
    )
    parser.add_argument("--category-key", type=str, help="Category column name")
    parser.add_argument(
        "--category2-key", type=str, help="Category-2 column name for Panel C"
    )
    parser.add_argument("--parameter-key", type=str, help="Parameter column name")
    parser.add_argument(
        "--experiment-names", type=str, nargs="*", help="Experiment names"
    )
    parser.add_argument(
        "--accuracy-measure", type=str, default="accuracy", help="Accuracy measure"
    )
    parser.add_argument(
        "--confidence-measure",
        type=str,
        default=None,
        help="Confidence measure (default: None, no confidence trace shown)",
    )
    parser.add_argument("--dt", type=float, default=2.0, help="Time resolution (ms)")
    parser.add_argument("--palette", type=str, help="JSON color palette")
    parser.add_argument("--naming", type=str, help="JSON naming dict")
    parser.add_argument("--ordering", type=str, help="JSON ordering dict")
    parser.add_argument("--subplot-filter", type=str, nargs="*", help="Subplot filter")
    parser.add_argument(
        "--jang-noise-type",
        type=str,
        default="gaussian",
        choices=["gaussian", "fourier"],
        help="Jang et al. noise type",
    )
    parser.add_argument(
        "--plot-individual-seeds",
        action="store_true",
        help="Plot each seed as a separate trace instead of averaging over seeds",
    )

    args = parser.parse_args()

    config = load_config_from_args(
        palette_str=args.palette,
        naming_str=args.naming,
        ordering_str=args.ordering,
    )

    plot_performance_manuscript(
        data_paths=args.data,
        output=args.output,
        subplot_var=args.subplot,
        row_var=args.row,
        hue_var=args.hue,
        category_key=args.category_key,
        parameter_key=args.parameter_key,
        experiment_names=args.experiment_names,
        data_ffonly_paths=args.data_ffonly,
        data_feedforward_paths=args.data_feedforward,
        data_category2_paths=args.data_category2,
        category2_key=args.category2_key,
        accuracy_measure=args.accuracy_measure,
        confidence_measure=args.confidence_measure,
        dt=args.dt,
        config=config,
        subplot_filter=args.subplot_filter,
        jang_noise_type=args.jang_noise_type,
        plot_individual_seeds=args.plot_individual_seeds,
    )


if __name__ == "__main__":
    main()
