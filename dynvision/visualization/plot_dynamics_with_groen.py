"""plot_dynamics_with_groen.py
Enhanced dynamics plot including Groen et al. 2022 human V1 data in panel C.

This script extends plot_dynamics.py by adding empirical human ECoG data
from Groen et al. (2022) to panel C for comparison with model results:
    - Duration experiment: Figure 2B (Temporal Summation)
    - Interval experiment: Figure 3B (Recovery from Adaptation)
    - Contrast experiment: Figure 4B (Time-to-Peak)

Example:
    ```bash
    python plot_dynamics_with_groen.py --data responses.csv --output dynamics_groen.png \\
        --parameter duration --category duration --focus-layer V1 --dt 2.0 \\
        --groen-data /path/to/groen2022_csv
    ```
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dynvision.utils.visualization_utils import (
    save_plot,
    load_config_from_args,
    get_display_name,
    format_parameter_value,
    DT_CONVERT_PARAMS,
)

# Import all functions and constants from plot_dynamics
from dynvision.visualization.plot_dynamics import (
    determine_experiment_info,
    add_panel_labels,
    calculate_metrics,
    create_panel_a,
    create_panel_b,
    create_panel_d,
    DYNAMICS_LAYOUT,
    DYNAMICS_STYLE,
    _extract_dimension_values,
    _get_colors_for_dimension,
    _add_horizontal_legend,
    RESPONSES_FORMATTING,
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Extended layout for 5-panel figure (3 panels on right side)
# Panel E bottom must align with Panel B's actual visual bottom (not lower_bottom)
# Panel B uses ridge plots with overlap, so its actual bottom is higher than lower_bottom
_panel_gap = 0.06  # Vertical gap between right-side panels

# Calculate Panel B's actual visual bottom based on ridge plot formula
# Ridge plots use: bottom_pos = ridge_top - (n-1)*spacing - plot_height
# where spacing = ridge_height/n * (1-overlap) and plot_height = ridge_height/n * 1.4
# For typical 7 parameter values with overlap=0.2:
_n_typical_params = 7
_ridge_overlap = 0.2
_ridge_height = DYNAMICS_LAYOUT["lower_top"] - DYNAMICS_LAYOUT["lower_bottom"]
_ridge_spacing = _ridge_height / _n_typical_params * (1 - _ridge_overlap)
_ridge_plot_height = _ridge_height / _n_typical_params * 1.4
_panel_b_actual_bottom = (
    DYNAMICS_LAYOUT["lower_top"]
    - (_n_typical_params - 1) * _ridge_spacing
    - _ridge_plot_height
)

# Use Panel B's actual bottom for Panel E alignment
_right_bottom = _panel_b_actual_bottom
_right_top = DYNAMICS_LAYOUT["upper_top"]  # Panel C top = Panel A top
_total_height = _right_top - _right_bottom
_available_height = _total_height - 2 * _panel_gap  # Height minus two gaps
_panel_height = _available_height / 3  # Equal height for C, D, E

DYNAMICS_LAYOUT_EXTENDED = {
    **DYNAMICS_LAYOUT,
    # Override right-side vertical positions for 3 panels with gaps
    # Build from bottom up to ensure alignment with Panel B's actual bottom
    "right_lower_bottom": _right_bottom,  # Panel E bottom (aligns with Panel B)
    "right_lower_top": _right_bottom + _panel_height,  # Panel E top
    "right_middle_bottom": _right_bottom
    + _panel_height
    + _panel_gap,  # Panel D bottom
    "right_middle_top": _right_bottom + 2 * _panel_height + _panel_gap,  # Panel D top
    "right_upper_bottom": _right_bottom
    + 2 * _panel_height
    + 2 * _panel_gap,  # Panel C bottom
    "right_upper_top": _right_top,  # Panel C top
}


def add_panel_labels_extended(fig):
    """Add panel labels A-E for extended 5-panel figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add labels to
    """
    layout = DYNAMICS_LAYOUT_EXTENDED
    panel_x_offset = layout["panel_label_x_offset"]
    panel_y_offset = layout["panel_label_y_offset"]
    plots_start = layout["plots_start"]
    left_width = layout["left_plots_width"]
    gap_width = layout["gap_width"]

    # Left side panel labels (A and B)
    left_x = plots_start + panel_x_offset
    fig.text(
        x=left_x,
        y=layout["upper_top"] + panel_y_offset,
        s="A)",
        fontsize=DYNAMICS_STYLE["fontsize_panel_labels"],
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        x=left_x,
        y=layout["lower_top"] + panel_y_offset,
        s="B)",
        fontsize=DYNAMICS_STYLE["fontsize_panel_labels"],
        fontweight="bold",
        ha="left",
        va="top",
    )

    # Right side panel labels (C, D, E)
    right_x = plots_start + left_width + gap_width + 1.5 * panel_x_offset
    fig.text(
        x=right_x,
        y=layout["right_upper_top"] + panel_y_offset,
        s="C)",
        fontsize=DYNAMICS_STYLE["fontsize_panel_labels"],
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        x=right_x,
        y=layout["right_middle_top"] + panel_y_offset,
        s="D)",
        fontsize=DYNAMICS_STYLE["fontsize_panel_labels"],
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        x=right_x,
        y=layout["right_lower_top"] + panel_y_offset,
        s="E)",
        fontsize=DYNAMICS_STYLE["fontsize_panel_labels"],
        fontweight="bold",
        ha="left",
        va="top",
    )


def load_groen_data(experiment_type: str, groen_data_path: Path) -> pd.DataFrame:
    """Load appropriate Groen et al. 2022 data based on experiment type.

    Parameters
    ----------
    experiment_type : str
        One of 'duration', 'interval', or 'contrast'
    groen_data_path : Path
        Directory containing Groen CSV files

    Returns
    -------
    pd.DataFrame
        Groen data for the specified experiment
    """
    exp_info = determine_experiment_info(experiment_type)
    exp_type = exp_info["type"]

    # Map experiment types to Groen data files
    file_mapping = {
        "duration": "groen2022_duration_data.csv",
        "interval": "groen2022_interval_data.csv",
        "contrast": "groen2022_contrast_data.csv",
    }

    if exp_type not in file_mapping:
        logger.warning(f"Unknown experiment type '{exp_type}', defaulting to duration")
        exp_type = "duration"

    file_path = groen_data_path / file_mapping[exp_type]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Groen data file not found: {file_path}. "
            f"Please ensure Groen data is extracted to {groen_data_path}"
        )

    logger.info(f"Loading Groen data from: {file_path}")
    return pd.read_csv(file_path)


def plot_groen_temporal_summation(ax, df_groen):
    """Plot Figure 2B: Temporal Summation (normalized summed response).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    df_groen : pd.DataFrame
        Groen duration data
    """
    # Calculate summed response for each duration (0-1s after onset)
    durations = []
    summed_responses = []

    for sample_idx in df_groen["sample_index"].unique():
        subset = df_groen[df_groen["sample_index"] == sample_idx]
        dur_ms = subset["duration_ms"].iloc[0]

        # Create integration window for this subset
        time = subset["time"].values
        integration_window = (time >= 0) & (time <= 1.0)

        # Sum response in integration window
        response_in_window = subset.loc[integration_window, "V1_response"].values
        summed = np.sum(response_in_window)

        durations.append(dur_ms)
        summed_responses.append(summed)

    # Normalize to longest duration
    summed_responses = np.array(summed_responses)
    summed_responses_norm = summed_responses / summed_responses[-1]
    durations = np.array(durations)

    # Plot data
    ax.plot(
        durations,
        summed_responses_norm,
        "ko-",
        linewidth=2.5,
        markersize=10,
        markeredgewidth=1.5,
        markeredgecolor="white",
        label="Human V1 (ECoG)",
        zorder=3,
    )

    # Linear prediction (extrapolated from longest duration)
    linear_prediction = durations / durations[-1]
    ax.plot(
        durations,
        linear_prediction,
        "k--",
        linewidth=1.5,
        alpha=0.5,
        label="Linear prediction",
        zorder=2,
    )

    # Styling
    ax.set_xlabel(
        "Stimulus Duration (ms)", fontsize=RESPONSES_FORMATTING["fontsize_label"]
    )
    ax.set_ylabel(
        "Summed Response",
        fontsize=RESPONSES_FORMATTING["fontsize_label"],
        fontweight="bold",
    )
    legend = ax.legend(
        fontsize=RESPONSES_FORMATTING["fontsize_legend"],
        frameon=False,
        fancybox=False,
        edgecolor="gray",
        loc="lower right",
        title="Groen et al. 2022",
        title_fontsize=RESPONSES_FORMATTING["fontsize_legend"],
    )
    if legend.get_title():
        legend.get_title().set_fontweight("bold")
    ax.grid(True, alpha=0.3, linestyle=":", zorder=1)

    # Use LINEAR scale for x-axis with proper numeric spacing
    ax.set_xlim(0, max(durations) * 1.05)

    # Set tick label font size
    ax.tick_params(axis="both", labelsize=RESPONSES_FORMATTING["fontsize_tick"])
    sns.despine(ax=ax, left=True, bottom=True)


def plot_groen_adaptation_recovery_simple(ax, df_groen):
    """Plot Recovery from Adaptation using simplified per-condition calculation.

    This method matches how recovery is calculated for model layer responses:
    - For each two-pulse condition independently:
      1. Find peak response to first pulse (in window after 1st pulse onset)
      2. Find peak response to second pulse (in window after 2nd pulse onset)
      3. Compute ratio: (2nd pulse peak) / (1st pulse peak)

    Unlike the published methodology, this does NOT:
    - Average across single-pulse reference conditions (ONEPULSE-4, ONEPULSE-5)
    - Subtract estimated first pulse response
    - Use partial time courses from other ISI conditions

    This simplified approach is more directly comparable to model response analysis
    but may show stronger recovery values since it doesn't control for continuing
    first pulse activity.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    df_groen : pd.DataFrame
        Groen interval data (two-pulse conditions)
    """
    isis = []
    ratios = []

    # Window for finding peaks after each pulse
    w = 0.3  # 300ms window

    for sample_idx in sorted(df_groen["sample_index"].unique()):
        subset = df_groen[df_groen["sample_index"] == sample_idx]
        isi_ms = subset["isi_ms"].iloc[0]
        time = subset["time"].values

        # Find the two stimulus pulses
        stim_on = subset[subset["stim"] == 1]
        stim_times_idx = stim_on["times_index"].values

        # Find gap between pulses
        gaps = np.where(np.diff(stim_times_idx) > 1)[0]
        if len(gaps) == 0:
            logger.warning(f"No gap found for ISI={isi_ms}ms, skipping")
            continue

        # Get pulse onset times
        pulse1_indices = stim_times_idx[: gaps[0] + 1]
        pulse2_indices = stim_times_idx[gaps[0] + 1 :]

        pulse1_onset_time = subset[subset["times_index"] == pulse1_indices[0]][
            "time"
        ].iloc[0]
        pulse2_onset_time = subset[subset["times_index"] == pulse2_indices[0]][
            "time"
        ].iloc[0]

        # Find peak in window after first pulse
        t_window1 = (time > pulse1_onset_time) & (time <= pulse1_onset_time + w)
        pulse1_peak = subset.loc[t_window1, "V1_response"].max()

        # Find peak in window after second pulse
        t_window2 = (time > pulse2_onset_time) & (time <= pulse2_onset_time + w)
        pulse2_peak = subset.loc[t_window2, "V1_response"].max()

        # Compute ratio (recovery metric)
        ratio = pulse2_peak / pulse1_peak if pulse1_peak > 0 else np.nan

        isis.append(isi_ms)
        ratios.append(ratio)

    isis = np.array(isis)
    ratios = np.array(ratios)

    # Plot data
    ax.plot(
        isis,
        ratios,
        "ko-",
        linewidth=2.5,
        markersize=10,
        markeredgewidth=1.5,
        markeredgecolor="white",
        label="Human V1 (ECoG)",
        zorder=3,
    )

    # Add no-adaptation reference line
    ax.axhline(
        1.0,
        color="k",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="No Adaptation",
        zorder=2,
    )

    # Styling
    ax.set_xlabel(
        "Stimulus Interval (ms)", fontsize=RESPONSES_FORMATTING["fontsize_label"]
    )
    ax.set_ylabel(
        "Peak Ratio",
        fontsize=RESPONSES_FORMATTING["fontsize_label"],
        fontweight="bold",
    )
    legend = ax.legend(
        fontsize=RESPONSES_FORMATTING["fontsize_legend"],
        frameon=False,
        fancybox=False,
        edgecolor="gray",
        loc="lower right",
        title="Groen et al. 2022",
        title_fontsize=RESPONSES_FORMATTING["fontsize_legend"],
    )
    if legend.get_title():
        legend.get_title().set_fontweight("bold")
    ax.grid(True, alpha=0.3, linestyle=":", zorder=1)

    # Use LINEAR scale for x-axis (not log, not equidistant)
    ax.set_xlim(0, max(isis) * 1.05)
    ax.set_ylim(0.15, 1.15)

    # Set tick label font size
    ax.tick_params(axis="both", labelsize=RESPONSES_FORMATTING["fontsize_tick"])
    sns.despine(ax=ax, left=True, bottom=True)


def plot_groen_contrast_time_to_peak(ax, df_groen):
    """Plot Figure 4B (middle): Time-to-Peak vs Contrast.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    df_groen : pd.DataFrame
        Groen contrast data
    """
    contrasts = []
    times_to_peak = []

    for sample_idx in df_groen["sample_index"].unique():
        subset = df_groen[df_groen["sample_index"] == sample_idx]
        contrast_pct = subset["contrast_pct"].iloc[0] / 100.0  # Convert to [0, 1]

        # Find time-to-peak
        peak_idx = subset["V1_response"].values.argmax()
        time_to_peak_s = subset["time"].iloc[peak_idx]
        time_to_peak_ms = time_to_peak_s * 1000  # Convert to ms

        contrasts.append(contrast_pct)
        times_to_peak.append(time_to_peak_ms)

    contrasts = np.array(contrasts)
    times_to_peak = np.array(times_to_peak)

    # Plot data
    ax.plot(
        contrasts,
        times_to_peak,
        "ko-",
        linewidth=2.5,
        markersize=10,
        markeredgewidth=1.5,
        markeredgecolor="white",
        label="Human V1 (ECoG)",
        zorder=3,
    )

    # Styling
    ax.set_xlabel("Contrast", fontsize=RESPONSES_FORMATTING["fontsize_label"])
    ax.set_ylabel(
        "Peak Time (ms)",
        fontsize=RESPONSES_FORMATTING["fontsize_label"],
        fontweight="bold",
    )
    legend = ax.legend(
        fontsize=RESPONSES_FORMATTING["fontsize_legend"],
        frameon=False,
        fancybox=False,
        edgecolor="gray",
        loc="upper right",
        title="Groen et al. 2022",
        title_fontsize=RESPONSES_FORMATTING["fontsize_legend"],
    )
    if legend.get_title():
        legend.get_title().set_fontweight("bold")
    ax.grid(True, alpha=0.3, linestyle=":", zorder=1)

    # Use LINEAR scale with proper numeric spacing
    ax.set_xlim(0, max(contrasts) * 1.05)

    # Set tick label font size
    ax.tick_params(axis="both", labelsize=RESPONSES_FORMATTING["fontsize_tick"])
    sns.despine(ax=ax, left=True, bottom=True)


def create_panel_c_with_groen(fig, experiment_type: str, groen_data_path: Path):
    """Create Panel C with appropriate Groen et al. 2022 figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add panel to
    experiment_type : str
        Type of experiment (duration, interval, contrast)
    groen_data_path : Path
        Path to directory containing Groen CSV files

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing Groen plot
    """
    plots_start = DYNAMICS_LAYOUT["plots_start"]
    left_width = DYNAMICS_LAYOUT["left_plots_width"]
    gap_width = DYNAMICS_LAYOUT["gap_width"]
    right_width = DYNAMICS_LAYOUT["right_plots_width"]
    upper_top = DYNAMICS_LAYOUT["upper_top"]
    upper_bottom = DYNAMICS_LAYOUT["upper_bottom"] + 1  # manual tweaking

    panel_height = upper_top - upper_bottom
    ax = fig.add_axes(
        [
            plots_start + left_width + gap_width,
            upper_bottom,
            right_width,
            panel_height,
        ]
    )

    # Check if groen_data_path is provided
    if groen_data_path is None:
        ax.text(
            0.5,
            0.5,
            "Groen data not provided\n\nUse --groen-data to specify path",
            ha="center",
            va="center",
            fontsize=RESPONSES_FORMATTING["fontsize_label"],
            alpha=0.7,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    try:
        # Load Groen data
        df_groen = load_groen_data(experiment_type, groen_data_path)
        exp_info = determine_experiment_info(experiment_type)

        # Plot appropriate figure based on experiment type
        if exp_info["type"] == "duration":
            plot_groen_temporal_summation(ax, df_groen)
        elif exp_info["type"] == "interval":
            # Use simplified method that matches model response calculation
            plot_groen_adaptation_recovery_simple(ax, df_groen)
        elif exp_info["type"] == "contrast":
            plot_groen_contrast_time_to_peak(ax, df_groen)
        else:
            # Fallback for unknown experiment types
            ax.text(
                0.5,
                0.5,
                "Groen data not available\nfor this experiment type",
                ha="center",
                va="center",
                fontsize=RESPONSES_FORMATTING["fontsize_label"],
                alpha=0.7,
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    except FileNotFoundError as e:
        logger.error(str(e))
        ax.text(
            0.5,
            0.5,
            f"Groen data not found\n{e}",
            ha="center",
            va="center",
            fontsize=RESPONSES_FORMATTING["fontsize_label"],
            alpha=0.7,
            transform=ax.transAxes,
            wrap=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    except Exception as e:
        logger.error(f"Error creating Groen panel: {e}")
        ax.text(
            0.5,
            0.5,
            f"Error loading Groen data:\n{str(e)[:50]}...",
            ha="center",
            va="center",
            fontsize=RESPONSES_FORMATTING["fontsize_label"],
            alpha=0.7,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    return ax


def create_panel_c_extended(fig, experiment_type: str, groen_data_path: Path):
    """Create Panel C with Groen data using extended 3-panel layout.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add panel to
    experiment_type : str
        Type of experiment (duration, interval, contrast)
    groen_data_path : Path
        Path to directory containing Groen CSV files

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing Groen plot
    """
    plots_start = DYNAMICS_LAYOUT_EXTENDED["plots_start"]
    left_width = DYNAMICS_LAYOUT_EXTENDED["left_plots_width"]
    gap_width = DYNAMICS_LAYOUT_EXTENDED["gap_width"]
    right_width = DYNAMICS_LAYOUT_EXTENDED["right_plots_width"]
    upper_top = DYNAMICS_LAYOUT_EXTENDED["right_upper_top"]
    upper_bottom = DYNAMICS_LAYOUT_EXTENDED["right_upper_bottom"]

    panel_height = upper_top - upper_bottom
    ax = fig.add_axes(
        [
            plots_start + left_width + gap_width,
            upper_bottom,
            right_width,
            panel_height,
        ]
    )

    # Check if groen_data_path is provided
    if groen_data_path is None:
        ax.text(
            0.5,
            0.5,
            "Groen data not provided\n\nUse --groen-data to specify path",
            ha="center",
            va="center",
            fontsize=RESPONSES_FORMATTING["fontsize_label"],
            alpha=0.7,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    try:
        # Load Groen data
        df_groen = load_groen_data(experiment_type, groen_data_path)
        exp_info = determine_experiment_info(experiment_type)

        # Plot appropriate figure based on experiment type
        if exp_info["type"] == "duration":
            plot_groen_temporal_summation(ax, df_groen)
        elif exp_info["type"] == "interval":
            plot_groen_adaptation_recovery_simple(ax, df_groen)
        elif exp_info["type"] == "contrast":
            plot_groen_contrast_time_to_peak(ax, df_groen)
        else:
            ax.text(
                0.5,
                0.5,
                "Groen data not available\nfor this experiment type",
                ha="center",
                va="center",
                fontsize=RESPONSES_FORMATTING["fontsize_label"],
                alpha=0.7,
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    except FileNotFoundError as e:
        logger.error(str(e))
        ax.text(
            0.5,
            0.5,
            f"Groen data not found\n{e}",
            ha="center",
            va="center",
            fontsize=RESPONSES_FORMATTING["fontsize_label"],
            alpha=0.7,
            transform=ax.transAxes,
            wrap=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    except Exception as e:
        logger.error(f"Error creating Groen panel: {e}")
        ax.text(
            0.5,
            0.5,
            f"Error loading Groen data:\n{str(e)[:50]}...",
            ha="center",
            va="center",
            fontsize=RESPONSES_FORMATTING["fontsize_label"],
            alpha=0.7,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    return ax


def create_panel_d_extended(
    fig,
    df,
    focus_layer,
    parameter,
    category,
    exp_info,
    hue_values,
    colors,
    dt,
    config,
):
    """Create Panel D (metrics) using extended 3-panel layout.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add panel to
    df : pd.DataFrame
        Model response data
    focus_layer : str
        Layer to focus on
    parameter : str
        Parameter column name
    category : str
        Category column name
    exp_info : dict
        Experiment info dictionary
    hue_values : list
        Unique category values
    colors : dict
        Color mapping for categories
    dt : float
        Time step in milliseconds
    config : dict
        Configuration dictionary

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing metrics plot
    """
    plots_start = DYNAMICS_LAYOUT_EXTENDED["plots_start"]
    left_width = DYNAMICS_LAYOUT_EXTENDED["left_plots_width"]
    gap_width = DYNAMICS_LAYOUT_EXTENDED["gap_width"]
    right_width = DYNAMICS_LAYOUT_EXTENDED["right_plots_width"]
    middle_top = DYNAMICS_LAYOUT_EXTENDED["right_middle_top"]
    middle_bottom = DYNAMICS_LAYOUT_EXTENDED["right_middle_bottom"]

    panel_height = middle_top - middle_bottom
    ax = fig.add_axes(
        [
            plots_start + left_width + gap_width,
            middle_bottom,
            right_width,
            panel_height,
        ]
    )

    # Calculate metrics (similar to create_panel_d in plot_dynamics.py)
    metric = exp_info["metric"]
    metric_col = f"{focus_layer}_{metric}"

    if metric_col not in df.columns:
        df = calculate_metrics(
            df=df,
            focus_layer=focus_layer,
            parameter=parameter,
            category=category,
            metric=metric,
            dt=dt,
        )

    df = df.copy()
    df[category] = df[category].astype(str)
    hue_values_str = [str(val) for val in hue_values]

    # Check if parameter is temporal and needs conversion to ms
    is_temporal = exp_info["type"].lower() in DT_CONVERT_PARAMS and dt is not None

    # Create x-axis column (converted to ms if temporal)
    x_col = f"{parameter}_ms" if is_temporal else parameter
    if is_temporal:
        df[x_col] = df[parameter].astype(float) * dt

    if not df[metric_col].notna().any():
        ax.text(
            0.5,
            0.5,
            f"No metric data for layer {focus_layer}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    # Plot using seaborn lineplot for consistency
    sns.lineplot(
        data=df,
        x=x_col,
        y=metric_col,
        ax=ax,
        marker="o",
        hue=category,
        hue_order=hue_values_str,
        palette=colors,
        markersize=8,
        linewidth=2,
        err_style="bars",
        errorbar="se",
        legend=False,  # Remove legend (redundant with horizontal legend)
    )

    # Add reference line at 1.0 for peak_ratio metric
    if metric == "peak_ratio":
        ax.axhline(
            1.0,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            zorder=1,
        )
    elif metric == "summed_response":
        # Linear prediction (extrapolated from longest duration)
        x_sorted = df[x_col].sort_values()
        linear_prediction = x_sorted / df[x_col].max()
        ax.plot(
            x_sorted,
            linear_prediction,
            "k--",
            linewidth=1.5,
            alpha=0.5,
            zorder=2,
        )
        ax.set_xlim(0, df[x_col].max())
        ax.set_ylim(0, 1)

    # Styling
    metric_display = exp_info.get("metric_label", metric.replace("_", " ").title())

    # No x-axis label for Panel D (shared with Panel E)
    ax.set_xlabel("")
    ax.set_ylabel(
        metric_display,
        fontsize=RESPONSES_FORMATTING["fontsize_label"],
        fontweight="bold",
    )

    # Hide x tick labels (shared axis with Panel E)
    ax.tick_params(axis="y", labelsize=RESPONSES_FORMATTING["fontsize_tick"])
    ax.tick_params(axis="x", labelbottom=False)
    ax.grid(True, alpha=0.3, linestyle=":", zorder=1)
    sns.despine(ax=ax, left=True, bottom=True)

    return ax


def create_panel_e(
    fig,
    df_category2,
    focus_layer,
    parameter,
    category2,
    exp_info,
    dt,
    config,
):
    """Create Panel E for category-2 metrics using plasma palette.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add panel to
    df_category2 : pd.DataFrame
        Category-2 response data
    focus_layer : str
        Layer to focus on
    parameter : str
        Parameter column name
    category2 : str
        Category-2 column name
    exp_info : dict
        Experiment info dictionary
    dt : float
        Time step in milliseconds
    config : dict
        Configuration dictionary

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing category-2 metrics plot
    """
    plots_start = DYNAMICS_LAYOUT_EXTENDED["plots_start"]
    left_width = DYNAMICS_LAYOUT_EXTENDED["left_plots_width"]
    gap_width = DYNAMICS_LAYOUT_EXTENDED["gap_width"]
    right_width = DYNAMICS_LAYOUT_EXTENDED["right_plots_width"]
    lower_top = DYNAMICS_LAYOUT_EXTENDED["right_lower_top"]
    lower_bottom = DYNAMICS_LAYOUT_EXTENDED["right_lower_bottom"]

    panel_height = lower_top - lower_bottom
    ax = fig.add_axes(
        [
            plots_start + left_width + gap_width,
            lower_bottom,
            right_width,
            panel_height,
        ]
    )

    if df_category2 is None or category2 is None:
        ax.text(
            0.5,
            0.5,
            "Category-2 data not provided",
            ha="center",
            va="center",
            fontsize=RESPONSES_FORMATTING["fontsize_label"],
            alpha=0.7,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    # Extract category-2 values
    cat2_values = _extract_dimension_values(
        df_category2, "category2", category2, config
    )

    # Generate plasma colors for category-2
    plasma_cmap = plt.cm.plasma
    n_colors = len(cat2_values)
    colors_cat2 = {
        str(val): plasma_cmap(i / (n_colors - 1) if n_colors > 1 else 0.5)
        for i, val in enumerate(cat2_values)
    }

    # Calculate metrics (same approach as create_panel_d_extended)
    metric = exp_info["metric"]
    metric_col = f"{focus_layer}_{metric}"

    df = df_category2.copy()
    if metric_col not in df.columns:
        df = calculate_metrics(
            df=df,
            focus_layer=focus_layer,
            parameter=parameter,
            category=category2,
            metric=metric,
            dt=dt,
        )

    df[category2] = df[category2].astype(str)
    cat2_values_str = [str(val) for val in cat2_values]

    # Check if parameter is temporal and needs conversion to ms
    is_temporal = exp_info["type"].lower() in DT_CONVERT_PARAMS and dt is not None

    # Create x-axis column (converted to ms if temporal)
    x_col = f"{parameter}_ms" if is_temporal else parameter
    if is_temporal:
        df[x_col] = df[parameter].astype(float) * dt

    if not df[metric_col].notna().any():
        ax.text(
            0.5,
            0.5,
            f"No metric data for layer {focus_layer}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    # Plot using seaborn lineplot
    sns.lineplot(
        data=df,
        x=x_col,
        y=metric_col,
        ax=ax,
        marker="o",
        hue=category2,
        hue_order=cat2_values_str,
        palette=colors_cat2,
        markersize=8,
        linewidth=2,
        err_style="bars",
        errorbar="se",
    )

    # Add reference line at 1.0 for peak_ratio metric
    if metric == "peak_ratio":
        ax.axhline(
            1.0,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            zorder=1,
        )
    elif metric == "summed_response":
        # Linear prediction (extrapolated from longest duration)
        x_sorted = df[x_col].sort_values()
        linear_prediction = x_sorted / df[x_col].max()
        ax.plot(
            x_sorted,
            linear_prediction,
            "k--",
            linewidth=1.5,
            alpha=0.5,
            zorder=2,
        )
        ax.set_xlim(0, df[x_col].max())
        ax.set_ylim(0, 1)

    # Styling - create proper x-axis label with naming from config
    param_display = exp_info.get("full_label", get_display_name(parameter, config))
    metric_display = exp_info.get("metric_label", metric.replace("_", " ").title())

    # Build x-axis label with units for temporal parameters
    if is_temporal:
        # Format like "Stimulus Interval (**I**) (ms)"
        x_label = f"{param_display} (ms)"
    else:
        x_label = param_display

    ax.set_xlabel(x_label, fontsize=RESPONSES_FORMATTING["fontsize_label"])
    ax.set_ylabel(
        metric_display,
        fontsize=RESPONSES_FORMATTING["fontsize_label"],
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=RESPONSES_FORMATTING["fontsize_tick"])
    ax.grid(True, alpha=0.3, linestyle=":", zorder=1)
    sns.despine(ax=ax, left=True, bottom=True)

    # Add legend with plasma-colored lines
    legend = ax.legend(
        fontsize=RESPONSES_FORMATTING["fontsize_legend"],
        frameon=False,
        loc="best",
        title=get_display_name(category2, config),
        title_fontsize=RESPONSES_FORMATTING["fontsize_legend"],
    )
    # Make legend title bold
    if legend.get_title():
        legend.get_title().set_fontweight("bold")

    return ax


def plot_unified_dynamics_with_groen(
    df,
    focus_layer,
    *,
    parameter="tau",
    category="interval",
    experiment=None,
    save_path=None,
    dt=None,
    config=None,
    groen_data_path=None,
    df_category2=None,
    category2=None,
    time_offset=0.0,
):
    """Create unified dynamics visualization with Groen data and optional category-2.

    When category-2 data is provided, creates a 5-panel figure (A, B on left;
    C, D, E on right). Otherwise creates the standard 4-panel layout.

    Parameters
    ----------
    df : pd.DataFrame
        Model response data
    focus_layer : str
        Layer to focus on for panel B and D
    parameter : str
        Parameter being varied
    category : str
        Category dimension for coloring
    experiment : str, optional
        Experiment name (used to determine which Groen figure to show)
    save_path : str or Path, optional
        Path to save figure
    dt : float, optional
        Time step in milliseconds
    config : dict, optional
        Configuration dictionary with palette, naming, ordering
    groen_data_path : str or Path, optional
        Path to directory containing Groen CSV files
    df_category2 : pd.DataFrame, optional
        Category-2 response data for Panel E
    category2 : str, optional
        Category-2 column name for Panel E
    """
    logger.info("=" * 60)
    logger.info("Starting unified dynamics plotting with Groen data")
    logger.info("=" * 60)

    # Validate inputs
    if parameter not in df.columns:
        raise ValueError(f"Parameter column '{parameter}' not found")
    if category not in df.columns:
        raise ValueError(f"Category column '{category}' not found")

    # Initialize config if not provided
    if config is None:
        config = {"palette": {}, "naming": {}, "ordering": {}}

    # Validate and set Groen data path
    if groen_data_path is None:
        logger.warning(
            "No Groen data path provided. Panel C will show placeholder text. "
            "Use --groen-data to specify the path to Groen et al. 2022 CSV files."
        )
    else:
        groen_data_path = Path(groen_data_path)
        if not groen_data_path.exists():
            logger.warning(
                f"Groen data path does not exist: {groen_data_path}. "
                f"Panel C will show placeholder text."
            )
            groen_data_path = None

    # Determine experiment info
    experiment = experiment or parameter
    exp_info = determine_experiment_info(experiment, config)

    # Extract dimension values
    layer_names = _extract_dimension_values(df, "layers", "layers", config)
    hue_values = _extract_dimension_values(df, "category", category, config)
    param_values = _extract_dimension_values(df, "parameter", parameter, config)

    logger.info(f"Layers: {layer_names}")
    logger.info(f"Category values: {hue_values}")
    logger.info(f"Parameter values: {param_values}")
    logger.info(f"Groen data path: {groen_data_path}")

    # Determine if we're using extended layout (5 panels with category-2)
    use_extended_layout = df_category2 is not None and category2 is not None
    logger.info(f"Using extended layout: {use_extended_layout}")

    # Get colors
    colors = _get_colors_for_dimension(hue_values, category, config)

    # Select middle parameter value
    middle_idx = len(param_values) // 2
    middle_param_value = param_values[middle_idx]
    logger.info(f"Using middle parameter value: {middle_param_value}")

    # Convert time if needed
    if dt is not None:
        df = df.copy()
        df["time_ms"] = df["times_index"] * dt + time_offset
        if df_category2 is not None:
            df_category2 = df_category2.copy()
            df_category2["time_ms"] = df_category2["times_index"] * dt + time_offset

    # Create figure
    fig = plt.figure(
        figsize=(DYNAMICS_LAYOUT["figure_width"], DYNAMICS_LAYOUT["figure_height"])
    )
    sns.set_style("ticks")
    sns.set_context("talk")

    # Add panel labels (with E if using extended layout)
    if use_extended_layout:
        add_panel_labels_extended(fig)
    else:
        add_panel_labels(fig)

    # Create panels
    logger.info("Creating Panel A: Layers")
    panel_a_axes = create_panel_a(
        fig,
        df,
        layer_names,
        parameter,
        category,
        middle_param_value,
        hue_values,
        colors,
        dt,
        config,
    )

    logger.info("Creating Panel B: Parameter variation")
    panel_b_axes = create_panel_b(
        fig,
        df,
        focus_layer,
        parameter,
        category,
        param_values,
        hue_values,
        colors,
        dt,
        config,
    )

    if use_extended_layout:
        # Use extended 3-panel right side layout
        logger.info("Creating Panel C: Groen et al. 2022 data (extended layout)")
        panel_c_ax = create_panel_c_extended(fig, experiment, groen_data_path)

        logger.info("Creating Panel D: Summary metrics (extended layout)")
        panel_d_ax = create_panel_d_extended(
            fig,
            df,
            focus_layer,
            parameter,
            category,
            exp_info,
            hue_values,
            colors,
            dt,
            config,
        )

        logger.info("Creating Panel E: Category-2 metrics")
        panel_e_ax = create_panel_e(
            fig,
            df_category2,
            focus_layer,
            parameter,
            category2,
            exp_info,
            dt,
            config,
        )

        # Sync x-axis limits between panels D and E
        xlim_d = panel_d_ax.get_xlim()
        xlim_e = panel_e_ax.get_xlim()
        shared_xlim = (min(xlim_d[0], xlim_e[0]), max(xlim_d[1], xlim_e[1]))
        panel_d_ax.set_xlim(shared_xlim)
        panel_e_ax.set_xlim(shared_xlim)

        # Sync x-ticks between panels D and E
        panel_d_ax.set_xticks(panel_e_ax.get_xticks())

        # Add layer circle indicator between panels D and E
        layout = DYNAMICS_LAYOUT_EXTENDED
        circle_x = (
            layout["plots_start"]
            + layout["left_plots_width"]
            + layout["gap_width"]
            + layout["right_plots_width"] / 2
        )
        # Position vertically between D and E (in the gap)
        circle_y = (layout["right_middle_bottom"] + layout["right_lower_top"]) / 2

        # Get color for the focus layer
        layer_colors = _get_colors_for_dimension([focus_layer], "layers", config)
        layer_color = layer_colors.get(focus_layer, "#808080ff")
        pad = 0.5 if focus_layer == "IT" else 0.4

        fig.text(
            circle_x,
            circle_y,
            focus_layer,
            horizontalalignment="center",
            verticalalignment="center",
            transform=fig.transFigure,
            bbox=dict(
                boxstyle=f"circle,pad={pad}",
                facecolor=layer_color,
                edgecolor="#353535ff",
                linewidth=2,
                alpha=0.8,
            ),
            fontsize=RESPONSES_FORMATTING["fontsize_label"],
            fontweight="bold",
        )
    else:
        # Use standard 2-panel right side layout
        logger.info("Creating Panel C: Groen et al. 2022 data")
        panel_c_ax = create_panel_c_with_groen(fig, experiment, groen_data_path)

        logger.info("Creating Panel D: Summary metrics")
        panel_d_ax = create_panel_d(
            fig,
            df,
            focus_layer,
            parameter,
            category,
            exp_info,
            hue_values,
            colors,
            dt,
            config,
        )

    # Add horizontal legend
    legend_left = DYNAMICS_LAYOUT["plots_start"]
    legend_width = DYNAMICS_LAYOUT["left_plots_width"]
    legend_height = DYNAMICS_LAYOUT["legend_height"] * 1.2
    legend_bottom = (
        DYNAMICS_LAYOUT["lower_top"] + 0.5 * DYNAMICS_LAYOUT["legend_margin"]
    )

    _add_horizontal_legend(
        fig=fig,
        hue_var="category",
        hue_key=category,
        hue_values=hue_values,
        colors=colors,
        config=config,
        dt=dt or 1.0,
        legend_bot=legend_bottom,
        **(
            RESPONSES_FORMATTING
            | {
                "legend_height": legend_height,
                "left_margin": legend_left,
                "column_width": legend_width,
                "max_elements": 4,
            }
        ),
    )

    # Save figure
    if save_path:
        logger.info(f"Saving figure to {save_path}")
        save_plot(file_path=Path(save_path), dpi=300)

    logger.info("=" * 60)
    logger.info("Dynamics plotting with Groen data complete")
    logger.info("=" * 60)

    return fig


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create unified dynamics visualization with Groen et al. 2022 data"
    )
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to response data CSV file(s). Multiple paths will be concatenated.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to save output figure"
    )
    parser.add_argument(
        "--parameter", type=str, required=True, help="Parameter column name"
    )
    parser.add_argument("--experiment", type=str, help="Experiment name")
    parser.add_argument(
        "--category", type=str, required=True, help="Category column name"
    )
    parser.add_argument(
        "--focus-layer", type=str, default="V1", help="Layer to focus on"
    )
    parser.add_argument("--dt", type=float, help="Time step in milliseconds")
    parser.add_argument("--idle-timesteps", type=int, default=0, help="Number of idle timesteps before recorded data (shifts time axis)")
    parser.add_argument("--palette", type=str, help="JSON color palette")
    parser.add_argument("--naming", type=str, help="JSON naming dict")
    parser.add_argument("--ordering", type=str, help="JSON ordering dict")
    parser.add_argument(
        "--groen-data",
        type=Path,
        help="Path to directory containing Groen CSV files",
    )
    parser.add_argument(
        "--data-category2",
        type=Path,
        nargs="*",
        help="Paths to category-2 CSV files for Panel E",
    )
    parser.add_argument(
        "--category2",
        type=str,
        help="Category-2 column name for Panel E",
    )

    args = parser.parse_args()

    # Load and concatenate data
    if len(args.data) > 1:
        logger.info(f"Loading and concatenating data from {len(args.data)} files")
        dfs = []
        for i, path in enumerate(args.data):
            logger.info(f"  Loading file {i+1}/{len(args.data)}: {path}")
            dfs.append(pd.read_csv(path))
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Concatenated total: {len(df)} rows")
    else:
        logger.info(f"Loading data from: {args.data[0]}")
        df = pd.read_csv(args.data[0])

    # Load category-2 data if provided
    df_category2 = None
    if args.data_category2:
        if len(args.data_category2) > 1:
            logger.info(
                f"Loading category-2 data from {len(args.data_category2)} files"
            )
            dfs = []
            for i, path in enumerate(args.data_category2):
                logger.info(f"  Loading file {i+1}/{len(args.data_category2)}: {path}")
                dfs.append(pd.read_csv(path))
            df_category2 = pd.concat(dfs, ignore_index=True)
            logger.info(f"Category-2 concatenated total: {len(df_category2)} rows")
        else:
            logger.info(f"Loading category-2 data from: {args.data_category2[0]}")
            df_category2 = pd.read_csv(args.data_category2[0])

    # Load config
    config = load_config_from_args(
        palette_str=args.palette,
        naming_str=args.naming,
        ordering_str=args.ordering,
    )

    # Compute time offset from idle timesteps
    time_offset = args.idle_timesteps * (args.dt if args.dt is not None else 1.0)

    # Create plot
    plot_unified_dynamics_with_groen(
        df=df,
        focus_layer=args.focus_layer,
        parameter=args.parameter,
        category=args.category,
        experiment=args.experiment,
        save_path=args.output,
        dt=args.dt,
        config=config,
        groen_data_path=args.groen_data,
        df_category2=df_category2,
        category2=args.category2,
        time_offset=time_offset,
    )


if __name__ == "__main__":
    main()
