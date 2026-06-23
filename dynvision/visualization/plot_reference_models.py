"""Plot reference model response dynamics in a three-column comparison.

Creates a three-column figure comparing reference models against DyRCNNx8:
  Column I   (experiment 1): CorNetRT  + DyRCNNx8
  Column II  (experiment 2): CordsNet  + DyRCNNx8
  Column III (experiment 3): CorNetRT  + CordsNet  + DyRCNNx8

Each column displays:
  (A) Performance subplot with per-model colored traces for top1-accuracy,
      top3-accuracy, and confidence.
  (B) Response ridge plots (layer-wise) with per-model colored traces.

CordsNet records 9 layers; a configurable index list selects 4
representative layers (default [1, 4, 7, 8]) so that the ridge plots
are visually aligned with the 4-layer models.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dynvision.visualization.plot_responses import (
    LAYOUT,
    FORMATTING,
    ERRORBAR_CONFIG,
    _extract_dimension_values,
    _resolve_measure_column,
)
from dynvision.utils.visualization_utils import (
    calculate_label_indicator,
    load_config_from_args,
    save_plot,
)

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# Default model colors
MODEL_COLORS = {
    "CorNetRT": "#e07b39",
    "CordsNet": "#4a90d9",
    "DyRCNNx8": "#c0c0c0",
}

# Three-column layout
REFMODEL_LAYOUT = {
    **LAYOUT,
    "title_bot": 0.92,
    "title_pad": 0.05,
    "column_width": 5,
    "column_spacing": 0.5,
    "accuracy_height": 0.15,
    "accuracy_pad": 0.05,
    "ridge_height": 0.65,
    "ridge_overlap": 0.4,
}

# Default CordsNet layer indices to select (0-based into ordered layer list)
DEFAULT_CORDSNET_LAYER_INDICES = [1, 4, 7, 8]


def _natural_sort_layers(layers: List[str]) -> List[str]:
    """Sort layers by trailing numeric suffix (e.g. layer0, layer1, ..., layer8)."""

    def _sort_key(name: str):
        match = re.search(r"(\d+)$", name)
        return int(match.group(1)) if match else float("inf")

    return sorted(layers, key=_sort_key)


def _select_layers(
    all_layers: List[str],
    indices: Optional[List[int]] = None,
) -> List[str]:
    """Sort layers numerically, then select a subset by index.

    Args:
        all_layers: Unordered list of layer names.
        indices: 0-based indices into the *numerically sorted* list.
            ``None`` keeps all.

    Returns:
        Selected layer names in numerically sorted order.
    """
    sorted_layers = _natural_sort_layers(all_layers)
    if indices is None:
        return list(reversed(sorted_layers))
    selected = [sorted_layers[i] for i in indices if i < len(sorted_layers)]
    if len(selected) < len(indices):
        logger.warning(
            "Some layer indices out of range: requested %s from %d layers",
            indices,
            len(sorted_layers),
        )
    return list(reversed(selected))


def _get_model_color(model_name: str, config: Dict) -> str:
    """Resolve color for a model name from config palette or defaults."""
    palette = config.get("palette", {})
    if model_name in palette:
        return palette[model_name]
    return MODEL_COLORS.get(model_name, "#808080")


# ── Performance panel ────────────────────────────────────────────────


def _plot_multimodel_accuracy_panel(
    ax: plt.Axes,
    model_datasets: List[Tuple[str, pd.DataFrame]],
    dt: float,
    time_offset: float,
    show_ylabel: bool,
    show_legend: bool,
    accuracy_cols: List[str],
    confidence_cols: List[str],
    config: Dict,
    **kwargs,
) -> None:
    """Plot accuracy/confidence traces for multiple models.

    Each model gets its own color.  Line style encodes the metric
    (solid = accuracy, dashed = top-3, dotted = confidence).

    Args:
        ax: Matplotlib axes.
        model_datasets: List of ``(model_name, dataframe)`` tuples.
        dt: Temporal resolution in ms.
        show_ylabel: Whether to show y-axis label.
        show_legend: Whether to show linestyle legend.
        accuracy_cols: Accuracy metric column names.
        confidence_cols: Confidence metric column names.
        config: Configuration dict.
        **kwargs: Override FORMATTING defaults.
    """
    fmt = {**FORMATTING, **kwargs}
    ax.patch.set_alpha(0)

    accuracy_linestyles = ["-", "--", "-."]
    confidence_linestyles = [":", (0, (3, 2)), (0, (1, 2))]

    plotted_accuracy_styles: List[Tuple[str, str]] = []  # (column, linestyle)
    plotted_confidence_styles: List[Tuple[str, str]] = []

    for model_name, df in model_datasets:
        color = _get_model_color(model_name, config)
        data_plot = df.copy()
        data_plot["time_ms"] = data_plot["times_index"] * dt + time_offset

        for idx, column in enumerate(accuracy_cols):
            resolved = _resolve_measure_column(data_plot, column)
            if not resolved:
                continue
            linestyle = accuracy_linestyles[idx % len(accuracy_linestyles)]
            sns.lineplot(
                data=data_plot,
                x="time_ms",
                y=resolved,
                ax=ax,
                legend=False,
                linewidth=fmt["linewidth_main"],
                alpha=fmt["alpha_line"],
                linestyle=linestyle,
                color=color,
                **ERRORBAR_CONFIG,
            )
            if (column, linestyle) not in plotted_accuracy_styles:
                plotted_accuracy_styles.append((column, linestyle))

        for idx, column in enumerate(confidence_cols):
            resolved = _resolve_measure_column(data_plot, column)
            if not resolved:
                continue
            linestyle = confidence_linestyles[idx % len(confidence_linestyles)]
            sns.lineplot(
                data=data_plot,
                x="time_ms",
                y=resolved,
                ax=ax,
                legend=False,
                linewidth=fmt["linewidth_main"],
                alpha=fmt["alpha_line"],
                linestyle=linestyle,
                color=color,
                **ERRORBAR_CONFIG,
            )
            if (column, linestyle) not in plotted_confidence_styles:
                plotted_confidence_styles.append((column, linestyle))

    # Styling
    ax.set_ylim(-0.01, 1.01)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    if show_ylabel:
        ax.set_ylabel("Performance", fontsize=fmt["fontsize_axis"], fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.set_xlabel("Time (ms)", fontsize=fmt["fontsize_axis"])
    ax.tick_params(labelsize=fmt["fontsize_tick"])

    # Linestyle legend (metric encoding, not model identity)
    if show_legend:
        handles, labels = [], []
        for column, linestyle in plotted_accuracy_styles:
            label = (
                "Accuracy"
                if len(plotted_accuracy_styles) == 1
                else f"Accuracy ({column})"
            )
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=fmt["linewidth_main"],
                    linestyle=linestyle,
                    alpha=fmt["alpha_line"],
                )
            )
            labels.append(label)
        for column, linestyle in plotted_confidence_styles:
            label = (
                "Confidence"
                if len(plotted_confidence_styles) == 1
                else f"Confidence ({column})"
            )
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=fmt["linewidth_main"],
                    linestyle=linestyle,
                    alpha=fmt["alpha_line"],
                )
            )
            labels.append(label)
        if handles:
            ax.legend(
                handles=handles,
                labels=labels,
                loc="best",
                frameon=False,
                fontsize=fmt["fontsize_legend"] - 1,
            )

    # Label indicator — derive from label_index (>= 0 means valid)
    li_candidates = [
        df_li for _, df_li in model_datasets if "label_index" in df_li.columns
    ]
    if li_candidates:
        li_df_src = max(li_candidates, key=lambda d: d["times_index"].nunique())
        li_work = li_df_src.copy()
        li_work["label_valid"] = (li_work["label_index"] >= 0).astype(int)
        try:
            label_indicator_df = calculate_label_indicator(
                li_work,
                "label_valid",
                ax.get_ylim(),
                0.1,
            )
            indicator_time = label_indicator_df["times_index"] * dt + time_offset
            ax.plot(
                indicator_time,
                label_indicator_df["label_indicator"],
                color="black",
                linewidth=fmt["linewidth_indicator"],
                drawstyle="steps-mid",
                alpha=0.6,
            )
        except Exception as e:
            logger.debug("Could not calculate label indicator: %s", e)

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=True, bottom=True)


# ── Multi-model ridge plots ─────────────────────────────────────────


def _plot_multimodel_ridges(
    fig: plt.Figure,
    column_left: float,
    column_width: float,
    model_datasets: List[Tuple[str, pd.DataFrame, List[str]]],
    dt: float,
    time_offset: float,
    show_ylabel: bool,
    config: Dict,
    ridge_top: float,
    **kwargs,
) -> List[plt.Axes]:
    """Plot ridge plots with multiple models overlaid per layer row.

    Each row corresponds to a layer index (matched across models by
    position).  Models are distinguished by color.

    Args:
        fig: Matplotlib figure.
        column_left: Left position of column in figure coords.
        column_width: Width of column in figure coords.
        model_datasets: List of ``(model_name, dataframe, layers)`` where
            *layers* is the ordered list of layer names for that model
            (already filtered if needed).  All entries must have the same
            length.
        dt: Temporal resolution in ms.
        show_ylabel: Whether to show y-axis label.
        config: Configuration dict.
        ridge_top: Top position for the first ridge row.
        **kwargs: Override FORMATTING / LAYOUT defaults.

    Returns:
        List of created axes.
    """
    layout = {
        **REFMODEL_LAYOUT,
        **{k: v for k, v in kwargs.items() if k in REFMODEL_LAYOUT},
    }
    fmt = {**FORMATTING, **{k: v for k, v in kwargs.items() if k in FORMATTING}}

    # Determine the number of rows from the first model
    n_rows = (
        max(len(layers) for _, _, layers in model_datasets) if model_datasets else 0
    )
    if n_rows == 0:
        return []

    spacing = layout["ridge_height"] / n_rows * (1 - layout["ridge_overlap"])
    plot_height = layout["ridge_height"] / n_rows * 1.4

    global_ymin, global_ymax = fmt["max_global_ymax"], fmt["min_global_ymin"]

    axes = []
    layer_label_info: List[Tuple[plt.Axes, List[str]]] = []

    for row_idx in range(n_rows):
        top_pos = ridge_top - row_idx * spacing
        bottom_pos = top_pos - plot_height
        ax = fig.add_axes([column_left, bottom_pos, column_width, plot_height])
        ax.patch.set_alpha(0)
        axes.append(ax)

        ax.axhline(0, color="gray", linestyle=":", alpha=0.7, linewidth=1)

        layer_names_this_row = []

        for model_name, df, layers in model_datasets:
            if row_idx >= len(layers):
                continue
            layer = layers[row_idx]
            response_col = f"{layer}_response_avg"
            if response_col not in df.columns:
                logger.warning(
                    "Response column '%s' not found for model '%s'",
                    response_col,
                    model_name,
                )
                continue

            data_plot = df.copy()
            data_plot["time_ms"] = data_plot["times_index"] * dt + time_offset
            color = _get_model_color(model_name, config)

            sns.lineplot(
                data=data_plot,
                x="time_ms",
                y=response_col,
                ax=ax,
                legend=False,
                linewidth=fmt["linewidth_main"],
                alpha=fmt["alpha_line"],
                color=color,
                **ERRORBAR_CONFIG,
            )

            if len(str(layer)) <= 3:
                layer_name = layer.upper()
            else:
                layer_name = layer.capitalize()
            if layer_name not in layer_names_this_row:
                layer_names_this_row.append(layer_name)

        # Sort labels: numbered layers first (Layer1, Layer4, ...) then named (V1, IT, ...)
        def _label_sort_key(name):
            if re.search(r"\d+$", name):
                return (0, name)  # numbered layers first
            return (1, name)  # named layers second

        layer_names_this_row.sort(key=_label_sort_key)

        layer_label_info.append((ax, layer_names_this_row))

        # Track y-limits (x stays per-column)
        ymin, ymax = ax.get_ylim()
        global_ymin = max(min(global_ymin, ymin), fmt["min_global_ymin"])
        global_ymax = min(max(global_ymax, ymax), fmt["max_global_ymax"])

        # Label indicator — only on the bottom ridge subplot
        if row_idx == n_rows - 1 and model_datasets:
            li_candidates = [
                df_li
                for _, df_li, _ in model_datasets
                if "label_index" in df_li.columns
            ]
            if li_candidates:
                li_df_src = max(
                    li_candidates, key=lambda d: d["times_index"].nunique()
                )
                li_work = li_df_src.copy()
                li_work["label_valid"] = (li_work["label_index"] >= 0).astype(int)
                try:
                    li_df = calculate_label_indicator(
                        li_work,
                        "label_valid",
                        (0, global_ymax),
                        0.1,
                    )
                    ax.plot(
                        li_df["times_index"] * dt + time_offset,
                        li_df["label_indicator"],
                        color="black",
                        linewidth=fmt["linewidth_indicator"],
                        drawstyle="steps-mid",
                        alpha=0.6,
                    )
                except Exception:
                    pass

        # Y-label on middle row
        if show_ylabel and row_idx == n_rows // 2:
            ax.set_ylabel(
                "Average Response",
                fontsize=fmt["fontsize_axis"],
                fontweight="bold",
                labelpad=6,
            )
        else:
            ax.set_ylabel("")

        # X-axis only on bottom
        if row_idx < n_rows - 1:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Time (ms)", fontsize=fmt["fontsize_axis"])
            ax.tick_params(labelsize=fmt["fontsize_tick"])

        sns.despine(ax=ax, left=True, bottom=True)

    # Unify y-limits and add layer labels
    for ax, layer_names in layer_label_info:
        ax.set_ylim(global_ymin, global_ymax)

        # y-tick scaling
        yticks_map = [
            (10.0, [0, 10], ["0", "10"]),
            (5.0, [0, 5], ["0", "5"]),
            (1.0, [0, 1], ["0", "1"]),
            (0.5, [0, 0.5], ["0", "0.5"]),
            (0.1, [0, 0.1], ["0", "0.1"]),
            (0.05, [0, 0.05], ["0", "0.05"]),
            (0.01, [0, 0.01], ["0", "0.01"]),
        ]
        for limit, ticks, labels in yticks_map:
            if global_ymax > limit:
                ax.set_yticks(ticks)
                ax.set_yticklabels(labels, fontsize=fmt["fontsize_tick"])
                break

        # Layer label circle / badge
        label_text = " / ".join(layer_names) if layer_names else ""
        if label_text:
            ax.text(
                0.95,
                0.25,
                label_text,
                ha="right",
                va="center",
                zorder=10,
                transform=ax.transAxes,
                fontsize=fmt["fontsize_label"],
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.8,
                ),
            )

    return axes


# ── Loss timestep markers ────────────────────────────────────────────


def _add_loss_timestep_markers(
    ax: plt.Axes,
    loss_timesteps: List[Tuple[str, List[int]]],
    dt: float,
    config: Dict,
    time_offset: float = 0.0,
) -> None:
    """Add tiny colored markers on the x-axis for category loss timesteps.

    Args:
        ax: Matplotlib axes to annotate.
        loss_timesteps: List of ``(model_name, timestep_list)`` tuples.
        dt: Temporal resolution in ms per timestep.
        config: Configuration dict for color lookup.
        time_offset: Time offset in ms for idle timesteps.
    """
    ymin = ax.get_ylim()[0]
    for model_name, timesteps in loss_timesteps:
        color = _get_model_color(model_name, config)
        times_ms = [t * dt + time_offset for t in timesteps]
        mkr = "*"
        sz = 100
        ax.scatter(
            times_ms,
            [ymin] * len(times_ms),
            marker=mkr,
            s=sz,
            linewidths=1.5,
            color=color,
            zorder=5,
            clip_on=False,
        )


# ── Main plotting function ──────────────────────────────────────────


def plot_reference_models(
    data_cornetrt_col1: List[Path],
    data_cordsnet_col2: List[Path],
    data_cornetrt_col3: List[Path],
    data_cordsnet_col3: List[Path],
    ref_data_col1: List[Path],
    ref_data_col2: List[Path],
    ref_data_col3: List[Path],
    output: Path,
    experiment_col1: Optional[str] = None,
    experiment_col2: Optional[str] = None,
    experiment_col3: Optional[str] = None,
    model_name_col1: str = "CorNetRT",
    model_name_col2: str = "CordsNet",
    accuracy_measure: str = "accuracy,accuracy_top3",
    confidence_measure: str = "first_label_confidence",
    cordsnet_layer_indices: Optional[List[int]] = None,
    dt: float = 2.0,
    config: Optional[Dict] = None,
) -> plt.Figure:
    """Create a three-column figure comparing reference models.

    Args:
        data_cornetrt_col1: CorNetRT test_data.csv paths for experiment I.
        data_cordsnet_col2: CordsNet test_data.csv paths for experiment II.
        data_cornetrt_col3: CorNetRT test_data.csv paths for experiment III.
        data_cordsnet_col3: CordsNet test_data.csv paths for experiment III.
        ref_data_col1: DyRCNNx8 test_data.csv paths for experiment I.
        ref_data_col2: DyRCNNx8 test_data.csv paths for experiment II.
        ref_data_col3: DyRCNNx8 test_data.csv paths for experiment III.
        output: Output figure path.
        experiment_col1: Experiment name for column I.
        experiment_col2: Experiment name for column II.
        experiment_col3: Experiment name for column III.
        model_name_col1: Display name for CorNetRT.
        model_name_col2: Display name for CordsNet.
        accuracy_measure: Comma-separated accuracy column names.
        confidence_measure: Comma-separated confidence column names.
        cordsnet_layer_indices: 0-based indices to select from CordsNet's
            ordered layer list (default: [1, 4, 7, 8]).
        dt: Temporal resolution in ms per timestep.
        config: Configuration dict with palette, naming, ordering.

    Returns:
        The matplotlib figure.
    """
    logger.info("=" * 60)
    logger.info("Starting reference model comparison plot (3 columns)")
    logger.info("=" * 60)

    if config is None:
        config = {"palette": {}, "naming": {}, "ordering": {}}
    if cordsnet_layer_indices is None:
        cordsnet_layer_indices = DEFAULT_CORDSNET_LAYER_INDICES

    layout = REFMODEL_LAYOUT
    fmt = FORMATTING

    accuracy_cols = [c.strip() for c in accuracy_measure.split(",") if c.strip()]
    confidence_cols = [c.strip() for c in confidence_measure.split(",") if c.strip()]

    # ── Load data ────────────────────────────────────────────────────
    def _load_concat(paths):
        dfs = [pd.read_csv(p) for p in paths]
        return pd.concat(dfs, ignore_index=True)

    df_cornetrt_1 = _load_concat(data_cornetrt_col1)
    df_cordsnet_2 = _load_concat(data_cordsnet_col2)
    df_cornetrt_3 = _load_concat(data_cornetrt_col3)
    df_cordsnet_3 = _load_concat(data_cordsnet_col3)
    df_ref_1 = _load_concat(ref_data_col1)
    df_ref_2 = _load_concat(ref_data_col2)
    df_ref_3 = _load_concat(ref_data_col3)

    # ── Extract layers ───────────────────────────────────────────────
    cornetrt_layers = _extract_dimension_values(
        df_cornetrt_1, "layers", "layers", config
    )
    all_cordsnet_layers = _extract_dimension_values(
        df_cordsnet_2, "layers", "layers", config
    )
    cordsnet_layers = _select_layers(all_cordsnet_layers, cordsnet_layer_indices)
    ref_layers = _extract_dimension_values(df_ref_1, "layers", "layers", config)

    logger.info("CorNetRT layers: %s", cornetrt_layers)
    logger.info("CordsNet layers (all): %s", all_cordsnet_layers)
    logger.info("CordsNet layers (selected): %s", cordsnet_layers)
    logger.info("DyRCNNx8 layers: %s", ref_layers)

    # ── Column definitions ───────────────────────────────────────────
    # Each column: (title, experiment, model_datasets_for_accuracy,
    #               model_datasets_for_ridges, model_names_for_legend)
    #
    # model_datasets_for_ridges: [(model_name, df, layers), ...]
    ref_name = "DyRCNNx8"

    # Compute experiment subtitles from stim timing
    def _make_subtitle(df: pd.DataFrame) -> str:
        stim = int(df["stim"].iloc[0])
        total = int(df["times_index"].max()) + 1
        null_ms = int((total - stim) * dt)
        image_ms = int(stim * dt)
        return f"{image_ms} ms Image + {null_ms} ms Null"

    columns = [
        {
            "title": model_name_col1,
            "subtitle": _make_subtitle(df_cornetrt_1),
            "experiment": experiment_col1,
            "accuracy_models": [
                (model_name_col1, df_cornetrt_1),
                (ref_name, df_ref_1),
            ],
            "ridge_models": [
                (model_name_col1, df_cornetrt_1, cornetrt_layers),
                (ref_name, df_ref_1, ref_layers),
            ],
            "legend_models": [model_name_col1, ref_name],
            "loss_timesteps": [(model_name_col1, [1])],
            "idle_timesteps": 0,
        },
        {
            "title": model_name_col2,
            "subtitle": _make_subtitle(df_cordsnet_2),
            "experiment": experiment_col2,
            "accuracy_models": [
                (model_name_col2, df_cordsnet_2),
                (ref_name, df_ref_2),
            ],
            "ridge_models": [
                (model_name_col2, df_cordsnet_2, cordsnet_layers),
                (ref_name, df_ref_2, ref_layers),
            ],
            "legend_models": [model_name_col2, ref_name],
            "loss_timesteps": [(model_name_col2, list(range(70, 101)))],
            "idle_timesteps": 100,
        },
        {
            "title": f"{model_name_col1} + {model_name_col2}",
            "subtitle": _make_subtitle(df_cornetrt_3),
            "experiment": experiment_col3,
            "accuracy_models": [
                (model_name_col1, df_cornetrt_3),
                (model_name_col2, df_cordsnet_3),
                (ref_name, df_ref_3),
            ],
            "ridge_models": [
                (model_name_col1, df_cornetrt_3, cornetrt_layers),
                (model_name_col2, df_cordsnet_3, cordsnet_layers),
                (ref_name, df_ref_3, ref_layers),
            ],
            "legend_models": [model_name_col1, model_name_col2, ref_name],
            "loss_timesteps": [],
            "idle_timesteps": 20,
        },
    ]

    # ── Figure setup ─────────────────────────────────────────────────
    n_columns = len(columns)
    total_width = (
        n_columns * layout["column_width"] + (n_columns - 1) * layout["column_spacing"]
    )
    fig = plt.figure(figsize=(total_width, layout["figure_height"]))
    sns.set_context("talk")

    relative_col_width = layout["column_width"] / total_width
    relative_spacing = layout["column_spacing"] / total_width

    all_ridge_axes: List[plt.Axes] = []
    col_labels = ["I", "II", "III"]
    col_idle_timesteps = [col_def["idle_timesteps"] for col_def in columns]

    for col_idx, col_def in enumerate(columns):
        column_left = col_idx * (relative_col_width + relative_spacing)
        show_ylabel = col_idx == 0

        logger.info("Processing column %s: %s", col_labels[col_idx], col_def["title"])

        # Title (two lines: model name + timing subtitle)
        title_ax = fig.add_axes(
            [
                column_left,
                layout["title_bot"],
                relative_col_width,
                layout["title_pad"],
            ]
        )
        title_ax.text(
            0.5,
            0.8,
            col_def["title"],
            ha="center",
            va="center",
            fontsize=fmt["fontsize_title"],
            fontweight="bold",
        )
        title_ax.text(
            0.5,
            0.0,
            col_def["subtitle"],
            ha="center",
            va="center",
            fontsize=fmt["fontsize_title"] - 1,
            fontstyle="italic",
        )
        title_ax.set_xlim(0, 1)
        title_ax.set_ylim(0, 1)
        title_ax.axis("off")

        # Column label
        title_ax.text(
            -0.05,
            0.7,
            col_labels[col_idx],
            ha="right",
            va="center",
            transform=title_ax.transAxes,
            fontsize=fmt["fontsize_title"] + 2,
            fontweight="bold",
        )

        # (A) Performance panel
        accuracy_bot = (
            layout["title_bot"] - layout["title_pad"] - layout["accuracy_height"]
        )
        acc_ax = fig.add_axes(
            [
                column_left,
                accuracy_bot,
                relative_col_width,
                layout["accuracy_height"],
            ]
        )
        acc_ax.text(
            -0.03,
            1.1,
            "A",
            ha="right",
            va="bottom",
            transform=acc_ax.transAxes,
            fontsize=fmt["fontsize_label"],
            fontweight="bold",
        )

        time_offset = col_def["idle_timesteps"] * dt

        _plot_multimodel_accuracy_panel(
            ax=acc_ax,
            model_datasets=col_def["accuracy_models"],
            dt=dt,
            time_offset=time_offset,
            show_ylabel=show_ylabel,
            show_legend=(col_idx == 0),
            accuracy_cols=accuracy_cols,
            confidence_cols=confidence_cols,
            config=config,
        )

        # Loss timestep markers on accuracy panel
        if col_def.get("loss_timesteps"):
            _add_loss_timestep_markers(
                acc_ax,
                col_def["loss_timesteps"],
                dt,
                config,
                time_offset=time_offset,
            )

        # (B) Ridge plots — start below accuracy panel
        ridge_top = accuracy_bot - layout["accuracy_pad"]
        ridge_axes = _plot_multimodel_ridges(
            fig=fig,
            column_left=column_left,
            column_width=relative_col_width,
            model_datasets=col_def["ridge_models"],
            dt=dt,
            time_offset=time_offset,
            show_ylabel=show_ylabel,
            config=config,
            ridge_top=ridge_top,
        )

        if ridge_axes:
            ridge_axes[0].text(
                -0.03,
                0.60,
                "B",
                ha="right",
                va="top",
                transform=ridge_axes[0].transAxes,
                fontsize=fmt["fontsize_label"],
                fontweight="bold",
            )
            # Model color legend — vertical, framed, in top ridge of column I
            if col_idx == 0:
                all_model_names = [model_name_col1, model_name_col2, ref_name]
                handles = []
                labels = []
                for name in all_model_names:
                    color = _get_model_color(name, config)
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color=color,
                            linewidth=fmt["linewidth_main"],
                            alpha=fmt["alpha_line"],
                        )
                    )
                    labels.append(name)
                legend = ridge_axes[0].legend(
                    handles,
                    labels,
                    loc="upper left",
                    ncol=3,
                    frameon=False,
                    columnspacing=1.0,
                    framealpha=0.9,
                    edgecolor="gray",
                    fontsize=fmt["fontsize_legend"],
                    handlelength=1.5,
                    handletextpad=0.5,
                    title="Model",
                    title_fontsize=fmt["fontsize_legend"],
                )
                if legend.get_title() is not None:
                    legend.get_title().set_fontweight("bold")
        all_ridge_axes.extend(ridge_axes)

    # ── Align y-axes across all columns (x-axes stay independent) ──
    if all_ridge_axes:
        global_ymin = min(ax.get_ylim()[0] for ax in all_ridge_axes)
        global_ymax = max(ax.get_ylim()[1] for ax in all_ridge_axes)

        global_ymin = max(global_ymin, fmt["min_global_ymin"])
        global_ymax = min(global_ymax, fmt["max_global_ymax"])

        for ax in all_ridge_axes:
            ax.set_ylim(global_ymin, global_ymax)

        fig.align_ylabels(all_ridge_axes)

    logger.info("Saving figure to: %s", output)
    save_plot(output)
    logger.info("Plotting complete")

    return fig


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot reference model response dynamics (3-column comparison)"
    )

    # Column I: CorNetRT + DyRCNNx8
    parser.add_argument(
        "--data-cornetrt-col1",
        type=Path,
        nargs="+",
        required=True,
        help="CorNetRT test_data.csv for experiment I",
    )
    parser.add_argument(
        "--ref-data-col1",
        type=Path,
        nargs="+",
        required=True,
        help="DyRCNNx8 test_data.csv for experiment I",
    )

    # Column II: CordsNet + DyRCNNx8
    parser.add_argument(
        "--data-cordsnet-col2",
        type=Path,
        nargs="+",
        required=True,
        help="CordsNet test_data.csv for experiment II",
    )
    parser.add_argument(
        "--ref-data-col2",
        type=Path,
        nargs="+",
        required=True,
        help="DyRCNNx8 test_data.csv for experiment II",
    )

    # Column III: all three
    parser.add_argument(
        "--data-cornetrt-col3",
        type=Path,
        nargs="+",
        required=True,
        help="CorNetRT test_data.csv for experiment III",
    )
    parser.add_argument(
        "--data-cordsnet-col3",
        type=Path,
        nargs="+",
        required=True,
        help="CordsNet test_data.csv for experiment III",
    )
    parser.add_argument(
        "--ref-data-col3",
        type=Path,
        nargs="+",
        required=True,
        help="DyRCNNx8 test_data.csv for experiment III",
    )

    parser.add_argument(
        "--output", type=Path, required=True, help="Output figure path"
    )

    parser.add_argument("--experiment-col1", type=str, default=None)
    parser.add_argument("--experiment-col2", type=str, default=None)
    parser.add_argument("--experiment-col3", type=str, default=None)
    parser.add_argument("--model-name-col1", type=str, default="CorNetRT")
    parser.add_argument("--model-name-col2", type=str, default="CordsNet")

    parser.add_argument(
        "--accuracy-measure",
        type=str,
        default="accuracy,accuracy_top3",
        help="Comma-separated accuracy column names",
    )
    parser.add_argument(
        "--confidence-measure",
        type=str,
        default="first_label_confidence",
        help="Comma-separated confidence column names",
    )
    parser.add_argument(
        "--cordsnet-layer-indices",
        type=str,
        default="1,4,7,8",
        help="Comma-separated 0-based layer indices for CordsNet (default: 1,4,7,8)",
    )
    parser.add_argument("--dt", type=float, default=2.0, help="Time resolution (ms)")
    parser.add_argument(
        "--idle-timesteps",
        type=int,
        default=0,
        help="Number of idle timesteps before recorded data (shifts time axis by this many timesteps * dt)",
    )
    parser.add_argument("--palette", type=str, help="JSON color palette")
    parser.add_argument("--naming", type=str, help="JSON naming dict")
    parser.add_argument("--ordering", type=str, help="JSON ordering dict")
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args, unknown = parser.parse_known_args()

    logging.basicConfig(level=getattr(logging, "DEBUG"), force=False)
    if unknown:
        logger.info("Ignoring unknown arguments: %s", unknown)

    config = load_config_from_args(
        palette_str=args.palette,
        naming_str=args.naming,
        ordering_str=args.ordering,
    )

    cordsnet_indices = [int(x) for x in args.cordsnet_layer_indices.split(",")]

    plot_reference_models(
        data_cornetrt_col1=args.data_cornetrt_col1,
        data_cordsnet_col2=args.data_cordsnet_col2,
        data_cornetrt_col3=args.data_cornetrt_col3,
        data_cordsnet_col3=args.data_cordsnet_col3,
        ref_data_col1=args.ref_data_col1,
        ref_data_col2=args.ref_data_col2,
        ref_data_col3=args.ref_data_col3,
        output=args.output,
        experiment_col1=args.experiment_col1,
        experiment_col2=args.experiment_col2,
        experiment_col3=args.experiment_col3,
        model_name_col1=args.model_name_col1,
        model_name_col2=args.model_name_col2,
        accuracy_measure=args.accuracy_measure,
        confidence_measure=args.confidence_measure,
        cordsnet_layer_indices=cordsnet_indices,
        dt=args.dt,
        config=config,
    )


if __name__ == "__main__":
    main()
