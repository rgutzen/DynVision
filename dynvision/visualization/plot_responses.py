"""Plot temporal ridge responses with flexible dimension mapping.

This module provides a generalized plotting function for visualizing temporal
dynamics of neural network layer responses with three flexible dimensions:
- Vertical subplots (ridge plots)
- Color hues
- Columns

Each dimension can represent: layers, category, parameter, classifier_topk, classifier_top{N}, or any
column present in the dataframe (e.g. *_index columns) so long as it remains
categorical (by default, fewer than 60 unique values).
"""

import argparse
import ast
import json
import logging
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dynvision.utils.visualization_utils import (
    aggregate_plot_data,
    calculate_label_indicator,
    find_classifier_meta_columns,
    find_classifier_value_columns,
    find_layer_response_columns,
    get_color,
    get_display_name,
    get_ordering,
    load_config_from_args,
    order_layers,
    resolve_measure_columns,
    save_plot,
    standardize_category_value,
    standardize_series,
)

# Configure logging - simplified to prevent duplicate output
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


# Global layout configuration
LAYOUT = {
    # Figure dimensions
    "figure_height": 14,  # Total figure height in inches
    "column_width": 7,  # Width of each column in inches
    "column_spacing": 0.25,  # Spacing between columns in inches
    # Vertical layout in % of figure size (bot = bottom position, pad = whitespace below element)
    "title_bot": 0.96,
    "title_pad": 0.02,  # Space below title
    "accuracy_pad": 0.02,  # Space below accuracy panel
    "accuracy_height": 0.13,  # Fixed height for accuracy panel
    "legend_pad": 0.02,  # Space below legend
    "legend_height": 0.1,  # Fixed height for legend
    "ridge_height": 0.65,  # Total height for all ridge plots
    "ridge_overlap": 0.25,  # Overlap fraction between ridges
    # Margins
    "left_margin": 0.08,
    "right_margin": 0.02,
    "legend_margin": 0.02,
}

# Global formatting configuration
FORMATTING = {
    "fontsize_title": 18,
    "fontsize_axis": 16,
    "fontsize_tick": 14,
    "fontsize_legend": 16,
    "fontsize_label": 14,
    "linewidth_main": 2.5,
    "linewidth_indicator": 3,
    "alpha_line": 0.8,
    "alpha_indicator": 0.6,
    "greyscale_color": "#5a5a5a",
    "layer_circle_colors": {
        "V1": "#ff69b4ff",
        "V2": "#dda0ddff",
        "V4": "#da70d6ff",
        "IT": "#ba55d3ff",
    },
    "min_global_ymin": -0.005,
    "max_global_ymax": 4,
}

CLASSIFIER_TOP_PREFIX = "classifier_top"
SPECIAL_DIMENSIONS = {"layers", "classifier_topk"}
MAX_DIMENSION_VALUES = 15

# Cache classifier unit indices for legend formatting
CLASSIFIER_UNIT_ID_MAP: Dict[str, Optional[int]] = {}


@dataclass
class MeasureSpec:
    requested: str
    column: str


@dataclass
class ColumnPlan:
    needed_columns: List[str]
    accuracy_specs: List[MeasureSpec]
    confidence_specs: List[MeasureSpec]
    layer_columns: List[str]
    classifier_columns: List[str]
    classifier_meta_columns: List[str]
    dimension_columns: List[str]

    def value_columns(self) -> List[str]:
        columns = {spec.column for spec in self.accuracy_specs}
        columns.update(spec.column for spec in self.confidence_specs)
        columns.update(self.layer_columns)
        columns.update(self.classifier_columns)
        return sorted(columns)


def _read_available_columns(path: Path) -> List[str]:
    header_df = pd.read_csv(path, nrows=0)
    return list(header_df.columns)


def _create_measure_specs(
    available_columns: Sequence[str],
    requested: List[str],
    measure_type: str,
) -> List[MeasureSpec]:
    if not requested:
        return []

    resolved, missing = resolve_measure_columns(available_columns, requested)
    if missing:
        logger.warning(
            "%s measure(s) not found in data columns and will be skipped: %s",
            measure_type.capitalize(),
            ", ".join(missing),
        )
    return [MeasureSpec(requested=req, column=col) for req, col in resolved]


def _build_column_plan(
    available_columns: Sequence[str],
    subplot_var: str,
    hue_var: str,
    column_var: Optional[str],
    subplot_key: str,
    hue_key: str,
    column_key: Optional[str],
    accuracy_requests: List[str],
    confidence_requests: List[str],
    focus_layer: Optional[str] = None,
) -> ColumnPlan:
    available_set = set(available_columns)
    needed = {"times_index", "label_index"}

    missing_base = [col for col in needed if col not in available_set]
    if missing_base:
        raise ValueError(
            "Required base column(s) missing from data: " + ", ".join(missing_base)
        )

    def _require_dimension(
        dimension: Optional[str], key: Optional[str], label: str
    ) -> None:
        if dimension and not _is_special_dimension(dimension):
            if key is None:
                raise ValueError(f"{label}_key required for dimension '{dimension}'")
            if key not in available_set:
                raise ValueError(
                    f"Column '{key}' (for {label} dimension '{dimension}') not found in data"
                )
            needed.add(key)

    _require_dimension(subplot_var, subplot_key, "subplot")
    _require_dimension(hue_var, hue_key, "hue")
    _require_dimension(column_var, column_key, "column")

    accuracy_specs = _create_measure_specs(
        available_columns, accuracy_requests, measure_type="accuracy"
    )
    confidence_specs = _create_measure_specs(
        available_columns, confidence_requests, measure_type="confidence"
    )
    for spec in (*accuracy_specs, *confidence_specs):
        needed.add(spec.column)

    needs_layer_columns = (
        subplot_var == "layers"
        or hue_var == "layers"
        or (column_var == "layers")
        or bool(focus_layer)
    )
    layer_columns = (
        find_layer_response_columns(available_columns) if needs_layer_columns else []
    )
    if needs_layer_columns and not layer_columns:
        raise ValueError(
            "Layer response columns not found (expected *_response_avg) but required by layout"
        )
    needed.update(layer_columns)

    needs_classifier_columns = (
        subplot_var == "classifier_topk"
        or hue_var == "classifier_topk"
        or (column_var == "classifier_topk")
    )
    classifier_columns = (
        find_classifier_value_columns(available_columns)
        if needs_classifier_columns
        else []
    )
    classifier_meta_columns = (
        find_classifier_meta_columns(available_columns)
        if needs_classifier_columns
        else []
    )
    if needs_classifier_columns and not classifier_columns:
        raise ValueError(
            "Classifier response columns (classifier_top*) required but not found in data"
        )
    needed.update(classifier_columns)
    needed.update(classifier_meta_columns)

    dimension_columns: List[str] = []
    if subplot_var and not _is_special_dimension(subplot_var) and subplot_key:
        dimension_columns.append(subplot_key)
    if hue_var and not _is_special_dimension(hue_var) and hue_key:
        dimension_columns.append(hue_key)
    if column_var and not _is_special_dimension(column_var) and column_key:
        dimension_columns.append(column_key)

    return ColumnPlan(
        needed_columns=sorted(needed),
        accuracy_specs=accuracy_specs,
        confidence_specs=confidence_specs,
        layer_columns=layer_columns,
        classifier_columns=classifier_columns,
        classifier_meta_columns=classifier_meta_columns,
        dimension_columns=dimension_columns,
    )


def _format_legend_label(key: str, value: str, config: Dict, dt: float = 1.0) -> str:
    """Format legend label based on key type and config.

    Args:
        key: The parameter/category key
        value: The value to format
        config: Configuration dict with naming mappings
        dt: Temporal resolution in ms per timestep (for converting timestep values)

    Returns:
        Formatted label string
    """
    # First check if there's a direct naming translation for this value
    if config and "naming" in config:
        # Check both the value directly and the standardized value
        for check_val in [value, standardize_category_value(value)]:
            translated_name = get_display_name(check_val, config)
            if translated_name and translated_name != check_val:
                return translated_name

    # Timestep parameters that should be converted to ms
    timestep_params = ["tsteps", "idle"]
    # Numerical parameters that are already in ms
    ms_params = ["tau", "trc", "tsk", "lossrt"]

    if key.lower() in timestep_params:
        try:
            timestep_value = int(float(value))
            ms_value = timestep_value * dt
            return f"{ms_value:.0f} ms"
        except (ValueError, TypeError):
            return f"{value} ms"
    elif key.lower() in ms_params:
        try:
            numeric_value = int(float(value))
            return f"{numeric_value} ms"
        except (ValueError, TypeError):
            return f"{value} ms"

    # For other parameters, check if boolean
    if str(value).lower() in ["true", "false"]:
        return str(value).capitalize()

    # Default: capitalize first letter
    return str(value).capitalize()


def _validate_dimensions(
    subplot_var: str, hue_var: str, column_var: Optional[str]
) -> None:
    """Validate that dimension variables are unique."""
    used = [subplot_var, hue_var]
    if column_var:
        used.append(column_var)

    if len(used) != len(set(used)):
        raise ValueError(
            f"Dimension variables must be unique. Got: subplot={subplot_var}, "
            f"hue={hue_var}, column={column_var}"
        )


def _get_dimension_key(
    dimension: str, category_key: Optional[str], parameter_key: Optional[str]
) -> str:
    """Get the dataframe column name for a dimension."""
    if dimension == "category":
        if not category_key:
            raise ValueError("category_key required when using 'category' dimension")
        return category_key
    elif dimension == "parameter":
        if not parameter_key:
            raise ValueError("parameter_key required when using 'parameter' dimension")
        return parameter_key
    elif dimension == "layers":
        return "layers"  # Special handling
    elif dimension == "classifier_topk":
        return CLASSIFIER_TOP_PREFIX  # Special handling
    elif dimension == "experiment":
        return "experiment"  # Special handling for experiment dimension
    else:
        if not dimension:
            raise ValueError("Dimension name cannot be empty")
        return dimension


def _normalize_dimension(
    dimension: Optional[str],
) -> Tuple[Optional[str], Optional[int]]:
    """Normalize dimension string and extract classifier_top limit if present."""
    if dimension is None:
        return None, None

    dim = dimension.strip()
    if not dim:
        return None, None

    if dim.startswith(CLASSIFIER_TOP_PREFIX):
        suffix = dim[len(CLASSIFIER_TOP_PREFIX) :]
        if suffix == "k" or suffix == "":
            return "classifier_topk", None
        if suffix.isdigit():
            return "classifier_topk", int(suffix)

    return dim, None


def _coerce_measure_list(
    selection: Optional[Union[str, List[str]]],
    default: Optional[str],
) -> List[str]:
    """Normalize measure selection into an ordered, duplicate-free list."""

    def _dedupe_append(target: List[str], value: Optional[str]) -> None:
        if value is None:
            return
        text = str(value).strip()
        if not text:
            return
        if text not in target:
            target.append(text)

    result: List[str] = []

    if selection is None:
        if default:
            _dedupe_append(result, default)
        return result

    if isinstance(selection, str):
        value = selection.strip()
        if not value:
            return result
        if "," in value and not (value.startswith("[") and value.endswith("]")):
            for part in value.split(","):
                _dedupe_append(result, part)
            return result
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError, MemoryError):
                _dedupe_append(result, value)
            else:
                return _coerce_measure_list(parsed, default=None)
        else:
            _dedupe_append(result, value)
        return result

    if isinstance(selection, (list, tuple, set)):
        for item in selection:
            _dedupe_append(result, item)
        return result

    _dedupe_append(result, selection)
    return result


def _prettify_measure_suffix(suffix: str) -> str:
    """Convert raw metric suffix into human-friendly text."""

    cleaned = suffix.strip(" _")
    if not cleaned:
        return ""

    compact = re.sub(r"[\s_]+", "", cleaned.lower())
    match = re.fullmatch(r"top(\d+)", compact)
    if match:
        return f"Top {match.group(1)}"

    normalized = cleaned.replace("_", " ").strip()
    return normalized.title()


def _format_measure_label(column: str, base_label: str) -> str:
    """Format metric legend text based on column name and base label."""

    base_lower = base_label.lower()
    column_lower = column.lower()

    if column_lower == base_lower:
        return base_label

    suffix: str
    if column_lower.startswith(f"{base_lower}_"):
        suffix = column[len(base_lower) + 1 :]
    elif column_lower.endswith(f"_{base_lower}"):
        suffix = column[: -(len(base_lower) + 1)]
    else:
        suffix = column

    pretty_suffix = _prettify_measure_suffix(suffix)
    if not pretty_suffix:
        return base_label
    return f"{base_label} ({pretty_suffix})"


def _classifier_sort_key(column_name: str) -> int:
    suffix = column_name.replace(CLASSIFIER_TOP_PREFIX, "", 1)
    if suffix.isdigit():
        return int(suffix)
    return float("inf")


def _extract_classifier_top_values(df: pd.DataFrame) -> List[str]:
    columns = [
        col
        for col in df.columns
        if col.startswith(CLASSIFIER_TOP_PREFIX) and not col.endswith("_id")
    ]

    sorted_cols = sorted(columns, key=_classifier_sort_key)

    # Update cache with unit indices when available
    CLASSIFIER_UNIT_ID_MAP.clear()
    for col in sorted_cols:
        meta_col = f"{col}_id"
        unit_id: Optional[int] = None
        if meta_col in df.columns:
            series = df[meta_col].dropna()
            if not series.empty:
                try:
                    unit_id = int(series.iloc[0])
                except (ValueError, TypeError):
                    unit_id = None
        CLASSIFIER_UNIT_ID_MAP[col] = unit_id

    return sorted_cols


def _format_classifier_label(value: str) -> str:
    unit_id = CLASSIFIER_UNIT_ID_MAP.get(value)
    if unit_id is not None:
        return f"Unit {unit_id}"

    suffix = value.replace(CLASSIFIER_TOP_PREFIX, "", 1)
    if suffix.isdigit():
        return f"Top {int(suffix)}"
    return value.replace("_", " ").title()


def _is_special_dimension(dimension: Optional[str]) -> bool:
    return dimension in SPECIAL_DIMENSIONS


def _extract_dimension_values(
    df: pd.DataFrame,
    dimension: str,
    dimension_key: str,
    config: Dict,
    dimension_limit: Optional[int] = None,
    prefer_numeric_sort: bool = False,
) -> List[str]:
    """Extract and order unique values for a dimension."""
    if dimension == "layers":
        # Extract layer names from column headers
        layer_cols = [col for col in df.columns if col.endswith("_response_avg")]
        layers = [col.replace("_response_avg", "") for col in layer_cols]

        if not layers:
            logger.warning(
                f"No layer response columns found in data (looking for *_response_avg)"
            )
            return []

        ordered = order_layers(layers, config)
        logger.info(f"Found {len(ordered)} layers: {ordered}")
        return ordered
    elif dimension == "classifier_topk":
        classifier_cols = _extract_classifier_top_values(df)
        if dimension_limit is not None:
            original_count = len(classifier_cols)
            classifier_cols = classifier_cols[:dimension_limit]
            if dimension_limit < original_count:
                logger.info(
                    "Limiter applied to classifier dimension: using %d of %d columns",
                    len(classifier_cols),
                    original_count,
                )
            else:
                logger.info(
                    "Classifier dimension limit %d exceeds available columns (%d); using all available",
                    dimension_limit,
                    original_count,
                )
        if not classifier_cols:
            logger.warning(
                "No classifier_top* columns found in data (expected classifier_top1, classifier_top2, ... )"
            )
            return []
        logger.info(
            f"Found {len(classifier_cols)} classifier units: {classifier_cols}"
        )
        return classifier_cols
    else:
        # Check if column exists
        if dimension_key not in df.columns:
            logger.warning(
                f"Column '{dimension_key}' not found in data. Available columns: {list(df.columns)}"
            )
            return []

        series = df[dimension_key]
        raw_values = pd.unique(series)
        if pd.api.types.is_numeric_dtype(series):
            raw_values = sorted(raw_values)
        else:
            raw_values = sorted(raw_values, key=lambda x: str(x))

        values: List[str] = []
        seen = set()
        for raw in raw_values:
            standardized = standardize_category_value(raw)
            if standardized not in seen:
                values.append(standardized)
                seen.add(standardized)

        if not values:
            logger.warning(f"No unique values found for dimension '{dimension_key}'")
            return []

        # Apply config ordering if available
        ordering = get_ordering(dimension_key, config)
        if ordering:
            # Standardize ordering values too
            standardized_ordering = [standardize_category_value(v) for v in ordering]
            # Filter to only include values present in data
            ordered = [v for v in standardized_ordering if v in values]
            # Add any missing values
            for v in values:
                if v not in ordered:
                    ordered.append(v)
            logger.info(
                f"Found {len(ordered)} values for '{dimension_key}': {ordered} (config ordering applied)"
            )
            values = ordered
        elif prefer_numeric_sort:
            numeric_pairs: List[Tuple[float, str]] = []
            for v in values:
                try:
                    numeric_pairs.append((float(v), v))
                except (ValueError, TypeError):
                    numeric_pairs = []
                    break
            if numeric_pairs:
                numeric_pairs.sort(key=lambda pair: pair[0])
                values = [val for _, val in numeric_pairs]
                logger.info(
                    "Applied numeric ordering for '%s': %s",
                    dimension_key,
                    values,
                )

        if len(values) > MAX_DIMENSION_VALUES:
            raise ValueError(
                f"Dimension '{dimension_key}' has {len(values)} unique values, which exceeds the maximum "
                f"allowed ({MAX_DIMENSION_VALUES}). Choose a column with categorical data or reduce the "
                "number of unique values."
            )

        logger.info(f"Found {len(values)} values for '{dimension_key}': {values}")
        return values


def _get_colors_for_dimension(
    values: List[str], dimension_key: str, config: Dict
) -> Dict[str, str]:
    """Get color mapping for dimension values."""
    colors = {}
    for val in values:
        color = get_color(val, config)
        if color:
            colors[val] = color

    # Fill missing colors with viridis
    if len(colors) < len(values):
        viridis_colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
        for i, val in enumerate(values):
            if val not in colors:
                colors[val] = plt.matplotlib.colors.to_hex(viridis_colors[i])

    return colors


def _filter_data_for_column(
    df: pd.DataFrame, column_var: Optional[str], column_value: Optional[str]
) -> pd.DataFrame:
    """Filter dataframe for a specific column value."""
    if column_var and column_value is not None and column_var in df.columns:
        standardized_column_value = standardize_category_value(str(column_value))
        column_series = df[column_var]
        if isinstance(column_series.dtype, pd.CategoricalDtype):
            mask = column_series == standardized_column_value
        else:
            mask = (
                column_series.astype(str).map(standardize_category_value)
                == standardized_column_value
            )
        return df.loc[mask]
    return df


def _plot_accuracy_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    hue_var: str,
    hue_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    dt: float,
    show_ylabel: bool,
    show_legend: bool,
    accuracy_specs: List[MeasureSpec],
    confidence_specs: List[MeasureSpec],
    show_error_bands: bool = False,
    **kwargs,
) -> None:
    """Plot accuracy and confidence trends using pre-aggregated statistics."""

    fmt = {**FORMATTING, **kwargs}
    ax.patch.set_alpha(0)

    accuracy_linestyles = ["-", "--", "-."]
    confidence_linestyles = [":", (0, (3, 2)), (0, (1, 2))]
    greyscale_color = fmt.get("greyscale_color", "#5a5a5a")

    if not accuracy_specs and not confidence_specs:
        logger.warning("No accuracy or confidence measures requested; skipping panel")
        return

    data_plot = data
    has_hue_column = hue_key in data_plot.columns and not _is_special_dimension(
        hue_var
    )
    hue_sequences: List[Tuple[Optional[str], pd.DataFrame]] = []

    if has_hue_column:
        allowed_hues = {standardize_category_value(val) for val in hue_values}
        filtered = (
            data_plot[data_plot[hue_key].isin(allowed_hues)]
            if allowed_hues
            else data_plot
        )
        if filtered.empty:
            logger.warning(
                "No data remaining after applying hue filter for accuracy panel"
            )
            return
        for hue_value in hue_values:
            standardized = standardize_category_value(hue_value)
            subset = filtered.loc[filtered[hue_key] == standardized]
            if not subset.empty:
                hue_sequences.append((standardized, subset))
        if not hue_sequences:
            hue_sequences.append((None, filtered))
    else:
        hue_sequences.append((None, data_plot))

    def _plot_metric(
        subset: pd.DataFrame,
        column: str,
        color: str,
        linestyle: str,
    ) -> bool:
        if column not in subset.columns or subset.empty:
            return False
        values = subset[column].to_numpy()
        if np.all(np.isnan(values)):
            return False
        times = subset["time_ms"].to_numpy()
        ax.plot(
            times,
            values,
            color=color,
            linewidth=fmt["linewidth_main"],
            alpha=fmt["alpha_line"],
            linestyle=linestyle,
        )
        err_col = f"{column}_err"
        if show_error_bands and err_col in subset.columns:
            errors = subset[err_col].to_numpy()
            if not np.all(np.isnan(errors)):
                ax.fill_between(
                    times,
                    values - errors,
                    values + errors,
                    color=color,
                    alpha=fmt.get("alpha_error", 0.2),
                    linewidth=0,
                )
        return True

    plotted_accuracy = False
    for idx, spec in enumerate(accuracy_specs):
        linestyle = accuracy_linestyles[idx % len(accuracy_linestyles)]
        metric_plotted = False
        for hue_value, subset in hue_sequences:
            color = (
                colors.get(hue_value, greyscale_color)
                if hue_value
                else greyscale_color
            )
            metric_plotted |= _plot_metric(subset, spec.column, color, linestyle)
        if not metric_plotted:
            logger.warning("Accuracy column '%s' not found, skipping", spec.column)
        else:
            plotted_accuracy = True

    plotted_confidence = False
    for idx, spec in enumerate(confidence_specs):
        linestyle = confidence_linestyles[idx % len(confidence_linestyles)]
        metric_plotted = False
        for hue_value, subset in hue_sequences:
            color = (
                colors.get(hue_value, greyscale_color)
                if hue_value
                else greyscale_color
            )
            metric_plotted |= _plot_metric(subset, spec.column, color, linestyle)
        if not metric_plotted:
            logger.warning("Confidence column '%s' not found, skipping", spec.column)
        else:
            plotted_confidence = True

    if not plotted_accuracy and not plotted_confidence:
        logger.warning("Accuracy panel skipped because no metrics were plotted")
        return

    ax.set_ylim(-0.01, 1.01)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    if show_ylabel:
        ax.set_ylabel("Performance", fontsize=fmt["fontsize_axis"], fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.set_xlabel("Time (ms)", fontsize=fmt["fontsize_axis"])
    ax.tick_params(labelsize=fmt["fontsize_tick"])

    if show_legend:
        legend_handles: List[plt.Line2D] = []
        legend_labels: List[str] = []

        single_accuracy = len(accuracy_specs) == 1
        single_confidence = len(confidence_specs) == 1

        for idx, spec in enumerate(accuracy_specs):
            label = (
                "Accuracy"
                if single_accuracy
                else _format_measure_label(spec.column, spec.requested or "Accuracy")
            )
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=fmt["linewidth_main"],
                    linestyle=accuracy_linestyles[idx % len(accuracy_linestyles)],
                    alpha=fmt["alpha_line"],
                )
            )
            legend_labels.append(label)

        for idx, spec in enumerate(confidence_specs):
            label = (
                "Confidence"
                if single_confidence
                else _format_measure_label(spec.column, spec.requested or "Confidence")
            )
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=fmt["linewidth_main"],
                    linestyle=confidence_linestyles[idx % len(confidence_linestyles)],
                    alpha=fmt["alpha_line"],
                )
            )
            legend_labels.append(label)

        if legend_handles:
            ax.legend(
                handles=legend_handles,
                labels=legend_labels,
                loc="best",
                frameon=False,
                fontsize=fmt["fontsize_legend"] - 2,
            )

    try:
        label_indicator_df = calculate_label_indicator(
            data,
            hue_key,
            ax.get_ylim(),
            0.1,
        )
        ax.plot(
            label_indicator_df["times_index"] * dt,
            label_indicator_df["label_indicator"],
            color="dimgray",
            linewidth=fmt["linewidth_indicator"],
            drawstyle="steps-mid",
            alpha=fmt["alpha_indicator"],
        )
    except Exception as e:
        logger.debug(f"Could not calculate label indicator for subplot {hue_key}: {e}")

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=True, bottom=True)


def _add_layer_circle(x, y, layer_name, ax=None, config=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    pad = 0.5 if layer_name == "IT" else 0.4

    colors = _get_colors_for_dimension(
        values=[layer_name], dimension_key="layers", config=config
    )
    fmt = {**FORMATTING, **{k: v for k, v in kwargs.items() if k in FORMATTING}}

    ax.text(
        x,
        y,
        layer_name,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        bbox=dict(
            boxstyle=f"circle,pad={pad}",
            facecolor=colors.get(layer_name, "#808080ff"),
            edgecolor="#353535ff",
            linewidth=2,
            alpha=0.8,
        ),
        fontsize=fmt["fontsize_label"],
        fontweight="bold",
    )


def _plot_response_ridges(
    fig: plt.Figure,
    column_left: float,
    column_width: float,
    data: pd.DataFrame,
    subplot_var: str,
    subplot_key: str,
    subplot_values: List[str],
    hue_var: str,
    hue_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    dt: float,
    show_ylabel: bool,
    config: Dict,
    ridge_top: Optional[float] = None,
    **kwargs,
) -> List[plt.Axes]:
    """Plot ridge plots for responses.

    Args:
        fig: Matplotlib figure
        column_left: Left position of column
        column_width: Width of column
        data: DataFrame with response data
        subplot_var: Variable for subplots
        subplot_key: Column name for subplot variable
        subplot_values: Ordered list of subplot values
        hue_var: Variable for color coding
        hue_key: Column name for hue variable
        hue_values: Ordered list of hue values
        colors: Color mapping for hue values
        dt: Temporal resolution in ms
        show_ylabel: Whether to show y-axis label
        config: Configuration dict
        **kwargs: Override FORMATTING and LAYOUT defaults

    Returns:
        List of created axes
    """
    layout = {**LAYOUT, **{k: v for k, v in kwargs.items() if k in LAYOUT}}
    fmt = {**FORMATTING, **{k: v for k, v in kwargs.items() if k in FORMATTING}}

    # Check if focus_layer is specified in kwargs (for Panel B case)
    focus_layer = kwargs.get("focus_layer", None)

    n_subplots = len(subplot_values)
    if n_subplots == 0:
        logger.warning("No subplot values provided, cannot create ridge plots")
        return []

    logger.info(f"Creating {n_subplots} ridge subplots for dimension '{subplot_var}'")

    spacing = layout["ridge_height"] / n_subplots * (1 - layout["ridge_overlap"])
    plot_height = layout["ridge_height"] / n_subplots * 1.4

    axes = []

    # Calculate ridge top position from other elements
    if ridge_top is None:
        ridge_top = (
            layout["title_bot"]
            - layout["title_pad"]
            - layout["accuracy_height"]
            - layout["accuracy_pad"]
            - layout["legend_height"]
            - layout["legend_pad"]
        )

    global_ymin, global_ymax = fmt["max_global_ymax"], fmt["min_global_ymin"]
    global_xmin, global_xmax = 0, 0

    for i, subplot_value in enumerate(subplot_values):
        # Position subplot
        top_pos = ridge_top - i * spacing
        bottom_pos = top_pos - plot_height
        ax = fig.add_axes([column_left, bottom_pos, column_width, plot_height])

        ax.patch.set_alpha(0)
        axes.append(ax)

        # Reference line
        ax.axhline(0, color="gray", linestyle=":", alpha=0.7, linewidth=1)

        def _plot_series(subset: pd.DataFrame, column: str, color: str) -> bool:
            if subset.empty or column not in subset.columns:
                return False
            values = subset[column].to_numpy()
            if np.all(np.isnan(values)):
                return False
            times = subset["time_ms"].to_numpy()
            ax.plot(
                times,
                values,
                color=color,
                linewidth=fmt["linewidth_main"],
                alpha=fmt["alpha_line"],
            )
            err_col = f"{column}_err"
            if err_col in subset.columns:
                errors = subset[err_col].to_numpy()
                if not np.all(np.isnan(errors)):
                    ax.fill_between(
                        times,
                        values - errors,
                        values + errors,
                        color=color,
                        alpha=fmt.get("alpha_error", 0.2),
                        linewidth=0,
                    )
            return True

        def _iter_hue_subsets(current_data: pd.DataFrame):
            if hue_key in current_data.columns and not _is_special_dimension(hue_var):
                for hue_value in hue_values:
                    standardized = standardize_category_value(hue_value)
                    subset = current_data.loc[current_data[hue_key] == standardized]
                    if not subset.empty:
                        yield hue_value, subset, colors.get(
                            standardized, fmt["greyscale_color"]
                        )
            else:
                yield None, current_data, fmt["greyscale_color"]

        greyscale_color = fmt.get("greyscale_color", "#5a5a5a")

        # Filter data for this subplot
        if subplot_var == "layers":
            response_col = f"{subplot_value}_response_avg"
            if response_col not in data.columns:
                logger.warning(
                    f"Response column '{response_col}' not found for layer '{subplot_value}', skipping"
                )
                continue
            plot_data = data
        elif subplot_var == "classifier_topk":
            response_col = subplot_value
            if response_col not in data.columns:
                logger.warning(
                    f"Classifier activation column '{response_col}' not found, skipping"
                )
                continue
            plot_data = data
        else:
            if subplot_key not in data.columns:
                logger.warning(
                    f"Subplot key '{subplot_key}' not found in data, skipping"
                )
                continue
            standardized = standardize_category_value(subplot_value)
            plot_data = data.loc[data[subplot_key] == standardized]
            if plot_data.empty:
                logger.warning(
                    f"No data found for {subplot_key}={subplot_value}, skipping subplot"
                )
                continue
            response_col = None

        logger.debug(
            "Subplot %s/%s (%s): %s datapoints",
            i + 1,
            n_subplots,
            subplot_value,
            len(plot_data),
        )

        plotted_any = False

        if subplot_var == "layers":
            for hue_value, subset, color in _iter_hue_subsets(data):
                plotted_any |= _plot_series(subset, response_col, color)
            _add_layer_circle(
                x=0.95,
                y=0.25,
                layer_name=subplot_value.upper(),
                ax=ax,
                config=config,
                **fmt,
            )
        elif subplot_var == "classifier_topk":
            for hue_value, subset, color in _iter_hue_subsets(data):
                plotted_any |= _plot_series(subset, response_col, color)
            label_text = _format_classifier_label(subplot_value)
            ax.text(
                0.95,
                0.25,
                label_text,
                horizontalalignment="right",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=fmt["fontsize_label"],
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.8,
                ),
            )
        else:
            if focus_layer and hue_var == "category":
                focus_column = f"{focus_layer}_response_avg"
                if focus_column not in plot_data.columns:
                    logger.warning(
                        f"Focus layer response column '{focus_column}' not found for parameter '{subplot_value}'"
                    )
                else:
                    for hue_value, subset, color in _iter_hue_subsets(plot_data):
                        plotted_any |= _plot_series(subset, focus_column, color)
            elif hue_var == "layers":
                for hue_val in hue_values:
                    response_col = f"{hue_val}_response_avg"
                    color = colors.get(hue_val, greyscale_color)
                    plotted_any |= _plot_series(plot_data, response_col, color)
            elif hue_var == "classifier_topk":
                for hue_val in hue_values:
                    color = colors.get(hue_val, greyscale_color)
                    plotted_any |= _plot_series(plot_data, hue_val, color)
            else:
                logger.warning(
                    "No supported hue configuration for subplot '%s'", subplot_value
                )

            display_name = get_display_name(subplot_key, config)
            label_text = f"{display_name}={subplot_value}"
            ax.text(
                0.95,
                0.25,
                label_text,
                horizontalalignment="right",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=fmt["fontsize_label"],
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.8,
                ),
            )

        if not plotted_any:
            logger.warning(
                f"No valid response columns found for subplot '{subplot_value}'"
            )

        # Adjust limits to common scale across all subplots
        ymin, ymax = ax.get_ylim()
        global_ymin = min(min(global_ymin, ymin), fmt["min_global_ymin"])
        global_ymax = min(max(global_ymax, ymax), fmt["max_global_ymax"])
        xmin, xmax = ax.get_xlim()
        global_xmin = min(global_xmin, xmin)
        global_xmax = max(global_xmax, xmax)

        # Add label indicator
        try:
            label_indicator_df = calculate_label_indicator(
                plot_data,
                subplot_key if subplot_var != "layers" else hue_key,
                (0, global_ymax),
                0.1,
            )
            indicator_time = label_indicator_df["times_index"] * dt
            ax.plot(
                indicator_time,
                label_indicator_df["label_indicator"],
                color="dimgray",
                linewidth=fmt["linewidth_indicator"],
                drawstyle="steps-mid",
                alpha=fmt["alpha_indicator"],
            )
        except Exception as e:
            logger.debug(
                f"Could not calculate label indicator for subplot {subplot_value}: {e}"
            )

        # Y-label on leftmost panel, middle subplot
        if show_ylabel and i == len(subplot_values) // 2:
            ax.set_ylabel(
                "Avg Layer Response",
                fontsize=fmt["fontsize_axis"],
                fontweight="bold",
                labelpad=6,
            )
        else:
            ax.set_ylabel("")

        # X-axis only on bottom subplot
        if i < len(subplot_values) - 1:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Time (ms)", fontsize=fmt["fontsize_axis"])
            ax.tick_params(labelsize=fmt["fontsize_tick"])

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"], fontsize=fmt["fontsize_tick"])
        sns.despine(ax=ax, left=True, bottom=True)

    for ax in axes:
        ax.set_ylim(global_ymin, global_ymax)
        ax.set_xlim(global_xmin, global_xmax)

    logger.info(f"Created {len(axes)} ridge plot axes")
    return axes


def _add_horizontal_legend(
    fig: plt.Figure,
    hue_var: str,
    hue_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    config: Dict,
    legend_bot: float = None,
    dt: float = 2.0,
    **kwargs,
) -> None:
    """Add horizontal legend for hue dimension.

    Args:
        fig: Matplotlib figure
        column_left: Left position of column
        column_width: Width of column
        hue_var: Variable for color coding
        hue_key: Column name for hue variable
        hue_values: Ordered list of hue values (already ordered by config if available)
        colors: Color mapping for hue values
        config: Configuration dict with naming mappings
        dt: Temporal resolution in ms per timestep
        **kwargs: Override FORMATTING and LAYOUT defaults
    """
    layout = {**LAYOUT, **{k: v for k, v in kwargs.items() if k in LAYOUT}}
    fmt = {**FORMATTING, **{k: v for k, v in kwargs.items() if k in FORMATTING}}

    left = layout["left_margin"] + layout["legend_margin"]
    width = layout["column_width"] - 2 * layout["legend_margin"]

    # Calculate legend position from other elements
    if legend_bot is None:
        legend_bot = (
            layout["title_bot"]
            - layout["title_pad"]
            - layout["accuracy_height"]
            - layout["accuracy_pad"]
            - layout["legend_height"]
        )

    legend_ax = fig.add_axes([left, legend_bot, width, layout["legend_height"]])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    legend_ax.patch.set_alpha(0)

    # Create legend elements using the ordered hue_values
    legend_elements = []
    legend_labels = []

    for val in hue_values:
        if val in colors:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=colors[val],
                    linewidth=fmt["linewidth_main"],
                    alpha=fmt["alpha_line"],
                )
            )

            if hue_var == "layers":
                label = val.upper()
            elif hue_var == "classifier_topk":
                label = _format_classifier_label(val)
            else:
                label = _format_legend_label(hue_key, val, config, dt)
            legend_labels.append(label)

    if legend_elements:
        # Get symbol for title
        symbol = get_display_name(hue_key, config)

        n_cols = min(len(legend_elements), 7)
        logger.debug(
            f"Adding legend with {len(legend_elements)} elements in {n_cols} columns"
        )
        legend = legend_ax.legend(
            legend_elements,
            legend_labels,
            loc="lower center",
            ncol=n_cols,
            frameon=False,
            fontsize=fmt["fontsize_legend"],
            handlelength=2,
            handletextpad=0.5,
            columnspacing=1.0,
            title=symbol if hue_var != "layers" else None,
            title_fontsize=fmt["fontsize_legend"],
        )
    else:
        logger.warning("No legend elements to display")


def plot_temporal_ridge_responses(
    data: Union[Path, List[Path], pd.DataFrame],
    output: Path,
    subplot_var: str,
    hue_var: str,
    column_var: Optional[str] = None,
    category_key: Optional[str] = None,
    parameter_key: Optional[str] = None,
    experiment: Optional[str] = None,
    confidence_measure: Optional[Union[str, List[str]]] = "first_label_confidence",
    accuracy_measure: Optional[Union[str, List[str]]] = "accuracy",
    dt: float = 2.0,
    config: Optional[Dict] = None,
    errorbar_type: str = "std",
    min_group_count: int = 1,
    **kwargs,
) -> None:
    """Plot temporal ridge responses with flexible dimension mapping.

    Args:
        data: Path to test_data.csv, list of paths to multiple test_data.csv files, or DataFrame
        output: Path to save figure
        subplot_var: Variable for vertical subplots (ridge plots). Accepts
            'layers', 'category', 'parameter', 'classifier_topk', 'classifier_top{N}',
            or any dataframe column name (e.g., 'first_label_index') with a manageable
            number of unique values (<= MAX_DIMENSION_VALUES).
        hue_var: Variable for color coding. Accepts the same options as
            subplot_var and should remain categorical.
        column_var: Variable for columns (optional). Accepts the same options as
            subplot_var and should remain categorical.
        category_key: Column name for category (e.g., 'rctype')
        parameter_key: Column name for parameter (e.g., 'duration', 'contrast')
        confidence_measure: Column name or list of names for confidence metrics
        accuracy_measure: Column name or list of names for accuracy metrics
        dt: Temporal resolution in ms per timestep
        config: Configuration dict with palette, naming, ordering
        errorbar_type: Statistical summary for shaded errors (none, std, sem, ci95)
        min_group_count: Drop aggregated groups with fewer than this many rows
        **kwargs: Override LAYOUT and FORMATTING defaults
    """
    logger.info("=" * 60)
    logger.info("Starting temporal ridge response plotting")
    logger.info("=" * 60)

    # Normalize dimension configuration
    raw_subplot_var = subplot_var
    raw_hue_var = hue_var
    raw_column_var = column_var

    subplot_var, subplot_limit = _normalize_dimension(subplot_var)
    hue_var, hue_limit = _normalize_dimension(hue_var)
    column_var, column_limit = _normalize_dimension(column_var)

    logger.info(
        "Dimension mapping: subplot=%s (base=%s, limit=%s), hue=%s (base=%s, limit=%s), column=%s (base=%s, limit=%s)",
        raw_subplot_var,
        subplot_var,
        subplot_limit,
        raw_hue_var,
        hue_var,
        hue_limit,
        raw_column_var,
        column_var,
        column_limit,
    )

    for dim_name, limit in (
        ("subplot", subplot_limit),
        ("hue", hue_limit),
        ("column", column_limit),
    ):
        if limit is not None and limit <= 0:
            raise ValueError(
                f"{dim_name} classifier limit must be positive; got {limit}"
            )

    if subplot_var is None or hue_var is None:
        raise ValueError("subplot and hue dimensions cannot be empty")

    _validate_dimensions(
        subplot_var=subplot_var, hue_var=hue_var, column_var=column_var
    )

    if config is None:
        config = {"palette": {}, "naming": {}, "ordering": {}}
        logger.info("No config provided, using empty defaults")

    accuracy_requests = _coerce_measure_list(accuracy_measure, default="accuracy")
    confidence_requests = _coerce_measure_list(
        confidence_measure, default="first_label_confidence"
    )

    logger.info(
        "Selected accuracy measures: %s",
        accuracy_requests if accuracy_requests else "[none]",
    )
    logger.info(
        "Selected confidence measures: %s",
        confidence_requests if confidence_requests else "[none]",
    )

    focus_layer = kwargs.get("focus_layer")

    # Get dimension keys
    logger.info("Determining dimension keys...")
    subplot_key = _get_dimension_key(
        dimension=subplot_var, category_key=category_key, parameter_key=parameter_key
    )
    hue_key = _get_dimension_key(
        dimension=hue_var, category_key=category_key, parameter_key=parameter_key
    )
    column_key = (
        _get_dimension_key(
            dimension=column_var,
            category_key=category_key,
            parameter_key=parameter_key,
        )
        if column_var
        else None
    )
    logger.info(
        f"Keys: subplot_key='{subplot_key}', hue_key='{hue_key}', column_key='{column_key}'"
    )

    df_input: Optional[pd.DataFrame]
    data_paths: Optional[List[Path]]
    df_input = None
    data_paths = None

    if isinstance(data, list):
        data_paths = [Path(p) for p in data]
    elif isinstance(data, Path):
        data_paths = [data]
    elif isinstance(data, pd.DataFrame):
        df_input = data.copy()
    else:
        raise TypeError("data must be a Path, list of Paths, or pandas DataFrame")

    if df_input is not None:
        available_columns = df_input.columns.tolist()
    elif data_paths:
        available_columns = _read_available_columns(data_paths[0])
    else:
        raise ValueError("No data sources provided")

    plan = _build_column_plan(
        available_columns=available_columns,
        subplot_var=subplot_var,
        hue_var=hue_var,
        column_var=column_var,
        subplot_key=subplot_key,
        hue_key=hue_key,
        column_key=column_key,
        accuracy_requests=accuracy_requests,
        confidence_requests=confidence_requests,
        focus_layer=focus_layer,
    )

    if df_input is not None:
        missing = [col for col in plan.needed_columns if col not in df_input.columns]
        if missing:
            raise ValueError(
                "Provided DataFrame is missing required columns: " + ", ".join(missing)
            )
        df = df_input[plan.needed_columns].copy()
    else:
        frames: List[pd.DataFrame] = []
        assert data_paths is not None
        logger.info("Loading selected columns from %d file(s)", len(data_paths))
        for idx, path in enumerate(data_paths, start=1):
            logger.info("  Loading file %s/%s: %s", idx, len(data_paths), path)
            df_temp = pd.read_csv(path, usecols=plan.needed_columns)
            frames.append(df_temp)
        df = pd.concat(frames, ignore_index=True)

    logger.info("Loaded %d rows across %d selected columns", len(df), len(df.columns))

    # Prepare dataframe for aggregation
    logger.info("Preparing dataframe for aggregation")
    df["times_index"] = df["times_index"].astype(int)
    if "label_index" in df.columns:
        df["label_valid"] = (df["label_index"].astype(float) >= 0).astype(int)

    for column in plan.dimension_columns:
        if column in df.columns:
            df[column] = standardize_series(df[column])

    for column in plan.value_columns():
        if column in df.columns:
            df[column] = df[column].astype(np.float32, copy=False)

    group_columns: List[str] = [
        col for col in plan.dimension_columns if col in df.columns
    ]
    group_columns.append("times_index")

    extra_aggs: Dict[str, Any] = {}
    if "label_valid" in df.columns:
        extra_aggs["label_valid"] = "max"
    for meta_col in plan.classifier_meta_columns:
        if meta_col in df.columns:
            extra_aggs[meta_col] = "first"
    if not extra_aggs:
        extra_aggs = None

    df = aggregate_plot_data(
        df=df,
        group_keys=group_columns,
        mean_columns=plan.value_columns(),
        error_type=errorbar_type,
        min_count=min_group_count,
        extra_aggs=extra_aggs,
    )
    df["time_ms"] = df["times_index"].astype(float) * dt
    logger.info("Aggregated dataframe has %d rows", len(df))

    # Extract dimension values
    logger.info("Extracting dimension values...")
    subplot_numeric = subplot_var is not None and not _is_special_dimension(
        subplot_var
    )
    hue_numeric = hue_var is not None and not _is_special_dimension(hue_var)

    subplot_values = _extract_dimension_values(
        df=df,
        dimension=subplot_var,
        dimension_key=subplot_key,
        config=config,
        dimension_limit=subplot_limit,
        prefer_numeric_sort=subplot_numeric,
    )
    hue_values = _extract_dimension_values(
        df=df,
        dimension=hue_var,
        dimension_key=hue_key,
        config=config,
        dimension_limit=hue_limit,
        prefer_numeric_sort=hue_numeric,
    )
    column_values = (
        _extract_dimension_values(
            df=df,
            dimension=column_var,
            dimension_key=column_key,
            config=config,
            dimension_limit=column_limit,
            prefer_numeric_sort=column_var is not None
            and not _is_special_dimension(column_var),
        )
        if column_var
        else [None]
    )

    # Check if we have data to plot
    if not subplot_values or not hue_values:
        logger.error("Insufficient data for plotting: missing subplot or hue values")
        logger.error(f"Subplot values: {subplot_values}")
        logger.error(f"Hue values: {hue_values}")
        raise ValueError("Cannot create plot with empty dimension values")

    # Get colors for hue dimension
    logger.info("Assigning colors to hue dimension...")
    colors = _get_colors_for_dimension(
        values=hue_values, dimension_key=hue_key, config=config
    )
    logger.debug(f"Color mapping: {colors}")

    accuracy_error_bands = kwargs.pop("accuracy_error_bands", False)

    # Merge layout defaults with overrides
    layout = {**LAYOUT, **{k: v for k, v in kwargs.items() if k in LAYOUT}}
    fmt = {**FORMATTING, **{k: v for k, v in kwargs.items() if k in FORMATTING}}

    # Calculate figure size - use the actual inches for figure size
    n_columns = len(column_values)
    total_width = (
        n_columns * layout["column_width"] + (n_columns - 1) * layout["column_spacing"]
    )

    logger.info(
        f"Creating figure: {total_width:.2f}\" x {layout['figure_height']}\" with {n_columns} column(s)"
    )

    fig = plt.figure(
        figsize=(
            total_width,
            layout["figure_height"],
        )
    )
    sns.set_context("talk")

    # Plot each column
    all_ridge_axes = []
    for col_idx, column_value in enumerate(column_values):
        logger.info(
            "#" * 10
            + f" Processing column {col_idx+1}/{n_columns}"
            + (f" ({column_key}={column_value}) " if column_value else " ")
            + "#" * 10
        )

        # Calculate column position - now in figure units (0-1)
        relative_column_width = layout["column_width"] / total_width
        relative_column_spacing = layout["column_spacing"] / total_width
        column_left = col_idx * (relative_column_width + relative_column_spacing)

        # Filter data for this column
        column_data = _filter_data_for_column(
            df=df, column_var=column_key, column_value=column_value
        )
        logger.info(f"Column data: {len(column_data)} rows")

        # Add title
        if column_var:
            title_ax = fig.add_axes(
                [
                    column_left,
                    layout["title_bot"],
                    relative_column_width,
                    layout["title_pad"],
                ]
            )
            title_name = (
                get_display_name(key=f"{experiment}_experiment", config=config)
                if experiment
                else ""
            )
            title_symbol = get_display_name(key=column_key, config=config)
            if title_name:
                title_text = f"{title_name} ({title_symbol} = {column_value})"
            else:
                title_text = f"{title_symbol} = {column_value}"
            title_ax.text(
                0.5,
                0.5,
                title_text,
                ha="center",
                va="center",
                fontsize=fmt["fontsize_title"],
                fontweight="bold",
            )
            title_ax.set_xlim(0, 1)
            title_ax.set_ylim(0, 1)
            title_ax.axis("off")
            logger.debug(f"Added title: {title_text}")

        # Calculate accuracy panel position
        accuracy_bot = (
            layout["title_bot"] - layout["title_pad"] - layout["accuracy_height"]
        )

        # Plot accuracy panel
        logger.info("Creating accuracy panel...")
        acc_ax = fig.add_axes(
            [
                column_left,
                accuracy_bot,
                relative_column_width,
                layout["accuracy_height"],
            ]
        )
        _plot_accuracy_panel(
            ax=acc_ax,
            data=column_data,
            hue_var=hue_var,
            hue_key=hue_key,
            hue_values=hue_values,
            colors=colors,
            dt=dt,
            show_ylabel=(col_idx == 0),
            show_legend=(col_idx == 0),
            accuracy_specs=plan.accuracy_specs,
            confidence_specs=plan.confidence_specs,
            show_error_bands=accuracy_error_bands,
            **kwargs,
        )

        # Plot response ridges
        logger.info("Creating ridge plots...")
        ridge_axes = _plot_response_ridges(
            fig=fig,
            column_left=column_left,
            column_width=relative_column_width,
            data=column_data,
            subplot_var=subplot_var,
            subplot_key=subplot_key,
            subplot_values=subplot_values,
            hue_var=hue_var,
            hue_key=hue_key,
            hue_values=hue_values,
            colors=colors,
            dt=dt,
            show_ylabel=(col_idx == 0),
            config=config,
            **kwargs,
        )
        all_ridge_axes.extend(ridge_axes)

        # Add legend
        logger.info("Adding legend...")
        _add_horizontal_legend(
            fig=fig,
            column_left=column_left,
            column_width=relative_column_width,
            hue_var=hue_var,
            hue_key=hue_key,
            hue_values=hue_values,
            colors=colors,
            config=config,
            dt=dt,
            **kwargs,
        )

        if all_ridge_axes:
            # Align xy-limits across all ridge axes
            global_ymin, global_ymax = fmt["max_global_ymax"], fmt["min_global_ymin"]
            global_xmin, global_xmax = 0, 0

            for ax in all_ridge_axes:
                ymin, ymax = ax.get_ylim()
                global_ymin = min(global_ymin, ymin)
                global_ymax = max(global_ymax, ymax)
                xmin, xmax = ax.get_xlim()
                global_xmin = min(global_xmin, xmin)
                global_xmax = max(global_xmax, xmax)

            global_ymax = min(global_ymax, fmt["max_global_ymax"])
            global_ymin = min(global_ymin, fmt["min_global_ymin"])

            for ax in all_ridge_axes:
                ax.set_ylim(global_ymin, global_ymax)
                ax.set_xlim(global_xmin, global_xmax)

        # Align y-labels
        fig.align_ylabels(all_ridge_axes)
        logger.info(f"Aligned y-labels for {len(all_ridge_axes)} axes")

    # Save
    logger.info(f"Saving figure to: {output}")
    save_plot(output)
    logger.info("=" * 60)
    logger.info("Plotting complete")
    logger.info("=" * 60)

    return fig


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot temporal ridge responses with flexible dimensions"
    )
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to test_data.csv file(s). Multiple paths will be concatenated.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output figure path"
    )
    parser.add_argument(
        "--subplot",
        type=str,
        required=True,
        help=(
            "Variable for vertical subplots. Use 'layers', 'category', 'parameter', "
            "'classifier_topk', 'classifier_top{N}', or any dataframe column name (e.g., 'first_label_index') "
            "with a manageable number of unique values."
        ),
    )
    parser.add_argument(
        "--hue",
        type=str,
        required=True,
        help=(
            "Variable for color hues. Use 'layers', 'category', 'parameter', "
            "'classifier_topk', 'classifier_top{N}', or any dataframe column name (e.g., 'first_label_index') "
            "with a manageable number of unique values."
        ),
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help=(
            "Variable for columns (optional). Use 'layers', 'category', 'parameter', "
            "'classifier_topk', 'classifier_top{N}', or any dataframe column name with a manageable number of unique values."
        ),
    )
    parser.add_argument("--category-key", type=str, help="Category column name")
    parser.add_argument("--parameter-key", type=str, help="Parameter column name")
    parser.add_argument("--experiment", type=str, help="Experiment name")
    parser.add_argument(
        "--confidence-measure",
        "--confidence_measure",
        type=str,
        default="first_label_confidence",
        help=(
            "Confidence measure column name or list (comma-separated or Python literal list)"
        ),
    )
    parser.add_argument(
        "--accuracy-measure",
        "--accuracy_measure",
        type=str,
        default="accuracy",
        help=(
            "Accuracy measure column name or list (comma-separated or Python literal list)"
        ),
    )
    parser.add_argument("--dt", type=float, default=2.0, help="Time resolution (ms)")
    parser.add_argument("--palette", type=str, help="JSON color palette")
    parser.add_argument("--naming", type=str, help="JSON naming dict")
    parser.add_argument("--ordering", type=str, help="JSON ordering dict")
    parser.add_argument(
        "--errorbar-type",
        "--errorbar_type",
        type=str,
        default="std",
        choices=["none", "std", "sem", "ci95"],
        help="Error metric used for shaded bands (default: std)",
    )
    parser.add_argument(
        "--min-group-count",
        "--min_group_count",
        type=int,
        default=1,
        help="Minimum rows per aggregated trace",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # Use parse_known_args to allow additional arguments
    args, unknown = parser.parse_known_args()

    # Configure logging - but don't add handlers if they already exist
    logging.basicConfig(
        level=getattr(logging, "DEBUG"),
        force=False,  # Don't override existing configuration
    )

    if unknown:
        logger.info(f"Ignoring unknown arguments: {unknown}")

    # Load config
    config = load_config_from_args(
        palette_str=args.palette,
        naming_str=args.naming,
        ordering_str=args.ordering,
    )

    # Handle single vs multiple data files
    data_input = args.data if len(args.data) > 1 else args.data[0]

    # Plot
    plot_temporal_ridge_responses(
        data=data_input,
        output=args.output,
        subplot_var=args.subplot,
        hue_var=args.hue,
        column_var=args.column,
        category_key=args.category_key,
        parameter_key=args.parameter_key,
        experiment=args.experiment,
        confidence_measure=args.confidence_measure,
        accuracy_measure=args.accuracy_measure,
        dt=args.dt,
        config=config,
        errorbar_type=args.errorbar_type,
        min_group_count=args.min_group_count,
    )


if __name__ == "__main__":

    main()
