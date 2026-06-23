"""Compute key weight distribution metrics for figure captions.

Reads a weight summary CSV produced by ``plot_weight_distributions.py``
(``--summary-output`` flag) and reports two views:

1. **Per layer, averaged over categories** — for each connection type
   (feedforward, recurrence, skip, feedback).
2. **Per category value, averaged over layers** — same connection types.

Each metric is reported as ``mean ± pooled_std``, where
``pooled_std = sqrt(mean(std²))`` (root-mean-square of within-distribution
standard deviations, assuming equal sample sizes across conditions).

Layer classification
--------------------
* **feedforward** – ``connection_type == 'feedforward'``, no ``addskip_``/
  ``addfeedback_`` prefix, not ``classifier``.
* **skip** – ``connection_type == 'feedforward'``, layer starts with ``addskip_``.
* **feedback** – ``connection_type == 'feedback'`` *or* layer starts with
  ``addfeedback_``.
* **recurrence** – ``connection_type == 'recurrence'``.
* **classifier** – ``layer == 'classifier'``.

Usage
-----
    python weight_caption_metrics.py summary.csv
    python weight_caption_metrics.py summary.csv --output metrics.csv --txt metrics.txt
    python weight_caption_metrics.py summary.csv --category-col category \\
        --layer-col layer --connection-col connection_type
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHT_TYPE_ORDER = ("feedforward", "recurrence", "skip", "feedback", "classifier")

WEIGHT_TYPE_LABELS: Dict[str, str] = {
    "feedforward": "Feedforward",
    "recurrence": "Recurrent",
    "skip": "Skip",
    "feedback": "Feedback",
    "classifier": "Classifier",
}

STAT_COLS = ("mean", "median", "min", "max", "std")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute weight metrics for figure captions from a summary CSV."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Summary CSV produced by plot_weight_distributions.py --summary-output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV file to write the aggregated metrics table.",
    )
    parser.add_argument(
        "--txt",
        type=Path,
        default=None,
        help="Optional text file to write the formatted caption-ready report.",
    )
    parser.add_argument(
        "--category-col",
        default="category",
        help="Column name for the category dimension (default: 'category').",
    )
    parser.add_argument(
        "--layer-col",
        default="layer",
        help="Column name for the layer dimension (default: 'layer').",
    )
    parser.add_argument(
        "--connection-col",
        default="connection_type",
        help="Column name for connection type (default: 'connection_type').",
    )
    parser.add_argument(
        "--exclude-layers",
        nargs="*",
        default=["classifier"],
        help="Layer names to exclude from feedforward/recurrence aggregation "
        "(default: classifier). Pass '' to exclude nothing.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of significant figures for reported values (default: 4).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def classify_weight_type(row: pd.Series, connection_col: str, layer_col: str) -> str:
    """Assign one of feedforward / recurrence / skip / feedback / classifier."""
    conn = str(row[connection_col]).lower()
    layer = str(row[layer_col]).lower()

    if layer == "classifier":
        return "classifier"
    if layer.startswith("addfeedback_") or conn == "feedback":
        return "feedback"
    if layer.startswith("addskip_"):
        return "skip"
    if conn == "recurrence":
        return "recurrence"
    return "feedforward"


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------


def _pooled_std(stds: pd.Series) -> float:
    """Root-mean-square of individual standard deviations (pooled std)."""
    vals = stds.dropna().values.astype(float)
    if len(vals) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(vals**2)))


def _aggregate(group: pd.DataFrame) -> pd.Series:
    """Aggregate a group of summary rows into a single metrics row."""
    return pd.Series(
        {
            "mean": group["mean"].mean(),
            "std_pooled": _pooled_std(group["std"]),
            "median": group["median"].mean(),
            "min": group["min"].min(),
            "max": group["max"].max(),
            "n_rows": len(group),
        }
    )


def compute_per_layer_metrics(
    df: pd.DataFrame,
    category_col: str,
    layer_col: str,
    exclude_layers: List[str],
) -> pd.DataFrame:
    """Aggregate over *categories* → one row per (weight_type, layer)."""
    frames: List[pd.DataFrame] = []

    for weight_type, group in df.groupby("weight_type"):
        if weight_type in ("skip", "feedback"):
            # Skip/feedback already have layer info (addskip_V1, etc.)
            for layer, sub in group.groupby(layer_col):
                agg = _aggregate(sub)
                agg["weight_type"] = weight_type
                agg[layer_col] = layer
                frames.append(agg.to_frame().T)
        else:
            layer_groups = group[~group[layer_col].isin(exclude_layers)].groupby(
                layer_col
            )
            for layer, sub in layer_groups:
                agg = _aggregate(sub)
                agg["weight_type"] = weight_type
                agg[layer_col] = layer
                frames.append(agg.to_frame().T)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["weight_type"] = pd.Categorical(
        result["weight_type"], categories=WEIGHT_TYPE_ORDER, ordered=True
    )
    return result.sort_values(["weight_type", layer_col]).reset_index(drop=True)


def compute_per_category_metrics(
    df: pd.DataFrame,
    category_col: str,
    layer_col: str,
    exclude_layers: List[str],
) -> pd.DataFrame:
    """Aggregate over *layers* → one row per (weight_type, category)."""
    frames: List[pd.DataFrame] = []

    for weight_type, group in df.groupby("weight_type"):
        if weight_type == "classifier":
            # Classifier has no meaningful layer aggregation; keep per category
            for cat, sub in group.groupby(category_col):
                agg = _aggregate(sub)
                agg["weight_type"] = weight_type
                agg[category_col] = cat
                frames.append(agg.to_frame().T)
            continue

        filtered = group
        if weight_type not in ("skip", "feedback"):
            filtered = group[~group[layer_col].isin(exclude_layers)]

        for cat, sub in filtered.groupby(category_col):
            agg = _aggregate(sub)
            agg["weight_type"] = weight_type
            agg[category_col] = cat
            frames.append(agg.to_frame().T)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["weight_type"] = pd.Categorical(
        result["weight_type"], categories=WEIGHT_TYPE_ORDER, ordered=True
    )
    return result.sort_values(["weight_type", category_col]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _sigfig(value: float, n: int) -> str:
    """Format a float to *n* significant figures."""
    if not np.isfinite(value):
        return str(value)
    if value == 0.0:
        return "0"
    magnitude = int(np.floor(np.log10(abs(value))))
    decimals = max(0, n - 1 - magnitude)
    return f"{value:.{decimals}f}"


def _fmt(mean: float, std: float, precision: int) -> str:
    return f"{_sigfig(mean, precision)} ± {_sigfig(std, precision)}"


def format_report(
    per_layer: pd.DataFrame,
    per_category: pd.DataFrame,
    layer_col: str,
    category_col: str,
    precision: int,
    source: Optional[Path] = None,
) -> str:
    buf = io.StringIO()

    header = "Weight Distribution Metrics for Figure Caption"
    if source:
        header += f"\nSource: {source}"
    print(header, file=buf)
    print("=" * len(header.splitlines()[0]), file=buf)

    # ------------------------------------------------------------------
    # Section 1: Per layer, averaged over categories
    # ------------------------------------------------------------------
    print(
        "\n[1] Per layer  (mean ± pooled_std, averaged over all category values)\n",
        file=buf,
    )

    for weight_type in WEIGHT_TYPE_ORDER:
        subset = per_layer[per_layer["weight_type"] == weight_type]
        if subset.empty:
            continue

        label = WEIGHT_TYPE_LABELS[weight_type]
        print(f"  {label}:", file=buf)

        max_layer_len = max(len(str(r)) for r in subset[layer_col]) + 2
        for _, row in subset.iterrows():
            layer_str = str(row[layer_col]).ljust(max_layer_len)
            val_str = _fmt(float(row["mean"]), float(row["std_pooled"]), precision)
            print(f"    {layer_str}{val_str}", file=buf)

        print("", file=buf)

    # ------------------------------------------------------------------
    # Section 2: Per category, averaged over layers
    # ------------------------------------------------------------------
    print(
        "\n[2] Per category value  (mean ± pooled_std, averaged over all layers)\n",
        file=buf,
    )

    for weight_type in WEIGHT_TYPE_ORDER:
        subset = per_category[per_category["weight_type"] == weight_type]
        if subset.empty:
            continue

        label = WEIGHT_TYPE_LABELS[weight_type]
        print(f"  {label}:", file=buf)

        cat_values = subset[category_col].tolist()
        max_cat_len = max(len(str(c)) for c in cat_values) + 2
        for _, row in subset.iterrows():
            cat_str = str(row[category_col]).ljust(max_cat_len)
            val_str = _fmt(float(row["mean"]), float(row["std_pooled"]), precision)
            print(f"    {category_col}={cat_str}{val_str}", file=buf)

        print("", file=buf)

    return buf.getvalue()


def build_csv_output(
    per_layer: pd.DataFrame,
    per_category: pd.DataFrame,
    layer_col: str,
    category_col: str,
) -> pd.DataFrame:
    """Combine both views into one tidy CSV with a 'view' and 'group' column."""
    rows: List[pd.Series] = []

    for _, row in per_layer.iterrows():
        rows.append(
            pd.Series(
                {
                    "view": "per_layer",
                    "weight_type": row["weight_type"],
                    "group_col": layer_col,
                    "group_value": row[layer_col],
                    "mean": row["mean"],
                    "std_pooled": row["std_pooled"],
                    "median": row["median"],
                    "min": row["min"],
                    "max": row["max"],
                    "n_rows": row["n_rows"],
                }
            )
        )

    for _, row in per_category.iterrows():
        rows.append(
            pd.Series(
                {
                    "view": "per_category",
                    "weight_type": row["weight_type"],
                    "group_col": category_col,
                    "group_value": row[category_col],
                    "mean": row["mean"],
                    "std_pooled": row["std_pooled"],
                    "median": row["median"],
                    "min": row["min"],
                    "max": row["max"],
                    "n_rows": row["n_rows"],
                }
            )
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    df = pd.read_csv(args.input)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    # Validate required columns
    required = {args.connection_col, args.layer_col, args.category_col}
    missing = required - set(df.columns)
    if missing:
        logger.error(
            "CSV is missing required columns: %s. Available: %s",
            sorted(missing),
            sorted(df.columns.tolist()),
        )
        sys.exit(1)

    for stat in STAT_COLS:
        if stat not in df.columns:
            logger.warning(
                "Stat column '%s' not found in CSV; it will be skipped.", stat
            )

    # Classify each row by weight type
    df["weight_type"] = df.apply(
        classify_weight_type,
        axis=1,
        connection_col=args.connection_col,
        layer_col=args.layer_col,
    )

    exclude_layers = [lay for lay in args.exclude_layers if lay]  # drop empty strings

    present_types = df["weight_type"].unique().tolist()
    logger.info("Weight types detected: %s", sorted(present_types))

    per_layer = compute_per_layer_metrics(
        df,
        category_col=args.category_col,
        layer_col=args.layer_col,
        exclude_layers=exclude_layers,
    )

    per_category = compute_per_category_metrics(
        df,
        category_col=args.category_col,
        layer_col=args.layer_col,
        exclude_layers=exclude_layers,
    )

    report = format_report(
        per_layer=per_layer,
        per_category=per_category,
        layer_col=args.layer_col,
        category_col=args.category_col,
        precision=args.precision,
        source=args.input,
    )

    print(report)

    if args.txt:
        args.txt.parent.mkdir(parents=True, exist_ok=True)
        args.txt.write_text(report)
        logger.info("Saved text report to %s", args.txt)

    if args.output:
        combined = build_csv_output(
            per_layer=per_layer,
            per_category=per_category,
            layer_col=args.layer_col,
            category_col=args.category_col,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.output, index=False)
        logger.info("Saved metrics CSV to %s", args.output)


if __name__ == "__main__":
    main()
