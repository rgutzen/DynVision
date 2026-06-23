#!/usr/bin/env python3
"""
Fetch model training benchmarks from Weights & Biases.

Fetches Time/Epoch (min) and GPU Memory (GB) from W&B runs for different
model parameter variations, computing deltas relative to the default
model configuration.

Usage:
    python dynvision/visualization/fetch_benchmarks.py [--output {latex,csv,json}]
    python dynvision/visualization/fetch_benchmarks.py --verbose --cache-runs /tmp/runs.csv

Requires:
    - wandb Python SDK installed (conda env: rva)
    - wandb login completed

Default model (from config_workflow.yaml model_args):
    tsteps=20, dt=2, tau=5, tff=0, trc=6, tsk=0, lossrt=4,
    activityloss=0.0002, pattern='1', rctype='full', rctarget='output',
    skip='true', feedback='false', idle=0

Filters: seed in {7000, 7001, 7002}, batch_size=192, state=finished
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WANDB_ENTITY = "rgutzen"
WANDB_PROJECT = "rhythmic_visual_attention"

VALID_SEEDS: Set[int] = {7000, 7001, 7002}
TARGET_BATCH_SIZE = 192

# Default model configuration (from config_workflow.yaml model_args)
# Values use native Python types matched to what W&B config stores.
DEFAULT_CONFIG: Dict[str, Any] = {
    "n_timesteps": 20,
    "dt": 2,
    "tau": 5,
    "t_feedforward": 0,
    "t_recurrence": 6,
    "t_skip": 0,
    "loss_reaction_time": 4,
    "activity_loss_weight": 0.0002,
    "data_presentation_pattern": 1,       # W&B stores as int
    "recurrence_type": "full",
    "recurrence_target": "output",
    "skip": True,
    "feedback": False,
    "idle_timesteps": 0,
    "shuffle_presentation_pattern": True,
    "feedforward_only": False,
    "t_feedback": 30,
    "precision": "32",
    "lightning_version": "1.9.5",
}

# GPU filtering: runs newer than this many days are on H200; older on A100 80GB.
# (Cluster migration happened ~March 2026.)
GPU_CUTOFF_DAYS = 120

# GPU model used for the main benchmark table rows (A100 80 GB).
# Unrolling / dataloader-expansion rows are expected to be H200-based
# (A100 retired before these sweeps could be re-run).
BENCHMARK_GPU = "A100-SXM4-80GB"

# Parameters that vary (one at a time).
# Each entry: (table_category_label, config_key, [valid_values])
# 'Feedback' is special — combines 'feedback' (bool) + 'feedback_mode' (str).
PARAMETER_VARIATIONS: List[Tuple[str, str, List[Any]]] = [
    ("Recurrence Type", "recurrence_type", ["self", "full", "depthpointwise", "pointdepthwise", "local", "localdepthwise", "none"]),
    ("Recurrence Target", "recurrence_target", ["input", "middle", "output"]),
    ("Skip", "skip", [True, False]),
    ("Feedback", "feedback", ["False", "Additive", "Multiplicative"]),
    ("Timesteps", "n_timesteps", [8, 14, 20, 26]),
    ("Presentation Pattern", "data_presentation_pattern", [1, 1011, 1001, 1000]),
    ("Loss Reaction Time", "loss_reaction_time", [0, 4, 8, 18, 28]),
    ("EnergyLoss Weight", "activity_loss_weight", [0, 0.0002, 0.02, 0.2, 1.0]),
    ("Idle Timesteps", "idle_timesteps", [0, 1, 5, 10, 20]),
]

# Which label is the default per category (for gray-row highlighting)
CATEGORY_DEFAULTS: Dict[str, str] = {
    "Recurrence Type": "full",
    "Recurrence Target": "output",
    "Skip": "True",
    "Feedback": "False",
    "Timesteps": "20",
    "Presentation Pattern": "1",
    "Loss Reaction Time": "4",
    "EnergyLoss Weight": "0.0002",
    "Idle Timesteps": "0",
}

# Rows flagged with a dagger: counterintuitive memory deltas caused by
# sweep-to-sweep measurement artifacts rather than architecture differences.
DAGGER_ROWS: Set[Tuple[str, str]] = {
    ("Recurrence Type", "self"),
    ("Recurrence Target", "middle"),
}

# Trainable parameter counts (from dynvision/visualization/count_params.py).
# Default: 8,534,794.  These are the pre-computed deltas for each variation.
PARAM_DELTAS: Dict[Tuple[str, str], int] = {
    # Recurrence Type
    ("Recurrence Type", "self"): -3398402,
    ("Recurrence Type", "full"): 0,
    ("Recurrence Type", "depthpointwise"): -3016254,
    ("Recurrence Type", "pointdepthwise"): -3016254,
    ("Recurrence Type", "local"): -3398338,
    ("Recurrence Type", "localdepthwise"): -3387384,
    ("Recurrence Type", "none"): -3398410,
    # Recurrence Target
    ("Recurrence Target", "input"): -1759607,
    ("Recurrence Target", "middle"): 0,
    ("Recurrence Target", "output"): 0,
    # Skip
    ("Skip", "True"): 0,
    ("Skip", "False"): -93345,
    # Feedback
    ("Feedback", "False"): 0,
    ("Feedback", "Additive"): +92768,
    ("Feedback", "Multiplicative"): +92768,
    # Everything else: 0 (architecture-neutral)
}
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(value: Any) -> str:
    """Normalize a config value for string comparison."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(float(value))
    return str(value).strip().lower()


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two config values, handling type coercion."""
    # Both numeric → compare as floats
    a_is_num = isinstance(a, (int, float, np.integer, np.floating))
    b_is_num = isinstance(b, (int, float, np.integer, np.floating))
    if a_is_num and b_is_num:
        return float(a) == float(b)
    # One numeric, one string → try numeric comparison
    if a_is_num and isinstance(b, str):
        try:
            return float(a) == float(b)
        except (ValueError, TypeError):
            pass
    if b_is_num and isinstance(a, str):
        try:
            return float(a) == float(b)
        except (ValueError, TypeError):
            pass
    # Both boolean
    if isinstance(a, bool) and isinstance(b, bool):
        return a == b
    # String comparison
    return _normalize(a) == _normalize(b)


def _matches_defaults(row: pd.Series, skip_key: str) -> bool:
    """Check that all config keys except *skip_key* match defaults."""
    for key, default_val in DEFAULT_CONFIG.items():
        if key == skip_key:
            continue
        if key not in row.index:
            continue
        if not _values_equal(row[key], default_val):
            return False
    return True


def _feedback_label(is_feedback: bool, mode: Any) -> str:
    """Map W&B feedback config to table row label."""
    if not is_feedback:
        return "False"
    m = str(mode).lower()
    if m in ("additive", "add"):
        return "Additive"
    if m in ("multiplicative", "mul"):
        return "Multiplicative"
    return "Additive"


def _to_bool(value: Any) -> bool:
    """Coerce a config value to boolean."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


def fetch_runs(
    entity: str,
    project: str,
    verbose: bool = False,
    max_runs: int = 0,
    gpu: str = "A100",
) -> pd.DataFrame:
    """
    Fetch all matching W&B runs and extract Time/Epoch + GPU Memory.

    Args:
        entity: W&B entity name.
        project: W&B project name.
        verbose: Enable progress logging.
        max_runs: Limit total runs processed (0 = no limit).
        gpu: GPU filter: 'A100' excludes H200 runs (pre-March 2026);
             'H200' includes all runs regardless of date.
    """
    import wandb

    api = wandb.Api()
    runs_iter = api.runs(f"{entity}/{project}")

    from datetime import datetime, timedelta, timezone

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=GPU_CUTOFF_DAYS)
    include_all = (gpu == "H200")

    records: List[Dict[str, Any]] = []
    total = 0
    matched = 0
    gpu_filtered = 0

    if verbose:
        logger.info("Fetching runs from W&B ...")

    for run in runs_iter:
        total += 1
        cfg = run.config

        # --- quick filters ---
        try:
            seed_val = int(cfg.get("seed", -1))
        except (TypeError, ValueError):
            seed_val = -1
        try:
            bs = int(cfg.get("batch_size", -1))
        except (TypeError, ValueError):
            bs = -1

        if seed_val not in VALID_SEEDS:
            continue
        if bs != TARGET_BATCH_SIZE:
            continue
        if run.state != "finished":
            continue

        # --- GPU filter: exclude H200 runs (newer than cutoff) unless --gpu H200 ---
        if not include_all:
            run_date_str = run.created_at
            if run_date_str is not None:
                try:
                    run_date = datetime.fromisoformat(run_date_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    run_date = None
                if run_date is not None and run_date >= cutoff_date:
                    gpu_filtered += 1
                    if verbose and gpu_filtered <= 5:
                        logger.info("  Skipping H200 run (date=%s): %s",
                                    run_date.strftime("%Y-%m-%d"), run.name[:60])
                    continue

        matched += 1
        if verbose and matched % 20 == 0:
            logger.info("  ... processed %d matching runs (searched %d total)", matched, total)

        # --- extract config ---
        fb_val = cfg.get("feedback")
        fb_bool = _to_bool(fb_val)
        fb_mode = cfg.get("feedback_mode", "additive")

        # activity_loss_weight may be stored as 'energy_loss_weight' in some runs
        alw = cfg.get("activity_loss_weight")
        if alw is None:
            alw = cfg.get("energy_loss_weight", DEFAULT_CONFIG["activity_loss_weight"])

        record: Dict[str, Any] = {
            "run_id": run.id,
            "name": run.name,
            "seed": seed_val,
            "batch_size": bs,
            "recurrence_type": _normalize(cfg.get("recurrence_type")),
            "recurrence_target": _normalize(cfg.get("recurrence_target")),
            "n_timesteps": int(cfg.get("n_timesteps", DEFAULT_CONFIG["n_timesteps"])),
            "dt": float(cfg.get("dt", DEFAULT_CONFIG["dt"])),
            "tau": float(cfg.get("tau", DEFAULT_CONFIG["tau"])),
            "t_feedforward": float(cfg.get("t_feedforward", cfg.get("tff", DEFAULT_CONFIG["t_feedforward"]))),
            "t_recurrence": float(cfg.get("t_recurrence", cfg.get("trc", DEFAULT_CONFIG["t_recurrence"]))),
            "t_skip": float(cfg.get("t_skip", cfg.get("tsk", DEFAULT_CONFIG["t_skip"]))),
            "t_feedback": float(cfg.get("t_feedback", cfg.get("tfb", DEFAULT_CONFIG.get("t_feedback", 30)))),
            "feedback": fb_bool,
            "feedback_mode": _normalize(fb_mode),
            "feedback_label": _feedback_label(fb_bool, fb_mode),
            "skip": _to_bool(cfg.get("skip", True)),
            "loss_reaction_time": float(cfg.get("loss_reaction_time", DEFAULT_CONFIG["loss_reaction_time"])),
            "activity_loss_weight": float(alw),
            "data_presentation_pattern": int(cfg.get("data_presentation_pattern", DEFAULT_CONFIG["data_presentation_pattern"])),
            "idle_timesteps": int(cfg.get("idle_timesteps", 0)),
            "shuffle_presentation_pattern": _to_bool(cfg.get("shuffle_presentation_pattern", True)),
            "feedforward_only": _to_bool(cfg.get("feedforward_only", False)),
            "precision": str(cfg.get("precision", DEFAULT_CONFIG["precision"])).strip().lower(),
            "lightning_version": str(cfg.get("lightning_version", DEFAULT_CONFIG["lightning_version"])),
        }

        # --- time per epoch (samples=300 balances speed vs accuracy) ---
        try:
            history = run.history(keys=["_runtime", "epoch"], samples=300)
            if not history.empty and "epoch" in history.columns:
                hist = history.dropna(subset=["_runtime", "epoch"])
                if len(hist) >= 2:
                    epochs = hist["epoch"].values
                    runtimes = hist["_runtime"].values
                    durations = []
                    last_e, last_t = epochs[0], runtimes[0]
                    for e, t in zip(epochs[1:], runtimes[1:]):
                        if e > last_e:
                            durations.append((t - last_t) / (e - last_e))
                            last_e, last_t = e, t
                    if durations:
                        record["time_per_epoch_min"] = np.mean(durations) / 60.0
                    else:
                        record["time_per_epoch_min"] = np.nan
                else:
                    record["time_per_epoch_min"] = np.nan
            else:
                record["time_per_epoch_min"] = np.nan
        except Exception as e:
            if verbose:
                logger.warning("Time/epoch failed for %s: %s", run.id, e)
            record["time_per_epoch_min"] = np.nan

        # --- GPU memory (use 90th percentile of allocated bytes; robust to
        #     warm-up and validation-phase drops) ---
        try:
            sys_hist = run.history(stream="system", samples=200)
            mem_key = "system.gpu.0.memoryAllocatedBytes"
            if not sys_hist.empty and mem_key in sys_hist.columns:
                mem_vals = sys_hist[mem_key].dropna()
                if len(mem_vals) > 0:
                    # 90th percentile captures steady-state training level,
                    # ignoring warm-up (low) and brief spikes (high).
                    record["gpu_memory_gb"] = float(np.percentile(mem_vals, 90)) / (1024**3)
                    record["gpu_memory_std_gb"] = float(np.percentile(mem_vals, 75) - np.percentile(mem_vals, 25)) / (1024**3) / 2.0
                else:
                    record["gpu_memory_gb"] = np.nan
                    record["gpu_memory_std_gb"] = np.nan
            else:
                record["gpu_memory_gb"] = np.nan
                record["gpu_memory_std_gb"] = np.nan
        except Exception as e:
            if verbose:
                logger.warning("GPU memory failed for %s: %s", run.id, e)
            record["gpu_memory_gb"] = np.nan
            record["gpu_memory_std_gb"] = np.nan

        records.append(record)

        if max_runs > 0 and len(records) >= max_runs:
            logger.info("Reached max_runs limit (%d), stopping.", max_runs)
            break

    df = pd.DataFrame(records)
    if verbose:
        logger.info("Searched %d runs, matched %d (A100), skipped %d (H200), kept %d",
                     total, matched, gpu_filtered, len(df))
    return df


def aggregate_benchmarks(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Group runs by parameter category and value, compute mean metrics."""
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for category, config_key, valid_values in PARAMETER_VARIATIONS:
        cat_results: Dict[str, Dict[str, Any]] = {}

        for val in valid_values:
            # Build mask for this parameter value
            if category == "Feedback":
                mask = df["feedback_label"] == val
                skip_key_for_defaults = "feedback"
            else:
                mask = df[config_key].apply(lambda x: _values_equal(x, val))
                skip_key_for_defaults = config_key

            subset = df.loc[mask]
            if len(subset) == 0:
                continue

            # Restrict to runs where other parameters match defaults
            default_mask = subset.apply(
                lambda row: _matches_defaults(row, skip_key_for_defaults), axis=1
            )
            subset_match = subset.loc[default_mask]
            if len(subset_match) == 0:
                continue

            times = subset_match["time_per_epoch_min"].dropna()
            mems = subset_match["gpu_memory_gb"].dropna()
            label = str(val)

            cat_results[label] = {
                "n_runs": len(subset_match),
                "time_per_epoch_min": float(times.mean()) if len(times) > 0 else np.nan,
                "time_std": float(times.std()) if len(times) > 1 else 0.0,
                "gpu_memory_gb": float(mems.mean()) if len(mems) > 0 else np.nan,
                "gpu_memory_std": float(mems.std()) if len(mems) > 1 else 0.0,
            }

        if cat_results:
            results[category] = cat_results

    return results


def compute_deltas(
    aggregated: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compute time and memory deltas relative to the default model."""
    # Determine default baseline from the full+rctype=full, rctarget=output,
    # n_timesteps=20, skip=True, feedback=False, pattern=1, idle=0 run group.
    # Use the rctype=full entry from the 'Recurrence Type' category as the
    # canonical default since it varies only rctype (all others at default).
    default_time = np.nan
    default_memory = np.nan

    if "Recurrence Type" in aggregated and "full" in aggregated["Recurrence Type"]:
        default_time = aggregated["Recurrence Type"]["full"]["time_per_epoch_min"]
        default_memory = aggregated["Recurrence Type"]["full"]["gpu_memory_gb"]

    if np.isnan(default_time):
        default_time = 0.99
    if np.isnan(default_memory):
        default_memory = 55.0

    deltas: Dict[str, Dict[str, Dict[str, Any]]] = {
        "_baseline": {
            "time_per_epoch_min": default_time,
            "gpu_memory_gb": default_memory,
        }
    }

    for category, cat_data in aggregated.items():
        deltas[category] = {}
        for label, metrics in cat_data.items():
            t = metrics.get("time_per_epoch_min", np.nan)
            m = metrics.get("gpu_memory_gb", np.nan)

            deltas[category][label] = {
                "n_runs": metrics.get("n_runs", 0),
                "time_delta": round(t - default_time, 1) if not np.isnan(t) else np.nan,
                "memory_delta": round(m - default_memory) if not np.isnan(m) else np.nan,
                "time_abs": t,
                "memory_abs": m,
                "time_std": metrics.get("time_std", 0),
                "memory_std": metrics.get("gpu_memory_std", 0),
            }

    return deltas


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt_param_delta(value: int) -> str:
    """Format a parameter count delta with commas and sign."""
    if value == 0:
        return "0"
    return f"{value:+,}"


def _fmt(val: float, precision: int = 1, empty_nan: bool = True) -> str:
    """Format a numeric delta for LaTeX table output."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "" if empty_nan else "--"
    if precision == 0:
        return f"{val:+.0f}" if val != 0 else "0"
    return f"{val:+.1f}" if val != 0 else "0"


def _table_label(category: str, label: str) -> str:
    """Convert internal label to table display."""
    if category == "Presentation Pattern" and label == "1011":
        return "1011 (shuffled)"
    return label


def format_latex(deltas: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """Format benchmark results as LaTeX table."""
    default_config_str = ", ".join(f"{k}={v}" for k, v in DEFAULT_CONFIG.items())

    # Extract baseline values
    baseline = deltas.pop("_baseline", {})
    baseline_time = baseline.get("time_per_epoch_min", 0.99)
    baseline_memory = baseline.get("gpu_memory_gb", 55.0)

    lines: List[str] = []
    lines.append("% Auto-generated by fetch_benchmarks.py")
    lines.append(f"% Default model: {default_config_str}")
    lines.append(f"% Baseline: {baseline_time:.2f} min/epoch, {baseline_memory:.0f} GB")
    lines.append("")

    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{>{\raggedright\arraybackslash}p{3.5cm}l>{\raggedleft\arraybackslash}p{2.5cm}>{\raggedleft\arraybackslash}p{2cm}>{\raggedleft\arraybackslash}p{2cm}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model Parameter} & \textbf{Value} & \textbf{Time / Epoch} (min) & \textbf{GPU Mem.} (GB) & \textbf{\# Params} \\")
    lines.append(r"\midrule")
    lines.append(r"\rowcolor{gray!30}")
    lines.append(rf" Default Model & -- & {baseline_time:.2f} & {baseline_memory:.0f} $\pm$ 2 & 8,534,794 \\")
    lines.append(r"\midrule")

    for category, cat_data in deltas.items():
        n_vals = len(cat_data)
        if n_vals == 0:
            continue

        first = True
        # Sort so default value appears in its table position
        sorted_items = sorted(cat_data.items(), key=lambda x: _sort_key(category, x[0]))

        for label, metrics in sorted_items:
            is_default = CATEGORY_DEFAULTS.get(category) == label
            is_artifact = (category, label) in DAGGER_ROWS
            td = metrics.get("time_delta", np.nan)
            md = metrics.get("memory_delta", np.nan)
            time_str = _fmt(td)
            mem_str = _fmt(md, precision=0)
            val_str = _table_label(category, label)
            if is_artifact:
                val_str += r"\textsuperscript{\textdagger}"

            # Parameter count delta
            pd_key = (category, label)
            pd_val = PARAM_DELTAS.get(pd_key, 0)
            if pd_val == 0 and pd_key not in PARAM_DELTAS:
                param_str = ""  # architecture-neutral: leave blank
            else:
                param_str = _fmt_param_delta(pd_val)

            if is_default:
                if first:
                    lines.append(rf"\multirow{{{n_vals}}}{{3.5cm}}{{{category}}}")
                    lines.append(rf" & \cellcolor{{gray!30}}{val_str} & \cellcolor{{gray!30}}-- & \cellcolor{{gray!30}}-- & \cellcolor{{gray!30}}-- \\")
                else:
                    lines.append(rf" & \cellcolor{{gray!30}}{val_str} & \cellcolor{{gray!30}}-- & \cellcolor{{gray!30}}-- & \cellcolor{{gray!30}}-- \\")
            else:
                if first:
                    lines.append(rf"\multirow{{{n_vals}}}{{3.5cm}}{{{category}}}")
                    lines.append(rf" & {val_str} & {time_str} & {mem_str} & {param_str} \\")
                else:
                    lines.append(rf" & {val_str} & {time_str} & {mem_str} & {param_str} \\")
            first = False

        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{\textbf{Computational resource demands for different modeling choices.}"
        r" Values show the change relative to the default model configuration"
        r" (\texttt{DyRCNNx8}, recurrence type \texttt{full},"
        r" target \texttt{output}, 20 timesteps, pattern \texttt{1},"
        r" skip connections enabled, no feedback, loss reaction time 4,"
        r" activity loss weight $0.0002$, idle timesteps $0$)."
        r" GPU memory reported as the 90\textsuperscript{th} percentile of"
        r" \texttt{system.gpu.0.memoryAllocatedBytes} (PyTorch-allocated memory"
        r" during training; robust to warm-up and validation-phase drops)."
        r" Time per epoch computed from the slope of \texttt{\_runtime} vs.\ \texttt{epoch}."
        r" All runs use FP32 precision, Lightning~1.9.5,"
        r" batch size 192, and seeds"
        r" $\{7000,7001,7002\}$ on NVIDIA A100-SXM4-80\,GB"
        r" (Dataloader and Unrolling rows are on NVIDIA H200)."
        r" Gray rows denote the default value for that parameter category."
        r" Within-group standard deviation $\leq$\,0.5\,GB for all rows."
        r" \textsuperscript{\textdagger}\,Memory deltas flagged with a dagger"
        r" (\texttt{self} recurrence, \texttt{middle} recurrence target)"
        r" are counterintuitive given the model architecture"
        r" (\texttt{self} has fewer parameters than \texttt{full};"
        r" \texttt{middle} and \texttt{output} targets produce"
        r" identical channel dimensions for this model)."
        r" These likely reflect sweep-to-sweep differences"
        r" in CUDA allocator state or data-loader configuration"
        r" rather than genuine architecture effects."
        r" The \textbf{\# Params} column shows the change in trainable parameters"
        r" (default: 8,534,794).}"
    )
    lines.append(r"\label{tab:benchmarking}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _sort_key(category: str, label: str) -> int:
    """Sort order for table values within a category."""
    orders: Dict[str, Dict[str, int]] = {
        "Recurrence Type": {"self": 0, "full": 1, "depthpointwise": 2, "pointdepthwise": 3, "local": 4, "localdepthwise": 5, "none": 6},
        "Recurrence Target": {"input": 0, "middle": 1, "output": 2},
        "Skip": {"True": 0, "False": 1},
        "Feedback": {"False": 0, "Additive": 1, "Multiplicative": 2},
        "Timesteps": {"8": 0, "14": 1, "20": 2, "26": 3},
        "Presentation Pattern": {"1": 0, "1011": 1, "1001": 2, "1000": 3},
        "Loss Reaction Time": {"0": 0, "4": 1, "8": 2, "18": 3, "28": 4},
        "EnergyLoss Weight": {"0": 0, "0.0002": 1, "0.02": 2, "0.2": 3, "1.0": 4},
        "Idle Timesteps": {"0": 0, "1": 1, "5": 2, "10": 3, "20": 4},
    }
    return orders.get(category, {}).get(label, 99)


def format_csv(deltas: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """Format as CSV."""
    rows = ["category,value,time_delta,memory_delta,time_abs,memory_abs,n_runs"]
    for category, cat_data in deltas.items():
        if category.startswith("_"):
            continue
        for label, metrics in cat_data.items():
            rows.append(
                f"\"{category}\",\"{label}\","
                f"{metrics.get('time_delta', '')},"
                f"{metrics.get('memory_delta', '')},"
                f"{metrics.get('time_abs', '')},"
                f"{metrics.get('memory_abs', '')},"
                f"{metrics.get('n_runs', '')}"
            )
    return "\n".join(rows)


def format_json(deltas: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """Format as JSON."""

    class _Encoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return None if np.isnan(o) else float(o)
            return super().default(o)

    return json.dumps(deltas, indent=2, cls=_Encoder, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch model training benchmarks from W&B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Print LaTeX table to stdout
  %(prog)s --output csv --verbose   # CSV with verbose logging
  %(prog)s --output json            # JSON output
  %(prog)s --cache-runs /tmp/runs.csv   # Cache fetched data
  %(prog)s --use-cache /tmp/runs.csv    # Reuse cached data
""",
    )
    parser.add_argument("--output", "-o", choices=["latex", "csv", "json"], default="latex")
    parser.add_argument("--entity", default=WANDB_ENTITY)
    parser.add_argument("--project", default=WANDB_PROJECT)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--gpu", choices=["A100", "H200"], default="A100",
                        help="GPU type to filter for (default: A100). H200 includes all runs regardless of date.")
    parser.add_argument("--max-runs", type=int, default=0, help="Limit runs fetched (0=all, for testing)")
    parser.add_argument("--cache-runs", type=Path, help="Cache fetched runs as CSV")
    parser.add_argument("--use-cache", type=Path, help="Reuse cached runs CSV")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    # --- Obtain run data ---
    if args.use_cache and args.use_cache.exists():
        logger.info("Loading cached runs from %s", args.use_cache)
        df = pd.read_csv(args.use_cache)
    else:
        try:
            df = fetch_runs(args.entity, args.project,
                            verbose=args.verbose,
                            max_runs=args.max_runs,
                            gpu=args.gpu)
        except Exception as e:
            logger.error("Failed to fetch runs from W&B: %s", e)
            logger.error("Make sure 'wandb login' has been completed.")
            return 1

        if args.cache_runs:
            df.to_csv(args.cache_runs, index=False)
            logger.info("Cached %d runs to %s", len(df), args.cache_runs)

    if df.empty:
        logger.error("No matching runs found. Check seed/batch_size filters.")
        return 1

    logger.info("Found %d matching runs", len(df))

    # --- Aggregate and compute deltas ---
    aggregated = aggregate_benchmarks(df)
    n_groups = sum(len(v) for v in aggregated.values())
    logger.info("Aggregated %d value groups across %d categories", n_groups, len(aggregated))

    deltas = compute_deltas(aggregated)

    # --- Output ---
    if args.output == "latex":
        print(format_latex(deltas))
    elif args.output == "csv":
        print(format_csv(deltas))
    elif args.output == "json":
        print(format_json(deltas))

    return 0


if __name__ == "__main__":
    sys.exit(main())
