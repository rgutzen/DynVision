# DynVision Visualization Refactor Plan

_Last updated: 2025-11-20_

## Context
- `plot_responses.py` fails with OOM on ~450 MB `test_data.csv` (≈7.5M rows) when Snakemake runs `plot_responses` rule.
- Current plotting path repeatedly copies the full DataFrame, and Seaborn recomputes confidence intervals for every trace.
- We must keep full temporal resolution (no downsampling) while reducing peak memory.
- New shared helpers should live in `dynvision/utils/visualization_utils.py` so other scripts can reuse them.
- Instead of editing `docs/development/guides/ai-style-guide.md`, we maintain this doc as the working design reference.

## Goals
1. Load only necessary columns from CSVs, downcast numeric types, and standardize categories once.
2. Aggregate data to the plotting granularity before rendering, computing the requested error metric up front.
3. Replace Seaborn's runtime error computation with deterministic plots that reuse the aggregated statistics.
4. Remove redundant DataFrame copies in `_filter_data_for_column`, `_plot_accuracy_panel`, and `_plot_response_ridges`.
5. Promote general-purpose helpers (column detection, aggregation, standardization) into `visualization_utils.py`.
6. Document the workflow so future visualization scripts can follow the same pattern.

## Constraints & Notes
- **No temporal downsampling:** every timestep present in `times_index` must be retained.
- **Error handling:** expose a CLI flag for `--errorbar-type` (e.g., `none`, `std`, `sem`, `ci95`, `percentile`), defaulting to the current visual expectations.
- **Compatibility:** `plot_temporal_ridge_responses` should accept paths or DataFrames as before; aggregated data should stay in-memory but compact.
- **Docs:** keep this file updated during the task; later, we can link it from higher-level guides when stable.

## Implementation Steps

### 1. Introduce Shared Helpers (`visualization_utils.py`)
- `standardize_categorical(series)`: vectorized helper wrapping `_standardize_category_value` logic.
- `determine_plot_columns(config, subplot_var, hue_var, column_var, measures, df_columns=None)`: returns the minimal column set required for reading CSVs.
- `aggregate_plot_data(df, group_keys, value_specs, error_type, min_count=1)`: groups by the provided keys and computes mean + error columns without changing timestep resolution. `value_specs` contains tuples like `(source_column, alias)` so we can aggregate `layer_response` columns and rename them systematically.
- `parse_error_type(arg: str) -> ErrorSpec`: central place to keep logic for std/sem/percentile.

_Status (Nov 20)_: `standardize_series`, `parse_error_type`, `aggregate_plot_data`, and discovery helpers for layer/classifier/measure columns now exist in `visualization_utils.py`. A future enhancement is a compact "column plan" helper that packages the selection + aggregation metadata for reuse across scripts.

### 2. Optimize Data Loading (`plot_responses.py`)
- Before reading CSVs, build `needed_columns` using the new helper plus always-required metadata (`times_index`, `label_index`, etc.). Pass to `pd.read_csv(..., usecols=needed_columns)`.
- After loading:
  - Downcast floats to `float32`, ints to `int32` where safe.
  - Apply `standardize_categorical` for each categorical dimension once; convert to `CategoricalDtype` to shrink memory.
  - Compute `time_ms` once and reuse.

_Status (Nov 21)_: Column-plan builder now drives a header-first load, `usecols` pruning, and categorical standardization. Numeric series are downcasted before aggregation, `label_valid` is synthesized beside `label_index`, and classifier `_id` metadata is preserved so legends keep unit labels.

### 3. Pre-Aggregate for Plotting
- Determine grouping keys: `times_index`, `time_ms`, plus whichever of `column_key`, `subplot_key`, `hue_key` are real columns (skip for special dims like `layers`).
- Build `value_specs`:
  - Accuracy/confidence columns resolved via `_coerce_measure_list` + `resolve_measure_columns`.
  - Response columns derived from `subplot_var` & `hue_var` (layers, classifier_topk, etc.).
- Call `aggregate_plot_data`, obtaining:
  - `agg_df`: compact DataFrame containing means and optional `<col>_err` columns.
  - `available_columns`: metadata to help `_extract_dimension_values` and plotting functions know what exists.

_Status (Nov 21)_: The ridge plot entrypoint now aggregates immediately after loading, using dimension-aware group keys, label-valid maxima, and `first` aggregations for classifier metadata. `time_ms` gets recomputed from the integer `times_index`, so every downstream plot consumes the compact aggregated frame.

### 4. Refactor Plotting Functions
- `_filter_data_for_column` should simply return `df.loc[mask]` on the aggregated table (no `.copy()`), since the data are immutable post-aggregation.
- `_plot_accuracy_panel` & `_plot_response_ridges` should:
  - Use direct `ax.plot` calls with the aggregated mean columns.
  - If `<col>_err` exists, draw shaded error via `ax.fill_between`. The helper should guard against NaNs.
  - Avoid per-loop copies; use boolean masks or grouped views.
- Ensure layer/classifier hue cases fetch the correct aggregated columns (`layer_response_avg`, classifier columns, etc.).

_Status (Nov 21)_: Accuracy panels and ridge plots now consume the aggregated metrics, rely on Matplotlib primitives, and shade `<metric>_err` bands when present. Seaborn remains only for context styling; all heavy lifting runs on the pre-aggregated table.

### 5. CLI & Config Updates
- Add CLI arguments to `plot_responses.py`:
  - `--errorbar-type` (default `std`).
  - Optional `--min-count` for aggregation.
- Thread these through Snakemake (`snake_visualizations.smk`) if needed after verifying defaults; otherwise leave rule unchanged if default behavior matches previous visuals.

_Status (Nov 21)_: Parser now exposes both knobs and forwards them into the plotting entrypoint. Snakemake still relies on defaults, so no workflow change is required until experiments need alternate settings.

### 6. Validation & Future Docs
- After code changes, test with a representative large CSV (if available) or simulate using truncated data to confirm memory stays bounded.
- Update this doc with any deviations encountered during implementation (e.g., additional helper functions, structure changes).
- Later, summarize the stable workflow into `docs/development/guides/claude-guide.md` once design settles.

_Reminder_: Once this refactor ships, capture the "column-plan → aggregation → plotting" flow (with diagrams) in the public-facing dev guides.

## Pending Items / Task Memory
- Need to decide which error metrics to support initially (likely `std` and `sem`).
- Confirm whether `process_test_data.py` already outputs averaged responses; if so, aggregation may simply re-mean identical rows (still needed for error calculation and dedup). Document findings here as they arise.
- If new helper names grow large, consider a dedicated module (e.g., `plotting_data_utils`), but for now keep under `visualization_utils.py` per request.
