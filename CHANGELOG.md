# Changelog

All notable changes to DynVision will be documented in this file.

## [Unreleased]

### Changed
- **Delay validation now rejects off-grid durations (breaking)**: `t_feedforward`,
  `t_recurrence`, `t_feedback`, `t_skip` and `loss_reaction_time` are measured in
  milliseconds but integrated on a fixed `dt` grid, so each must be a whole
  number of `dt` steps. A configuration that violated this was previously
  accepted and silently converted, with the conversion sites disagreeing:
  `int(value / dt)` truncated for the feedforward and recurrent delays while
  `math.ceil` rounded up for the reaction window. `dt=3` with `t_recurrence=4`
  and `loss_reaction_time=4` therefore ran a 3 ms recurrence and a 6 ms
  reaction window. Validation now raises `DynVisionValidationError` (params
  layer) or `ValueError` (model construction), naming the parameter and both
  neighbouring valid values.
- **`dt` must be positive**: previously unchecked, and a non-positive `dt` made
  the delay comparisons undefined.
- **Composite alias maps are derived from their components**: `TrainingParams`
  and `InitParams` listed only a subset of `ModelParams`' aliases, so the model
  timing aliases (`trc`, `tff`, `tfb`, `tsk`, `lossrt`) and others were absent
  from them while present in `TestingParams`. A bare key such as `--trc 4` was
  therefore attributed to the model component unscoped, stored as an inert extra
  attribute (`extra="allow"`), and only reached the model because
  `get_model_kwargs` merges `__pydantic_extra__` and the model classes re-open
  the alias in their own `@alias_kwargs` decorator. That route bypassed
  parameter validation entirely. `CompositeParams.get_component_aliases` now
  scopes each component's own aliases, so adding an alias to a component routes
  it through every composite that contains it.

### Added
- **`ModelParams.recurrence_target` and `DataParams.data_loader` are declared
  fields.** Both were previously reachable only as aliases
  (`rctarget`, `data_loader`) whose target was not a declared field, so the value
  was kept by `extra="allow"` and reached its consumer only via the
  `__pydantic_extra__` merge in `ModelParams.get_model_kwargs` or by direct
  attribute access in `DataModule.setup_dataloader`. Neither was validated, and
  neither appeared in the summary or in provenance. `recurrence_target` is
  constrained to `input` / `middle` / `output` (case-insensitively); the
  `rctarget` alias is unchanged.
- **`tests/params/test_component_alias_routing.py`**,
  **`tests/params/test_timing_multiples.py`** and
  **`tests/params/test_recurrence_target_and_data_loader.py`**:
  completeness and precedence of component alias routing across all three
  composites, the dt-multiple constraint on both validation layers, and the two
  newly declared fields including their forwarding into model kwargs.
- **`tests/base/test_recurrence_target_middle.py`**: the `'middle'` target guard.

### Fixed
- **`RecurrentConnectedConv2d` crashed on `recurrence_target='middle'` without
  `mid_channels`**: `_calculate_recurrence_output_dims` subscripts
  `kernel_size` / `stride` / `padding`, which are only normalised to `(y, x)`
  tuples when `mid_channels` is set; otherwise they stay scalars and the
  subscript raised `TypeError: 'int' object is not subscriptable`. `DyRCNNx2`
  and `DyRCNNx4` build every recurrent layer without `mid_channels`, so
  `rctarget=middle` -- a value in `config_experiments.yaml`'s `rctarget`
  category -- crashed on them. The invalid combination is now rejected at layer
  construction with a message naming `mid_channels`.
- **`get_data_loader_class` on an unset name**: it raised `TypeError: argument of
  type 'NoneType' is not iterable` from a substring test. Now that `data_loader`
  is a field whose unset value is `None`, it raises a `ValueError` naming the
  parameter and listing the registry.
- **`ModelParams.get_timing_summary()` raised `AttributeError`**: it returned
  `effective_tau_steps`, `total_processing_time` and `total_processing_steps`,
  none of which existed, and `is_stable` raised `TypeError` whenever `dt` or
  `tau` was unset. `effective_tau_steps` (`tau / dt`) is now a real property, the
  two ambiguous `total_processing_*` keys are gone, and every field is
  `None`-safe. This is documented public API in the parameter-handling guide.
- **Model-level re-check of effective durations**: `TemporalBase` validates the
  values it actually integrates, which catches a duration left unset in the
  configuration and supplied by a model class default. `t_feedback` and
  `t_skip` are checked only when feedback or skip is enabled, since they are
  inert otherwise.

## [0.1.0] - 2026-06-23

### Added
- **ActivityLoss**: Signed EI-balance mode for excitation/inhibition regularization (renamed from EnergyLoss)
- **Layer-resolution response storage**: Reduces test output size by storing per-layer rather than full tensors
- **Jang et al. (2021) data loader**: Human/CNN noise benchmark dataset for psychometric comparisons
- **Visualization enhancements**: Plot reference models, weight caption metrics, parameter counting, benchmark fetching
- **Seed series 9000+**: Extended seed configuration for robustness experiments
- **Cluster config updates**: New storage paths and SLURM account support
- **Comprehensive documentation**: Expanded user guide, API reference, and developer docs
- **Test suite**: 170 tests covering base, params, data, workflow, cluster, and losses modules

### Changed
- **scikit-learn**: Relaxed version constraint from `~=1.1.0` to `>=1.2.0,<2`
- **Makefile**: Fixed project naming (`rva` → `dvn`, `rhythmic_visual_attention` → `dynvision`)
- **project_paths.py**: Renamed `project_name` to `DynVision_Working` for clarity
- **Documentation**: Fixed 6+ broken references, missing images, and stale class names across reference docs
- **README**: Updated badges, installation instructions, citation with preprint DOI
- **Development status**: Updated classifier from "Planning" to "Beta"

### Removed
- 7 deprecated/backup files (`*_old.py`, `*_backup.py`) — no code references remained

### Fixed
- **ActivityLoss `__del__` guard**: Prevent `AttributeError` when destructor fires before init completes

### Known Issues
- Python 3.12 not yet supported (blocked on FFCV compatibility testing)
- 7 pre-existing test failures (3 ActivityLoss API mismatch, 4 params `data_timesteps` validation)
- Makefile contains user-specific paths (not tracked in git)

### Upcoming (Post-0.1)
- Manuscript-specific code separation to manuscript repository
- Python 3.12 support after FFCV validation
- Logging standardization across params module
- Pre-commit hooks and CI/CD pipeline
- Quick-start Jupyter notebook
- Zenodo DOI archival (see `.zenodo.json`; DOI assigned on first release)

---

## [0.0.1] - Initial development

Pre-release development phase. Core architecture established:
- RCNN models with biologically-inspired dynamics (DyRCNNx2/4/8, AlexNet, ResNet, CorNetRT, CordsNet)
- Continuous-time ODE solvers (Euler, RK4)
- 6 recurrence types (self, full, depthpointwise, pointdepthwise, local, localdepthwise)
- Snakemake workflow system for reproducible experiments
- Pydantic-based parameter validation
- FFCV data loading
- PyTorch Lightning training integration
