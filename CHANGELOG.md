# Changelog

All notable changes to DynVision will be documented in this file.

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
