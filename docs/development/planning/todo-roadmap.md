# DynVision Development Roadmap

This file outlines the development roadmap for DynVision, organizing future enhancements, tools, and features by implementation priority and category.

**Note**: For documentation fixes and implementation mismatches, see [`todo-docs.md`](./todo-docs.md). This roadmap focuses on new features, tools, and infrastructure improvements.

---

## Quick Navigation

- [Implementation Phases](#implementation-phases)
  - [Phase 1: Quick Wins](#phase-1-quick-wins-1-2-weeks)
  - [Phase 2: Foundation](#phase-2-foundation-2-3-weeks)
  - [Phase 3: Tools & Analysis](#phase-3-tools--analysis-3-4-weeks)
  - [Phase 4: Advanced Features](#phase-4-advanced-features-4-weeks)
- [By Category](#organized-by-category)
- [Effort Matrix](#effort-matrix)
- [Dependencies](#dependency-graph)

---

## Legend

**Size Estimates:**
- 🟢 **SMALL**: < 4 hours
- 🟡 **MEDIUM**: 4-12 hours
- 🔴 **LARGE**: 12+ hours

**Priority Levels:**
- 🔥 **CRITICAL**: Blocking users or causing major friction
- ⚡ **HIGH**: Significant impact on usability/development
- ⭐ **MEDIUM**: Important but not blocking
- 💡 **LOW**: Nice to have, low impact

---

## Implementation Phases

These phases represent a suggested implementation order based on dependencies, impact, and effort.

---

### Phase 1: Quick Wins (1-2 weeks)

High-impact, low-effort tasks that immediately improve user and developer experience.

#### **#41. Quick-Start Jupyter Notebook** 🟢 🔥

- **Type**: Tutorial/Documentation
- **Issue**: No hands-on introduction for new users; README provides overview but no interactive start
- **Needed**:
  - Single notebook in `examples/quickstart.ipynb`
  - Cover: install, load data, define simple model, train, evaluate, visualize
  - Should run in < 10 minutes on CPU
  - Include comments explaining each step
- **Impact**: Fastest way for new users to understand DynVision; reduces onboarding barrier
- **Effort**: 3-4 hours
- **Prerequisites**: None
- **Deliverables**:
  - `examples/quickstart.ipynb`
  - Update README.md to link to notebook
  - Test on fresh environment

#### **#34. Configuration Validator** 🟢 ⚡

- **Type**: Developer Tool
- **Issue**: Config errors only discovered at runtime, often after expensive setup or partial training
- **Needed**:
  - CLI tool: `python -m dynvision.utils.validate_config <config.yaml>`
  - Validate against schema (required fields, types, value ranges)
  - Check file paths exist
  - Warn about common misconfigurations
  - Integration with Snakemake workflow (optional pre-flight check)
- **Impact**: Prevents wasted compute time; catches errors before long-running jobs
- **Effort**: 3-4 hours
- **Prerequisites**: None
- **Deliverables**:
  - `dynvision/utils/validate_config.py`
  - Config schema definitions
  - Unit tests for validation logic
  - Documentation in user guide

#### **#43. Common Errors Troubleshooting Guide** 🟢 ⚡

- **Type**: Documentation
- **Issue**: Users encounter same errors repeatedly (OOM, shape mismatches, import errors, config issues)
- **Needed**:
  - Document in `docs/user-guide/troubleshooting.md`
  - Sections: Installation, Data Loading, Model Initialization, Training, Evaluation
  - Each error with: symptoms, cause, solution, prevention
  - Link from error messages where possible
- **Impact**: Reduces support burden; empowers users to self-solve
- **Effort**: 2-3 hours
- **Prerequisites**: None
- **Deliverables**:
  - `docs/user-guide/troubleshooting.md`
  - Link from FAQ and getting-started guide

#### **#26. Old/Deprecated Files Cleanup** 🟢 ⭐

- **Type**: Code Maintenance
- **Issue**: Repository contains backup files that confuse contributors
- **Location**:
  - `dynvision/data/noise_old.py`
  - `dynvision/models/cornet_old.py`
  - `dynvision/visualization/plot_response_tripytch_old.py`
  - `dynvision/visualization/plot_response_tripytch_old_old.py`
  - `dynvision/visualization/process_test_data_old.py`
- **Needed**:
  - Review each file to confirm not needed
  - Remove from repository
  - Check for any references in other files
  - Update .gitignore to prevent future `*_old.py` files
- **Impact**: Cleaner codebase; reduces confusion
- **Effort**: 30 minutes
- **Prerequisites**: None
- **Deliverables**: Clean repository, updated .gitignore

#### **#25. Pre-commit Hooks Setup** 🟢 ⚡

- **Type**: Developer Experience / Infrastructure
- **Issue**: No automated code quality checks; inconsistent formatting enters repository
- **Needed**:
  - Setup `.pre-commit-config.yaml`
  - Hooks: black (formatting), flake8 (linting), trailing-whitespace, end-of-file-fixer
  - Optional: mypy (type checking), isort (import sorting)
  - Documentation in CONTRIBUTING.md
- **Impact**: Maintains code quality automatically; prevents formatting debates
- **Effort**: 1-2 hours
- **Prerequisites**: None
- **Deliverables**:
  - `.pre-commit-config.yaml`
  - Updated CONTRIBUTING.md
  - CI check that pre-commit passes

---

### Phase 2: Foundation (2-3 weeks)

Core infrastructure that enables confident development and experimentation.

#### **#29. Temporal Dynamics Unit Tests** 🟡 ⚡

- **Type**: Testing
- **Issue**: Core temporal functionality not systematically tested; changes risk breaking dynamics
- **Needed**: Test suite in `tests/test_temporal.py`:
  - **DataBuffer tests**: Circular buffer correctness, overflow handling, indexing
  - **Delay propagation**: Signals delayed by correct number of timesteps
  - **Timestep alignment**: Layers process correct temporal positions
  - **Integration accuracy**: Euler vs RK4 solver outputs match expected trajectories
  - **Edge cases**: Zero delays, delays > buffer size, variable batch sizes
- **Impact**: Confidence in core functionality; prevents regressions
- **Effort**: 8-12 hours
- **Prerequisites**: #32 (CI/CD) for automated running (but can develop tests first)
- **Deliverables**:
  - `tests/test_temporal.py` (100+ lines)
  - `tests/fixtures/temporal.py` (shared test fixtures)
  - Documentation of test coverage

#### **#30. Recurrence Integration Tests** 🟡 ⚡

- **Type**: Testing
- **Issue**: Each recurrence type (full, self, depthwise, local) needs validation across configurations
- **Needed**: Test suite in `tests/test_recurrence.py`:
  - **Type correctness**: Each recurrence type produces expected connectivity
  - **Shape preservation**: Output shapes match input shapes appropriately
  - **Integration methods**: Additive and multiplicative both work
  - **Delay handling**: Recurrent connections properly delayed
  - **Edge cases**: Single timestep, no recurrence, different kernel sizes
  - **Parameter persistence**: State maintained across forward passes
- **Impact**: Ensures core feature works correctly; enables confident refactoring
- **Effort**: 6-8 hours
- **Prerequisites**: #29 (temporal tests provide foundation)
- **Deliverables**:
  - `tests/test_recurrence.py`
  - Test each recurrence type with multiple configurations
  - Integration with CI

#### **#32. CI/CD Pipeline Setup** 🟡 ⚡

- **Type**: Infrastructure
- **Issue**: No automated testing on push/PR; manual testing is error-prone and slow
- **Needed**:
  - GitHub Actions workflow in `.github/workflows/test.yml`
  - Run on: push to main, PRs, manual trigger
  - Jobs: linting (flake8), formatting (black check), type checking (mypy), tests (pytest)
  - Matrix: Python 3.9, 3.10, 3.11
  - Cache dependencies for speed
  - Badge in README showing build status
- **Impact**: Catch issues before merge; maintain code quality
- **Effort**: 4-6 hours
- **Prerequisites**: #29, #30 (tests to run)
- **Deliverables**:
  - `.github/workflows/test.yml`
  - `.github/workflows/lint.yml`
  - Update README with build badges

#### **#42. Minimal Working Examples** 🟡 ⚡

- **Type**: Documentation/Examples
- **Issue**: No minimal reference implementations; users copy-paste from complex models
- **Needed**: Create `examples/minimal/`:
  - **minimal_model.py**: Simplest possible RCNN (2 layers, 10 lines)
  - **minimal_training.py**: Train model on MNIST (50 lines)
  - **minimal_evaluation.py**: Evaluate and print metrics (20 lines)
  - **minimal_visualization.py**: Plot layer responses (30 lines)
  - Each heavily commented, explaining every line
- **Impact**: Clear starting point for customization; reduces learning curve
- **Effort**: 4-6 hours
- **Prerequisites**: #41 (quickstart notebook provides context)
- **Deliverables**:
  - `examples/minimal/` directory with 4 scripts
  - README in directory explaining each
  - Link from main documentation

#### **#27. Dev Environment Setup Script** 🟢 ⭐

- **Type**: Developer Onboarding
- **Issue**: New developers manually install dependencies, configure paths, troubleshoot environment
- **Needed**:
  - Script: `scripts/setup_dev_env.sh`
  - Create conda environment
  - Install dependencies (including dev dependencies)
  - Configure project paths
  - Download small test dataset
  - Run smoke test (import dynvision, create simple model)
  - Print success message with next steps
- **Impact**: Reduces onboarding time from hours to minutes
- **Effort**: 2-3 hours
- **Prerequisites**: None
- **Deliverables**:
  - `scripts/setup_dev_env.sh`
  - Update CONTRIBUTING.md to reference script
  - Test on fresh system

---

### Phase 3: Tools & Analysis (3-4 weeks)

Developer tools and analysis capabilities that enhance productivity and research workflows.

#### **#33. Model Inspector/Debugger** 🟡 ⭐

- **Type**: Developer Tool
- **Issue**: Difficult to examine layer activations, internal state, connectivity during development
- **Needed**:
  - CLI tool: `python -m dynvision.tools.inspect <model_path>`
  - Features:
    - Display model architecture (layers, shapes, parameters)
    - Show connectivity (feedforward, recurrent, skip, feedback)
    - Print temporal configuration (delays, timesteps)
    - Forward pass visualization (activations at each layer/timestep)
    - State inspection (DataBuffer contents, hidden states)
  - Interactive mode for exploring model
- **Impact**: Faster debugging; better understanding of model behavior
- **Effort**: 6-10 hours
- **Prerequisites**: #42 (minimal examples to test on)
- **Deliverables**:
  - `dynvision/tools/inspect.py`
  - CLI interface with argparse
  - Documentation with examples

#### **#38. Benchmarking Suite** 🟡 ⭐

- **Type**: Performance Tool
- **Issue**: No systematic performance comparison; optimization decisions lack data
- **Needed**: Scripts in `scripts/benchmark/`:
  - **recurrence_types.py**: Compare speed of each recurrence type
  - **data_loaders.py**: FFCV vs PyTorch loading speed
  - **memory_usage.py**: Memory consumption by model size/timesteps
  - **precision_modes.py**: Mixed precision speedup measurements
  - **solver_comparison.py**: Euler vs RK4 speed and accuracy
  - Generate markdown tables and plots
- **Impact**: Data-driven optimization; identify bottlenecks
- **Effort**: 8-12 hours
- **Prerequisites**: #37 (profiling scripts provide foundation)
- **Deliverables**:
  - `scripts/benchmark/` directory with 5+ scripts
  - Benchmark results in `docs/benchmarks.md`
  - Automated benchmark running in CI (optional)

#### **#45. Model Comparison Utilities** 🟡 ⭐

- **Type**: Analysis Tool
- **Issue**: Comparing multiple models manually is tedious and error-prone
- **Needed**: Script `dynvision/analysis/compare_models.py`:
  - Input: List of model paths or experiment directories
  - Metrics: Accuracy, parameters, FLOPs, training time, inference time
  - Temporal: Response latency, peak timing by layer
  - Output: Comparison table (markdown/CSV), plots (bar charts, scatter)
  - Support for filtering by model type, experiment, dataset
- **Impact**: Streamline experimental analysis; publication-ready comparisons
- **Effort**: 6-8 hours
- **Prerequisites**: #48 (temporal profiling provides metrics)
- **Deliverables**:
  - `dynvision/analysis/compare_models.py`
  - Example in documentation
  - Jupyter notebook demo

#### **#48. Temporal Response Profiling** 🟡 ⭐

- **Type**: Analysis Tool
- **Issue**: Core feature (temporal dynamics) lacks dedicated analysis tools
- **Needed**: Module `dynvision/analysis/temporal_profiling.py`:
  - **Response latency**: Time to first activation per layer
  - **Peak timing**: When each layer reaches maximum response
  - **Temporal alignment**: Cross-layer synchronization analysis
  - **Delay validation**: Verify propagation delays match config
  - Visualization: Timeline plots, latency cascade diagrams
- **Impact**: Validate biological plausibility; analyze temporal properties
- **Effort**: 6-8 hours
- **Prerequisites**: None
- **Deliverables**:
  - `dynvision/analysis/temporal_profiling.py`
  - Example analysis notebook
  - Add to visualization workflow

#### **#44. API Reference Auto-generation** 🟡 ⭐

- **Type**: Documentation Infrastructure
- **Issue**: No comprehensive API reference; users must read source code
- **Needed**:
  - Setup Sphinx or MkDocs with auto-documentation
  - Configure to extract from docstrings
  - Generate HTML documentation
  - Deploy to GitHub Pages or ReadTheDocs
  - CI workflow to update on push
- **Impact**: Complete, always-up-to-date API docs
- **Effort**: 4-6 hours
- **Prerequisites**: None (but benefits from consistent docstrings)
- **Deliverables**:
  - `docs/conf.py` (Sphinx) or `mkdocs.yml`
  - `.github/workflows/docs.yml`
  - Published documentation site

---

### Phase 4: Advanced Features (4+ weeks)

Long-term enhancements and research-oriented features.

#### **Performance & Optimization**

**#28. Type Hints in Core Code** 🟡 ⭐
- Add type hints to all functions in core modules
- Run mypy for type checking
- Effort: 5-10 hours (gradual)

**#37. Profiling Scripts** 🟢 ⭐
- Create CPU/GPU profiling scripts using cProfile, PyTorch profiler
- Identify bottlenecks in forward/backward passes
- Effort: 3-4 hours

**#39. Memory Usage Analysis** 🟢 ⚡
- Tool to estimate memory from config before training
- Prevent OOM errors on cluster
- Effort: 3-4 hours

**#40. Gradient Checkpointing Support** 🟡 ⭐
- Add gradient checkpointing to recurrent layers
- Enable training of deeper models with limited memory
- Effort: 6-8 hours

#### **Analysis & Research Tools**

**#46. Statistical Analysis Pipeline** 🟡 💡
- Automated statistical tests on experimental results
- Significance testing, effect sizes, confidence intervals
- Effort: 8-10 hours

**#47. Biological Validation Tools** 🔴 💡
- Tools to compare model dynamics with neural data
- Load neural recordings, compute similarity metrics
- Effort: 15-20 hours

#### **Deployment & Production**

**#49. Resource Estimation Tool** 🟢 ⭐
- Estimate CPU, GPU, memory, time from config
- Efficient cluster resource requests
- Effort: 4-5 hours

**#50. Job Submission Templates** 🟢 💡
- Tested templates for SLURM, PBS, SGE
- Include resource requests, environment setup
- Effort: 3-4 hours

**#51. Distributed Training Examples** 🟡 💡
- Working examples with DDP, FSDP
- Multi-node training documentation
- Effort: 6-8 hours

#### **Advanced Features**

**#31. Parameter Validation Tests** 🟡 ⭐
- Test Pydantic validation rules
- Test alias resolution and mode detection
- Effort: 4-6 hours

**#36. Logging Enhancement** 🟡 ⭐
- Standardize logging levels across modules
- Add structured logging (JSON output option)
- Effort: 3-5 hours

**#35. Model Architecture Visualizer** 🟢 ⭐
- Generate diagrams from model definition
- Show connectivity, layer shapes, parameters
- Effort: 4-6 hours

**#52. Transfer Learning Examples** 🟢 ⭐
- Examples of fine-tuning pre-trained models
- Load weights, freeze layers, train classifier
- Effort: 3-4 hours

**#53. Attention-Based Recurrence** 🔴 💡
- Learnable attention-based recurrent connections
- Research feature for more flexible models
- Effort: 15-20 hours

**#54. Learnable Delay Parameters** 🟡 💡
- Make temporal delays learnable parameters
- Biologically inspired adaptive delays
- Effort: 8-12 hours

---

## Organized by Category

Same items reorganized by functional category for cross-reference.

### Development Infrastructure

- **#25**: Pre-commit Hooks Setup 🟢 ⚡
- **#26**: Old/Deprecated Files Cleanup 🟢 ⭐
- **#27**: Dev Environment Setup Script 🟢 ⭐
- **#28**: Type Hints in Core Code 🟡 ⭐

### Testing & Validation

- **#29**: Temporal Dynamics Unit Tests 🟡 ⚡
- **#30**: Recurrence Integration Tests 🟡 ⚡
- **#31**: Parameter Validation Tests 🟡 ⭐
- **#32**: CI/CD Pipeline Setup 🟡 ⚡

### Developer Tools

- **#33**: Model Inspector/Debugger 🟡 ⭐
- **#34**: Configuration Validator 🟢 ⚡
- **#35**: Model Architecture Visualizer 🟢 ⭐
- **#36**: Logging Enhancement 🟡 ⭐

### Performance & Optimization

- **#37**: Profiling Scripts 🟢 ⭐
- **#38**: Benchmarking Suite 🟡 ⭐
- **#39**: Memory Usage Analysis 🟢 ⚡
- **#40**: Gradient Checkpointing Support 🟡 ⭐

### Learning Resources

- **#41**: Quick-Start Jupyter Notebook 🟢 🔥
- **#42**: Minimal Working Examples 🟡 ⚡
- **#43**: Common Errors Troubleshooting Guide 🟢 ⚡
- **#44**: API Reference Auto-generation 🟡 ⭐

### Analysis & Research Tools

- **#45**: Model Comparison Utilities 🟡 ⭐
- **#46**: Statistical Analysis Pipeline 🟡 💡
- **#47**: Biological Validation Tools 🔴 💡
- **#48**: Temporal Response Profiling 🟡 ⭐

### Deployment & Production

- **#49**: Resource Estimation Tool 🟢 ⭐
- **#50**: Job Submission Templates 🟢 💡
- **#51**: Distributed Training Examples 🟡 💡

### Advanced Features

- **#52**: Transfer Learning Examples 🟢 ⭐
- **#53**: Attention-Based Recurrence 🔴 💡
- **#54**: Learnable Delay Parameters 🟡 💡

---

## Effort Matrix

Visual overview of items by size and priority.

| Priority | 🟢 Small (<4h) | 🟡 Medium (4-12h) | 🔴 Large (12h+) |
|----------|---------------|-------------------|-----------------|
| **🔥 Critical** | #41 Quickstart | | |
| **⚡ High** | #34 Config Validator<br>#43 Troubleshooting<br>#25 Pre-commit<br>#39 Memory Analysis | #29 Temporal Tests<br>#30 Recurrence Tests<br>#32 CI/CD<br>#42 Minimal Examples | |
| **⭐ Medium** | #26 Cleanup<br>#27 Dev Setup<br>#37 Profiling<br>#35 Viz Tool<br>#49 Resource Est<br>#52 Transfer Learning | #33 Inspector<br>#38 Benchmarking<br>#45 Model Comparison<br>#48 Temporal Profiling<br>#44 API Docs<br>#28 Type Hints<br>#40 Grad Checkpoint<br>#31 Param Tests<br>#36 Logging | |
| **💡 Low** | #50 Job Templates | #46 Stats Pipeline<br>#51 Distributed<br>#54 Learnable Delays | #47 Bio Validation<br>#53 Attention Recurrence |

**Key Insights:**
- Most high-priority items are small-to-medium (achievable quickly)
- Critical path: #41 → #42 → #44 (user onboarding)
- Foundation path: #25 → #29 → #30 → #32 (testing infrastructure)
- Large items are all low-priority research features

---

## Dependency Graph

Order matters! These dependencies should guide implementation sequence.

```
Foundation Layer:
#25 Pre-commit ──┐
#26 Cleanup ─────┼──> (Can work on anything after this)
#27 Dev Setup ───┘

Testing Layer:
#29 Temporal Tests ──┐
                     ├──> #32 CI/CD Pipeline
#30 Recurrence Tests─┘

Documentation Layer:
#41 Quickstart ──> #42 Minimal Examples ──> #43 Troubleshooting

Tool Layer:
#34 Config Validator  (Independent)
#37 Profiling ──> #38 Benchmarking

Analysis Layer:
#48 Temporal Profiling ──> #45 Model Comparison

Advanced:
#33 Inspector uses #42 (minimal examples for testing)
#38 Benchmarking uses #37 (profiling foundation)
#45 Comparison uses #48 (temporal metrics)
#53, #54 (research features, no dependencies but significant effort)
```

**Critical Paths:**
1. **User Onboarding**: #41 → #42 → #43 (Get users productive fast)
2. **Testing Foundation**: #25 → #29 → #30 → #32 (Enable confident development)
3. **Performance**: #37 → #38 (Data-driven optimization)

---

## Success Metrics

How to measure completion and impact of each phase:

### Phase 1 Success Criteria
- [ ] New users can get started in < 30 minutes (with #41)
- [ ] Zero config-related runtime errors in new experiments (with #34)
- [ ] Support questions decrease by 30% (with #43)
- [ ] No formatting issues in PRs (with #25)

### Phase 2 Success Criteria
- [ ] 80%+ test coverage on temporal and recurrence modules
- [ ] CI catches breaking changes before merge
- [ ] New developers reference minimal examples (not complex models)
- [ ] Onboarding time < 1 hour (with #27)

### Phase 3 Success Criteria
- [ ] Model debugging time reduced by 50% (with #33)
- [ ] Quantitative performance data available for all major components
- [ ] Model comparison automated (no manual metric collection)
- [ ] API documentation visits > GitHub code views

### Phase 4 Success Criteria
- [ ] Cluster jobs fail < 5% due to resource issues (with #49)
- [ ] Transfer learning adoption by users (with #52)
- [ ] Research features enable new publications (with #53, #54)

---

## Implementation Notes

### For Contributors

**Starting a New Item:**
1. Create feature branch: `feature/issue-<number>-<short-name>`
2. Reference this roadmap in PR description
3. Update roadmap with "In Progress" status
4. Mark related items that become unblocked

**Completing an Item:**
1. Ensure all deliverables are present
2. Add tests (if applicable)
3. Update relevant documentation
4. Mark item as complete in roadmap
5. Note any deviations or scope changes

### Scope Creep Prevention

Each item has defined deliverables. If scope expands:
- Create new roadmap item for additional work
- Keep original item focused
- Update dependency graph if needed

### Adjusting Priorities

Priorities may shift based on:
- User feedback and pain points
- Research needs and deadlines
- Resource availability
- External dependencies

Update this roadmap to reflect changes in priority.

---

## Roadmap Maintenance

This roadmap should be reviewed and updated:
- **Weekly**: Mark items as started/completed
- **Monthly**: Adjust priorities based on feedback
- **Quarterly**: Add new items, retire completed sections
- **After major releases**: Reflect on what worked, what didn't

**Last Updated**: 2025-10-23
**Next Review**: TBD

---

## Related Documentation

- **[todo-docs.md](./todo-docs.md)**: Documentation fixes and implementation mismatches
- **[Claude Code Guide](../guides/claude-guide.md)**: Architecture overview and development guide
- **[CONTRIBUTING.md](../../contributing.md)**: How to contribute
- **[README.md](../../../README.md)**: Project overview
- **[Developer Guide](../index.md)**: Overview of development documentation

---

## Questions or Suggestions?

Have ideas for new roadmap items? Found dependencies we missed? Open an issue or PR to discuss!
