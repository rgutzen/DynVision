# Documentation and Implementation TODO

This file tracks inconsistencies between documentation and implementation, areas needing improvement, and future work items.

## Critical Issues

### 1. Project Naming Inconsistencies

**Issue**: Multiple naming schemes used throughout the project
- **Location**: `Makefile`, `project_paths.py`, documentation
- **Problem**:
  - Makefile uses `rhythmic_visual_attention` instead of `dynvision`
  - `project_paths.py` has both `project_name = "rhythmic_visual_attention"` and `toolbox_name = "DynVision"`
  - Default `working_dir` points to `/home/rgutzen/01_PROJECTS/rhythmic_visual_attention`
- **Impact**: Confusion for new users, inconsistent commands
- **Fix Required**:
  - Update Makefile targets to use `dynvision`
  - Reconcile project naming in `project_paths.py`
  - Update documentation to reflect single canonical name

### 2. Documentation References Non-Existent Files

**Issue**: Documentation links to files that don't exist
- **Location**: `docs/index.md`
- **Missing Links**:
  - `tutorials/visualization-tutorial.md` - Referenced but doesn't exist
  - `user-guide/evaluation.md` - Referenced but doesn't exist
  - `user-guide/faq.md` - Referenced in support section but doesn't exist
  - `user-guide/training.md` - Referenced in model-base.md but doesn't exist
- **Impact**: Broken documentation navigation
- **Fix Required**: Either create these files or remove the references

### 3. Base Class Documentation Mismatch

**Issue**: Documentation describes classes that don't exist exactly as documented
- **Location**: `docs/reference/model-base.md`
- **Problems**:
  - Documents `BaseModel` inheritance from `StorageBuffer`, `Monitoring`, `DtypeDeviceCoordinator` directly
  - In reality, inherits from their `*Mixin` variants: `StorageBufferMixin`, `MonitoringMixin`, `DtypeDeviceCoordinatorMixin`
  - The distinction between base classes and mixins is important for Lightning hooks
- **Impact**: Confusion about which classes to inherit from
- **Fix Required**: Update documentation to accurately reflect Mixin vs base class distinction

### 4. Parameter System Documentation vs Implementation

**Issue**: Extensive parameter handling documentation but unclear integration
- **Location**: `docs/user-guide/parameter-handling.md`
- **Problem**:
  - Documents sophisticated Pydantic-based parameter system in `dynvision/params/`
  - System exists in code but documentation doesn't explain how it integrates with:
    - `@alias_kwargs` decorator system used throughout models
    - Snakemake config system
    - PyTorch Lightning's hyperparameter tracking
  - Unclear which system has precedence when both are used
- **Impact**: Developers unsure which parameter system to use
- **Fix Required**:
  - Document integration points between systems
  - Show examples of both systems working together
  - Clarify when to use Pydantic params vs @alias_kwargs

## Documentation Gaps

### 5. Missing Implementation Details

**Issue**: Documentation describes features but lacks implementation examples
- **Location**: `docs/user-guide/custom-models.md`
- **Gaps**:
  - TODO placeholders for "Training Configurations" section (line 148-152)
  - TODO placeholder for "Troubleshooting Guide" (line 226)
  - References to features but no code examples
- **Impact**: Users can't implement described features
- **Fix Required**: Complete TODO sections with actual content

### 6. Recurrence Type Images Missing

**Issue**: Documentation references images that don't exist
- **Location**: `docs/reference/recurrence-types.md`
- **Missing Images**:
  - `docs/assets/recurrence_types.png` (line 12-13)
  - `docs/assets/self_recurrence.png` (line 21)
  - `docs/assets/full_recurrence.png` (line 49)
  - `docs/assets/depthwise_recurrence.png` (line 78)
  - `docs/assets/local_recurrence.png` (line 127)
- **Impact**: Visual explanations missing
- **Fix Required**: Create or remove image references

### 7. Dynamics Equation Image Missing

**Issue**: Reference to equation image that doesn't exist
- **Location**: `docs/reference/dynamics-solvers.md`
- **Missing**: `docs/assets/dynamical_systems_equation.png` (line 10)
- **Impact**: Key equation not visually displayed
- **Fix Required**: Create image or use inline LaTeX

### 8. Custom Models Template Files Missing

**Issue**: Documentation mentions template files that don't exist
- **Location**: README.md mentions "template files and guides"
- **Problem**: No templates found in repository
- **Impact**: New users can't easily bootstrap custom models
- **Fix Required**: Create template files or remove references

## Minor Inconsistencies

### 9. Model Initialization Sequence

**Issue**: Documentation lists slightly different initialization orders
- **Location**:
  - `docs/reference/model-base.md` (lines 86-88)
  - Actual implementation in `base/temporal.py`
- **Discrepancy**: Documentation missing `_verify_architecture()` step
- **Impact**: Minor - sequence is mostly correct
- **Fix Required**: Add missing step to documentation

### 10. Experiment Configuration Wildcards

**Issue**: Documentation uses different wildcard formats in examples
- **Location**:
  - `docs/user-guide/workflows.md` shows one format
  - `CLAUDE.md` shows expanded format with data_loader and data_args
- **Problem**: Not clear which wildcards are optional vs required
- **Impact**: Users may specify incomplete paths
- **Fix Required**: Standardize wildcard documentation with clear required/optional markers

### 11. Config Mode Detection

**Issue**: Documentation incomplete on mode detection logic
- **Location**: `docs/user-guide/parameter-handling.md`
- **Problem**:
  - States debug mode triggered by `log_level="DEBUG"` OR `epochs <= 5`
  - Doesn't explain what happens if both conditions are met
  - Doesn't explain mode override precedence
- **Impact**: Unpredictable mode activation
- **Fix Required**: Document complete decision tree for mode detection

### 12. GitHub URLs Placeholder

**Issue**: Documentation has placeholder GitHub URLs
- **Location**: `docs/index.md` (line 51)
- **Problem**: "https://github.com/yourusername/dynvision/issues"
- **Impact**: Users can't find actual issue tracker
- **Fix Required**: Update with actual GitHub organization URL

### 13. Repository Citation Missing

**Issue**: Citation section commented out in README
- **Location**: `README.md` (lines 95-109)
- **Problem**: No citation available for users who want to reference DynVision
- **Impact**: Can't properly attribute usage
- **Fix Required**: Add proper citation once paper is published, or add preprint/arxiv

## Code vs Documentation Mismatches

### 14. Data Loader Names

**Issue**: Inconsistent naming of data loaders
- **Location**: `docs/explanation/temporal_dynamics.md` vs `dynvision/data/datasets.py`
- **Documented**: `StimulusDurationDataLoader`, `StimulusIntervalDataLoader`, `StimulusContrastDataLoader`
- **Actual**: `StimulusDuration`, `StimulusInterval`, `StimulusContrast` (without "DataLoader" suffix)
- **Impact**: Code examples won't work
- **Fix Required**: Standardize on actual class names in documentation

### 15. Operation Sequence Names

**Issue**: Documentation uses different operation names than code
- **Location**: `docs/reference/model-architecture.md`
- **Documented**: "tstep", "nonlin", "pool"
- **Actual Implementation**: May vary by model - not clear if these are standardized
- **Impact**: Users may expect operations that don't exist
- **Fix Required**: Document actual available operations from code

### 16. Solver Naming Inconsistency

**Issue**: Documentation uses different solver names
- **Location**: `docs/reference/dynamics-solvers.md`
- **Documented**: `RungeKuttaStep`
- **Config**: Likely expects `rk4` as string identifier
- **Impact**: Configuration may fail if wrong name used
- **Fix Required**: Clarify string identifiers vs class names

## Performance and Optimization

### 17. FFCV Integration Documentation

**Issue**: Documentation mentions FFCV but lacks troubleshooting
- **Location**: Multiple files mention `use_ffcv: true`
- **Problem**: No documentation on:
  - FFCV installation issues
  - When FFCV is beneficial vs overhead
  - Fallback behavior if FFCV fails
- **Impact**: Users may have setup issues
- **Fix Required**: Add FFCV troubleshooting guide

### 18. Mixed Precision Documentation

**Issue**: Mixed precision mentioned but not fully documented
- **Location**: References to `precision: "bf16-mixed"` in configs
- **Problem**:
  - No guidance on GPU requirements
  - No discussion of numerical stability implications
  - No benchmarks showing speedup
- **Impact**: Users may enable inappropriately
- **Fix Required**: Add mixed precision best practices guide

## Future Enhancements

### 19. Test Suite Needed

**Issue**: No formal test suite exists
- **Current State**: No `tests/` directory, no pytest configuration
- **Needed Tests**:
  - Temporal dynamics correctness
  - Recurrence integration
  - Data loading pipelines
  - Parameter validation
  - Model initialization
  - Coordination network building
- **Impact**: No automated validation of code changes
- **Fix Required**: Implement comprehensive test suite

### 20. Visualization Examples Needed

**Issue**: Visualization code exists but lacks examples
- **Location**: `dynvision/visualization/` exists but not well documented
- **Needed**:
  - Example notebooks showing common visualizations
  - Gallery of possible plots
  - Customization guide
- **Impact**: Users may not leverage visualization capabilities
- **Fix Required**: Create visualization tutorial and gallery

### 21. Model Zoo Expansion

**Issue**: README mentions "Comprehensive Model Zoo" but limited models
- **Current Models**: AlexNet, CorNetRT, ResNet variants, CordsNet, DyRCNN
- **Potential Additions**: VGG, EfficientNet, Vision Transformers with recurrence
- **Impact**: Users may expect more pre-built models
- **Fix Required**: Either expand model zoo or adjust marketing language

### 22. Cluster Integration Documentation

**Issue**: Cluster integration exists but not well documented
- **Location**: `cluster/` directory and `docs/user-guide/cluster-integration.md` (referenced but may not exist)
- **Needed**:
  - Cluster-specific setup guides (SLURM, PBS, SGE)
  - Resource allocation best practices
  - Debugging failed cluster jobs
- **Impact**: Difficulty scaling to HPC systems
- **Fix Required**: Complete cluster documentation

## Documentation Style Issues

### 23. Inconsistent Code Example Style

**Issue**: Code examples use different formatting
- **Locations**: Throughout user guides
- **Variations**:
  - Some use full class paths, others don't
  - Some show imports, others assume them
  - Inconsistent comment styles
- **Impact**: Confusing for learners
- **Fix Required**: Establish and follow code example style guide

### 24. Missing Type Hints in Examples

**Issue**: Documentation examples often lack type hints
- **Location**: Throughout tutorials and guides
- **Problem**: Code uses type hints extensively, but examples don't
- **Impact**: Users may not understand expected types
- **Fix Required**: Add type hints to all code examples

## Priority Recommendations

**High Priority** (User-blocking issues):
1. Fix project naming inconsistencies (Issue #1)
2. Remove or create missing documentation links (Issue #2)
3. Complete TODO sections in custom-models.md (Issue #5)
4. Fix data loader naming in docs (Issue #14)

**Medium Priority** (Quality improvements):
5. Clarify parameter system integration (Issue #4)
6. Update base class documentation (Issue #3)
7. Document FFCV setup and troubleshooting (Issue #17)
8. Create visualization examples (Issue #20)

**Low Priority** (Nice to have):
9. Create missing images for recurrence types (Issue #6)
10. Add code example style guide (Issue #23)
11. Expand model zoo (Issue #21)
12. Add comprehensive test suite (Issue #19)

## Notes for Contributors

When updating documentation:
- Always verify class names, method signatures, and file paths against actual code
- Include working code examples that have been tested
- Use consistent terminology (check existing docs)
- Add type hints to code examples
- Consider adding diagrams for complex concepts
- Link to related documentation sections

When updating code:
- Update relevant documentation when changing APIs
- Add docstrings following existing style
- Consider backward compatibility for documented features
- Update CLAUDE.md if architecture changes significantly
