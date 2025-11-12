# Parameter Tests

This directory contains comprehensive unit tests for the DynVision parameter system.

## Test Files

### `test_params_mode_precedence.py`
Tests the complete 5-level parameter precedence hierarchy:
1. CLI arguments (highest priority)
2. `mode.component.param` (e.g., `test.data.train`)
3. `mode.param` (e.g., `test.shuffle`)
4. `component.param` (e.g., `data.train`)
5. `param` (base defaults, lowest priority)

**Key Features:**
- Tests all three mode types: `init`, `train`, `test`
- Verifies cross-component precedence
- Tests CLI override behavior
- End-to-end parameter flow validation

**Run:** `pytest tests/params/test_params_mode_precedence.py -v`

### `test_params_mode_precedence_simple.py`
Simplified precedence tests without file system dependencies.

**Key Features:**
- Focused on core precedence rules
- No file system validation (faster execution)
- Tests basic mode override mechanics
- Suitable for quick sanity checks

**Run:** `pytest tests/params/test_params_mode_precedence_simple.py -v`

### `test_params_mode_config_merging.py`
Tests the internal mechanics of mode-specific configuration processing.

**Key Features:**
- Deep merge algorithm (`_deep_merge`)
- Conflict resolution (`_remove_conflicting_base_keys`)
- Component section flattening (`_flatten_component_sections`)
- Edge cases and error conditions
- End-to-end integration tests

**Run:** `pytest tests/params/test_params_mode_config_merging.py -v`

## Running All Tests

Run all parameter tests:
```bash
pytest tests/params/ -v
```

Run with coverage:
```bash
pytest tests/params/ --cov=dynvision.params --cov-report=html
```

Run specific test class:
```bash
pytest tests/params/test_params_mode_config_merging.py::TestConflictResolution -v
```

Run specific test:
```bash
pytest tests/params/test_params_mode_config_merging.py::TestConflictResolution::test_remove_conflicting_base_keys_simple -v
```

## Shared Fixtures

The `conftest.py` file provides shared fixtures:

- **`temp_config_file`**: Creates temporary YAML config files
- **`temp_model_files`**: Creates temporary model-related files
- **`base_config_defaults`**: Loads the real `config_defaults.yaml`

Example usage:
```python
def test_example(temp_config_file):
    config_path = temp_config_file({"seed": 42})
    params = TestingParams.from_cli_and_config(config_path=str(config_path))
    assert params.seed == 42
```

## Test Organization

Tests are organized into logical classes:

- **Unit tests**: Test individual methods in isolation
- **Integration tests**: Test complete workflows
- **Edge case tests**: Test boundary conditions and error handling

Each test file follows this structure:
1. Module docstring with overview
2. Imports
3. Helper functions (if needed)
4. Test classes grouped by feature
5. `if __name__ == "__main__"` block for standalone execution

## Adding New Tests

When adding new tests:

1. Follow the naming convention: `test_params_<feature>.py`
2. Add module docstring explaining what the file tests
3. Group related tests into classes
4. Use descriptive test names: `test_<what>_<expected_behavior>`
5. Use fixtures from `conftest.py` when possible
6. Add docstrings to test classes and complex tests
7. Update this README with new test file description

## Test Coverage

Current coverage areas:

✅ Mode-specific precedence (5-level hierarchy)  
✅ Config file merging and overrides  
✅ Conflict resolution between base and scoped keys  
✅ Component section flattening  
✅ CLI argument parsing and precedence  
✅ Cross-component parameter routing  
✅ Edge cases and error conditions  

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:

- No external dependencies (beyond package requirements)
- Fast execution (< 1 second total)
- Isolated (no shared state between tests)
- Deterministic (no randomness or timing issues)

All tests use temporary files that are automatically cleaned up.
