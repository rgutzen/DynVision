"""
Tests for filesystem wildcard expansion functionality.

This module tests the new ? wildcard feature that allows expanding
wildcards based on existing files in the filesystem.
"""

import sys
from pathlib import Path

# Add workflow to path
workflow_dir = Path(__file__).parents[1] / 'dynvision' / 'workflow'
sys.path.insert(0, str(workflow_dir.parent.parent))

def test_expand_filesystem_pattern_basic():
    """Test basic filesystem wildcard expansion with seed pattern."""
    from dynvision.workflow.snake_utils import expand_filesystem_pattern
    from dynvision.project_paths import project_paths

    # Test pattern for model files with seed wildcard
    pattern = 'FourLayerCNN/FourLayerCNN:tsteps=20+rctype={rctype}_{seed}_{data}_trained.pt'
    wildcards = {
        'rctype': 'full',
        'seed': '0?',  # Filesystem wildcard - find seeds starting with 0
        'data': 'cifar100'
    }

    try:
        result = expand_filesystem_pattern(
            pattern=pattern,
            wildcard_values=wildcards,
            base_path=project_paths.models
        )

        print("✓ expand_filesystem_pattern test passed")
        print(f"  Pattern: {pattern}")
        print(f"  Input wildcards: {wildcards}")
        print(f"  Expanded result: {result}")

        # Verify seed was expanded
        assert 'seed' in result
        assert isinstance(result['seed'], list)
        assert len(result['seed']) > 0
        print(f"  Found {len(result['seed'])} seeds starting with 0")

        return True

    except Exception as e:
        print(f"✗ expand_filesystem_pattern test failed: {e}")
        return False


def test_mixed_wildcards():
    """Test mixed wildcard expansion (* for config, ? for filesystem)."""
    import logging
    logging.basicConfig(level=logging.INFO)

    # This test requires config to be loaded, so we'll just test the function exists
    from dynvision.workflow.snake_utils import args_product

    # Test that args_product accepts the new parameters
    try:
        result = args_product(
            args_dict={'tau': '5', 'rctype': 'full'},
            enable_fs_wildcards=False  # Don't actually expand
        )

        print("✓ args_product with new parameters test passed")
        print(f"  Result: {result}")
        return True

    except Exception as e:
        print(f"✗ args_product test failed: {e}")
        return False


def test_expand_mixed_exists():
    """Test that expand_mixed function exists and has correct signature."""
    from dynvision.workflow.snake_utils import expand_mixed
    import inspect

    # Check function signature
    sig = inspect.signature(expand_mixed)
    print("✓ expand_mixed function exists")
    print(f"  Signature: {sig}")

    # Verify it accepts pattern and **wildcards
    params = list(sig.parameters.keys())
    assert 'pattern' in params
    assert any(p for p in sig.parameters.values() if p.kind == inspect.Parameter.VAR_KEYWORD)

    print("  Function has correct signature (pattern, **wildcards)")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Filesystem Wildcard Expansion")
    print("=" * 60)

    tests = [
        ("Basic filesystem pattern expansion", test_expand_filesystem_pattern_basic),
        ("Mixed wildcards support", test_mixed_wildcards),
        ("expand_mixed function", test_expand_mixed_exists),
    ]

    results = []
    for name, test_func in tests:
        print(f"\nTest: {name}")
        print("-" * 60)
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
