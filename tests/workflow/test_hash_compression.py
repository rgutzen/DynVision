"""Unit tests for hash compression utilities.

Tests the compute_hash() function used for creating short model identifiers
to avoid filesystem length limitations.
"""

import pytest


# Test data
TEST_CASES = [
    # (args, expected_properties)
    (
        (':tsteps=20+dt=2+tau=5', '42', 'imagenette'),
        {'starts_with': ':hash=', 'length': 14}  # ':hash=' + 8 hex chars
    ),
    (
        ('tsteps=20+dt=2', '42', 'imagenette'),  # No leading colon
        {'starts_with': ':hash=', 'length': 14}
    ),
    (
        (':hash=a7f3c9d4', '42', 'imagenette'),  # Already hashed (idempotent)
        {'starts_with': ':hash=', 'length': 14}
    ),
]


def test_compute_hash_deterministic():
    """Hash function produces same output for same input."""
    # Import here to avoid import errors before implementation
    try:
        from dynvision.workflow.snake_utils import compute_hash
    except ImportError:
        pytest.skip("compute_hash not implemented yet")

    args1 = ':tsteps=20+dt=2+tau=5'
    seed1 = '42'
    data1 = 'imagenette'

    hash1 = compute_hash(args1, seed1, data1)
    hash2 = compute_hash(args1, seed1, data1)

    assert hash1 == hash2, "Same inputs should produce same hash"
    assert hash1.startswith(':hash='), "Hash should start with ':hash='"
    assert len(hash1) == 14, "Hash should be ':hash=' + 8 hex chars"


def test_compute_hash_idempotent():
    """Hashing a hash returns it unchanged."""
    try:
        from dynvision.workflow.snake_utils import compute_hash
    except ImportError:
        pytest.skip("compute_hash not implemented yet")

    args = ':tsteps=20+dt=2'
    seed = '42'
    data = 'imagenette'

    hash1 = compute_hash(args, seed, data)
    hash2 = compute_hash(hash1, seed, data)  # Hash the hash

    assert hash1 == hash2, "Hashing a hash should return it unchanged"


def test_compute_hash_variadic():
    """Hash function works with different numbers of arguments."""
    try:
        from dynvision.workflow.snake_utils import compute_hash
    except ImportError:
        pytest.skip("compute_hash not implemented yet")

    # Two args
    hash1 = compute_hash('arg1', 'arg2')
    assert hash1.startswith(':hash=')

    # Three args
    hash2 = compute_hash('arg1', 'arg2', 'arg3')
    assert hash2.startswith(':hash=')

    # Different args should produce different hashes
    assert hash1 != hash2


def test_compute_hash_strips_colon():
    """Leading colons are stripped before hashing for consistency."""
    try:
        from dynvision.workflow.snake_utils import compute_hash
    except ImportError:
        pytest.skip("compute_hash not implemented yet")

    hash_with_colon = compute_hash(':tsteps=20', '42', 'imagenette')
    hash_without_colon = compute_hash('tsteps=20', '42', 'imagenette')

    assert hash_with_colon == hash_without_colon, \
        "Colon should be stripped before hashing"


def test_compute_hash_different_inputs():
    """Different inputs produce different hashes."""
    try:
        from dynvision.workflow.snake_utils import compute_hash
    except ImportError:
        pytest.skip("compute_hash not implemented yet")

    hash1 = compute_hash(':tsteps=20', '42', 'imagenette')
    hash2 = compute_hash(':tsteps=30', '42', 'imagenette')  # Different args
    hash3 = compute_hash(':tsteps=20', '43', 'imagenette')  # Different seed
    hash4 = compute_hash(':tsteps=20', '42', 'imagenet')    # Different data

    # All should be different
    hashes = {hash1, hash2, hash3, hash4}
    assert len(hashes) == 4, "Different inputs should produce different hashes"


def test_compute_hash_format():
    """Hash has correct format."""
    try:
        from dynvision.workflow.snake_utils import compute_hash
    except ImportError:
        pytest.skip("compute_hash not implemented yet")

    hash_val = compute_hash(':tsteps=20', '42', 'imagenette')

    # Should start with ':hash='
    assert hash_val.startswith(':hash=')

    # Should be exactly 14 characters (':hash=' + 8 hex chars)
    assert len(hash_val) == 14

    # After prefix, should be valid hex
    hex_part = hash_val[6:]  # Skip ':hash='
    assert len(hex_part) == 8
    assert all(c in '0123456789abcdef' for c in hex_part), \
        "Hash should be lowercase hexadecimal"


def test_compute_hash_custom_length():
    """Hash function respects custom length parameter."""
    try:
        from dynvision.workflow.snake_utils import compute_hash
    except ImportError:
        pytest.skip("compute_hash not implemented yet")

    # Test with different length
    hash_10 = compute_hash(':tsteps=20', '42', 'imagenette', length=10)
    assert len(hash_10) == 16, "Should be ':hash=' + 10 hex chars"

    hash_6 = compute_hash(':tsteps=20', '42', 'imagenette', length=6)
    assert len(hash_6) == 12, "Should be ':hash=' + 6 hex chars"


@pytest.mark.parametrize("args,expected", TEST_CASES)
def test_compute_hash_properties(args, expected):
    """Test hash properties with various inputs."""
    try:
        from dynvision.workflow.snake_utils import compute_hash
    except ImportError:
        pytest.skip("compute_hash not implemented yet")

    hash_val = compute_hash(*args)

    assert hash_val.startswith(expected['starts_with'])
    assert len(hash_val) == expected['length']


def test_compute_hash_collision_resistance():
    """Test that similar inputs don't collide (probabilistic)."""
    try:
        from dynvision.workflow.snake_utils import compute_hash
    except ImportError:
        pytest.skip("compute_hash not implemented yet")

    # Generate hashes for similar parameter combinations
    hashes = set()
    for tsteps in range(10, 30):
        for dt in range(1, 5):
            hash_val = compute_hash(f':tsteps={tsteps}+dt={dt}', '42', 'imagenette')
            hashes.add(hash_val)

    # With 8 hex chars (32 bits), collisions should be extremely rare
    # for ~80 inputs
    expected_unique = 20 * 4  # 80 combinations
    assert len(hashes) == expected_unique, \
        f"Hash collision detected: {expected_unique} inputs produced {len(hashes)} unique hashes"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
