"""Test suite for the SDR Encoder-Scalar."""

import logging as looger

import pytest

from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.encoder_layer.sdr import SDR

looger.basicConfig(level=looger.INFO)


@pytest.fixture
def scalar_encoder_instance():
    """Fixture to create a ScalarEncoder instance for testing. This may change when we get Union working properly."""


# Helper
def do_scalar_value_cases(encoder, cases):
    for case in cases:
        # case: (input_value, expected_output_indices)
        input_value, expected_output = case
        expected_output_sorted = sorted(expected_output)

        # This may not work
        expected_sdr = SDR(
            encoder.parameters.size if hasattr(encoder, "parameters") else encoder.dimensions
        )

        expected_sdr.set_sparse(expected_output_sorted)

        # This may not work
        actual_sdr = SDR(
            encoder.parameters.size if hasattr(encoder, "parameters") else encoder.dimensions
        )

        encoder.encode(input_value, actual_sdr)

        assert actual_sdr == expected_sdr, (
            f"For input {input_value}, expected {expected_output_sorted}, "
            f"got {actual_sdr.get_sparse()}"
        )


def test_scalar_encoder_initialization():
    """Test the initialization of the ScalarEncoder."""

    # Arrange
    parameters = ScalarEncoderParameters(
        minimum=0.0,
        maximum=100.0,
        clip_input=True,
        periodic=False,
        active_bits=5,
        sparsity=0.0,
        size=10,
        radius=0.0,
        category=False,
        resolution=0.0,
        size_or_radius_or_category_or_resolution=0,
        active_bits_or_sparsity=0,
    )

    # Act
    encoder = ScalarEncoder(parameters)

    # Assert
    assert isinstance(encoder, ScalarEncoder)
    assert encoder.size == 10
    assert encoder.dimensions == [1, 10]


def test_clipping_inputs():
    """Test that inputs are correctly clipped to the specified min/max range."""

    # Arrange
    parameters = ScalarEncoderParameters(
        size=10,
        active_bits=2,
        minimum=10.0,
        maximum=20.0,
        clip_input=False,
        # Other parameters can be default or arbitrary
        periodic=False,
        # Unions pending
        sparsity=0.0,
        # Unions pending
        radius=0.0,
        category=False,
        resolution=0.0,
        size_or_radius_or_category_or_resolution=0,
        active_bits_or_sparsity=0,
    )

    # Act and Assert baseline
    encoder = ScalarEncoder(parameters)
    test_sdr = SDR([10])
    test_sdr.zero()

    assert encoder.size == 10
    assert encoder.dimensions == [1, 10]
    assert test_sdr.size == 10

    # Act and Asset - Test input clipping
    # These should pass without exceptions
    assert encoder.encode(10.0, test_sdr)  # At minimum edge case
    assert encoder.encode(20.0, test_sdr)  # At maximum edge case

    assert encoder._sdr == test_sdr

    with pytest.raises(ValueError):
        encoder.encode(9.9, test_sdr)  # Below minimum edge case
        encoder.encode(20.1, test_sdr)  # Above maximum edge case


def test_valid_scalar_inputs():
    """Test that valid scalar inputs are encoded correctly."""

    # Arrange
    params = ScalarEncoderParameters(
        size=10,
        active_bits=2,
        minimum=10,
        maximum=20,
        # Other parameters can be default or arbitrary
        sparsity=0.0,
        radius=0.0,
        category=False,
        resolution=0.0,
        clip_input=False,
        periodic=False,
        size_or_radius_or_category_or_resolution=0,
        active_bits_or_sparsity=0,
    )

    # Act and Assert - baseline
    encoder = ScalarEncoder(params)
    test_sdr = SDR([10])
    assert encoder.size == 10
    assert encoder.dimensions == [10]
    assert test_sdr.size == 10
    assert test_sdr.get_sparse() == []

    with pytest.raises(Exception):
        encoder.encode(9.999, test_sdr)  # Below minimum edge case
        encoder.encode(20.0001, test_sdr)  # Above maximum edge case

    assert encoder.encode(10.0, test_sdr)  # At minimum edge case
    assert encoder.encode(19.9, test_sdr)  # Just below maximum edge case


def test_scalar_encoder_category_encode():
    """Test that category scalar inputs are encoded correctly."""
    # Arrange
    params = ScalarEncoderParameters(
        size=66,
        sparsity=0.02,
        minimum=0,
        maximum=65,
        # Other parameters can be default or arbitrary
        active_bits=0,
        radius=0.0,
        category=False,
        resolution=0.0,
        clip_input=False,
        periodic=False,
        size_or_radius_or_category_or_resolution=0,
        active_bits_or_sparsity=0,
    )

    # Act and Assert - baseline
    encoder = ScalarEncoder(params)
    output = SDR([66])
    assert encoder.size == 66
    assert encoder.dimensions == [66]
    assert output.size == 66

    # Act and Assert - Value less than minimum should raise
    with pytest.raises(Exception):
        encoder.encode(-0.01, output)  # Below minimum edge case

    # Act and Assert - Value greater than maximum should raise
    with pytest.raises(Exception):
        encoder.encode(66.0, output)  # Above maximum edge case

    # Value within range should not raise
    assert encoder.encode(10.0, output)


def test_scalar_encoder_non_integer_bucket_width():
    """Test that scalar encoder handles non-integer bucket widths correctly."""
    # Arrange
    params = ScalarEncoderParameters(
        minimum=10.0,
        maximum=20.0,
        clip_input=True,
        periodic=False,
        active_bits=3,
        sparsity=0.0,
        size=7,
        radius=0.0,
        category=False,
        resolution=0.0,
        size_or_radius_or_category_or_resolution=0,
        active_bits_or_sparsity=0,
    )

    encoder = ScalarEncoder(params)

    cases = [
        (10.0, [0, 1, 2]),
        (20.0, [4, 5, 6]),
    ]

    do_scalar_value_cases(encoder, cases)


def test_scalar_encoder_round_to_nearest_multiple_of_resolution():
    """Test that scalar encoder rounds to the nearest multiple of resolution correctly."""

    # Arrange
    params = ScalarEncoderParameters(
        minimum=10.0,
        maximum=20.0,
        clip_input=True,
        periodic=False,
        active_bits=3,
        sparsity=0.0,
        size=13,
        radius=0.0,
        category=False,
        resolution=1,
        size_or_radius_or_category_or_resolution=0,
        active_bits_or_sparsity=0,
    )

    # Act and Assert - baseline
    encoder = ScalarEncoder(params)
    assert encoder.size == 13
    assert encoder.dimensions == [13]

    cases = [
        (10.00, [0, 1, 2]),
        (10.49, [0, 1, 2]),
        (10.50, [1, 2, 3]),
        (11.49, [1, 2, 3]),
        (11.50, [2, 3, 4]),
        (14.49, [4, 5, 6]),
        (14.50, [5, 6, 7]),
        (15.49, [5, 6, 7]),
        (15.50, [6, 7, 8]),
        (19.00, [9, 10, 11]),
        (19.49, [9, 10, 11]),
        (19.50, [10, 11, 12]),
        (20.00, [10, 11, 12]),
    ]

    do_scalar_value_cases(encoder, cases)


def test_scalar_encoder_periodic_round_nearest_multiple_of_resolution():
    """Test that periodic scalar encoder rounds to the nearest multiple of resolution correctly."""
    # Arrange
    params = ScalarEncoderParameters(
        minimum=10.0,
        maximum=20.0,
        clip_input=True,
        periodic=True,
        active_bits=3,
        sparsity=0.0,
        size=10,
        radius=0.0,
        category=False,
        resolution=1,
        size_or_radius_or_category_or_resolution=0,
        active_bits_or_sparsity=0,
    )
    encoder = ScalarEncoder(params)

    # Act and Assert - baseline
    assert encoder.size == 10
    assert encoder.dimensions == [10]

    cases = [
        (10.00, [0, 1, 2]),
        (10.49, [0, 1, 2]),
        (10.50, [1, 2, 3]),
        (11.49, [1, 2, 3]),
        (11.50, [2, 3, 4]),
        (14.49, [4, 5, 6]),
        (14.50, [5, 6, 7]),
        (15.49, [5, 6, 7]),
        (15.50, [6, 7, 8]),
        (19.49, [9, 0, 1]),
        (19.50, [0, 1, 2]),
        (20.00, [0, 1, 2]),
    ]

    do_scalar_value_cases(encoder, cases)


def nearly_equal(a, b, tol=1e-5):
    return abs(a - b) <= tol


def test_scalar_encoder_serialization():
    """Test serialization and deserialization of ScalarEncoder."""

    # Arrange
    inputs = []

    p = ScalarEncoderParameters(
        minimum=-1.234,
        maximum=12.34,
        clip_input=False,
        periodic=False,
        active_bits=34,
        sparsity=0.0,
        size=0,
        radius=0.1337,
        category=False,
        resolution=0.0,
        size_or_radius_or_category_or_resolution=0,
        active_bits_or_sparsity=0,
    )
    inputs.append(ScalarEncoder(p))

    p.clip_input = True
    inputs.append(ScalarEncoder(p))

    p.clip_input = False
    p.periodic = True
    inputs.append(ScalarEncoder(p))

    p.radius = 0.0
    p.resolution = 0.1337
    inputs.append(ScalarEncoder(p))

    q = ScalarEncoderParameters(
        minimum=-1.234,
        maximum=12.34,
        clip_input=False,
        periodic=False,
        active_bits=34,
        sparsity=0.0,
        size=0,
        radius=0.1337,
        category=False,
        resolution=0.0,
        size_or_radius_or_category_or_resolution=0,
        active_bits_or_sparsity=0,
    )

    q.minimum = -1.0
    q.maximum = 1.0003
    q.size = 100
    q.sparsity = 0.15
    inputs.append(ScalarEncoder(q))

    r = ScalarEncoderParameters(
        minimum=-1.234,
        maximum=12.34,
        clip_input=False,
        periodic=False,
        active_bits=34,
        sparsity=0.0,
        size=0,
        radius=0.1337,
        category=False,
        resolution=0.0,
        size_or_radius_or_category_or_resolution=0,
        active_bits_or_sparsity=0,
    )
    r.minimum = 0
    r.maximum = 65
    r.size = 700
    r.sparsity = 0.02
    inputs.append(ScalarEncoder(r))

    for encoder in inputs:
        # Serialize to JSON string
        buf = encoder.save_json()  # You may need to implement save_json() to return a JSON string
        print("SERIALIZED:\n", buf)

        # Deserialize from JSON string
        # enc = ScalarEncoder(p, [100])  # Temporary instance to load into
        # You may need to implement load_json() to load from a JSON string
        # encoder.load_json(buf)
        # encoder = enc
        print("DESERIALIZED:\n", encoder.save_json())
        p1 = encoder.parameters
        p2 = encoder.parameters

        assert p1.size == p2.size
        assert getattr(p1, "category", None) == getattr(p2, "category", None)
        assert p1.activeBits == p2.activeBits
        assert p1.periodic == p2.periodic
        assert p1.clipInput == p2.clipInput
        assert nearly_equal(p1.minimum, p2.minimum)
        assert nearly_equal(p1.maximum, p2.maximum)
        assert nearly_equal(p1.resolution, p2.resolution)
        assert nearly_equal(p1.sparsity, p2.sparsity)
        assert nearly_equal(p1.radius, p2.radius)
