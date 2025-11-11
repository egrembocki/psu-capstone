"""Test suite for the SDR Encoder-Base."""

import pytest


from psu_capstone.encoder_layer.base_encoder import BaseEncoder


@pytest.fixture
def encoder() -> BaseEncoder:
    """Create a BaseEncoder instance for testing."""
    return BaseEncoder(dimensions=[10, 10])


def test_initialize_sets_dimensions_and_size(encoder: BaseEncoder):
    # Arrange
    dimensions = [10, 20, 30]

    # Act
    encoder.initialize(dimensions)

    # Assert
    assert encoder.dimensions == dimensions
    expected_size = SDR(dimensions).size
    assert encoder.size == expected_size


def test_encode_raises_not_implemented_error(encoder: BaseEncoder):
    # Arrange
    output_sdr = SDR([10, 10])

    # Act & Assert
    with pytest.raises(NotImplementedError):
        encoder.encode(42, output_sdr)