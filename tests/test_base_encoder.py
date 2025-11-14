"""Base Encoder Test Suite"""

import pytest

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.sdr import SDR


@pytest.fixture
def base_encoder_instance():
    """Fixture to create a BaseEncoder instance for testing."""

    class TestEncoder(BaseEncoder):
        def encode(self, input_value, output_sdr: SDR) -> bool:
            """Dummy encode method for testing."""
            return True

    return TestEncoder([10, 10])


def test_base_encoder_initialization(base_encoder_instance):
    """Test that the BaseEncoder initializes correctly."""
    encoder = base_encoder_instance
    assert encoder.dimensions == [10, 10]
    assert encoder.size == 100
