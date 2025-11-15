"""Integration tests for Encoder to HTM layer."""

import pytest
from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.sdr import SDR


@pytest.fixture
def TestEncoder() -> BaseEncoder:
    """Fixture for a simple test encoder."""

    class TestEncoder(BaseEncoder):
        """A simple test encoder implementation."""

        def encode(self, input_data: float) -> SDR:
            """Encodes the input data into a simple SDR representation."""
            # For testing purposes, create an SDR with fixed parameters
            sdr = SDR([10, 2])
            sdr.randomize(0.02)
            # Simple encoding logic: activate bits based on input value
            index = int(input_data) % (sdr.num_bits - sdr.active_bits + 1)
            for i in range(sdr.active_bits):
                sdr.set_bit(index + i)
            return sdr
