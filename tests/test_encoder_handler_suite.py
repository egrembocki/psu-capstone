"""Test cases for EncoderHandler to build union SDRs"""

import copy
from typing import List

import pandas as pd
import pytest

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.encoder_handler import EncoderHandler
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.encoder_layer.sdr import SDR


@pytest.fixture
def handler() -> EncoderHandler:
    """Fixture to create an EncoderHandler with multiple encoders"""
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

    encoders: List[BaseEncoder] = []

    encoders = [ScalarEncoder(parameters, [2, 5]), ScalarEncoder(parameters, [2, 5])]

    return EncoderHandler(encoders)


def test_handler_singleton(handler: EncoderHandler):
    """Test that EncoderHandler enforces singleton pattern"""

    # Arrange
    test_encoders = handler._encoders

    # Act
    h1 = handler
    h2 = EncoderHandler(test_encoders)

    # Assert
    assert h1 is h2


def test_copy_deepcopy_sdr(handler: EncoderHandler):
    """Test copying and deep copying SDRs from multiple encoders"""

    test_data = pd.DataFrame({"scalarOne": [25.0], "scalarTwo": [75.0]})

    # Extract scalar values directly for each ,encoder
    input_values = [test_data["scalarOne"].iloc[0], test_data["scalarTwo"].iloc[0]]

    sdrs = []

    encoders: List[BaseEncoder] = handler._encoders

    for i, encoder in enumerate(encoders):
        input_value = float(input_values[i])
        output_sdr = SDR(encoder.dimensions)
        output_sdr.zero()

        assert output_sdr.get_sparse() == []

        try:
            encoder.encode(input_value, output_sdr)
            assert output_sdr.get_sparse() != []
        except Exception as e:
            pytest.fail(f"Encoding failed with exception: {e}")

        copied_sdr = copy.deepcopy(output_sdr)
        sdrs.append(copied_sdr)
        assert sdrs[i].get_sparse() == output_sdr.get_sparse()
        assert sdrs[i].get_sparse() != []
        output_sdr.zero()
        assert sdrs[i].get_sparse() != output_sdr.get_sparse()
