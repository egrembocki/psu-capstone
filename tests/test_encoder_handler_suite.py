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

    test_handler = handler

    # Act & Assert
    assert isinstance(test_handler, EncoderHandler)

    with pytest.raises(Exception):
        EncoderHandler([])


def test_copy_deepcopy_sdr(handler: EncoderHandler):
    """Test copying and deep copying SDRs from multiple encoders"""

    test_data = pd.DataFrame({"scalarOne": [25.0, 75.0, 100.0], "scalarTwo": [75.0, 80.0, 85.0]})

    rows = []
    rows.append(test_data.iloc[0])
    rows.append(test_data.iloc[1])

    sdrs = []

    encoders: List[BaseEncoder] = handler._encoders

    for i, encoder in enumerate(encoders):
        input_value = rows[i]
        output_sdr = SDR(encoder.dimensions)
        output_sdr.zero()

        assert output_sdr.get_sparse() == []

        print(f"Encoding value {input_value} with encoder {i}")
        print(f"Encoder dimensions: {encoder.dimensions}, size: {encoder.size}")
        print(f"Output SDR before encoding: {output_sdr}")

        try:
            encoder.encode(input_value, output_sdr)
            assert output_sdr.get_sparse() != []
            print(f"Output SDR after encoding: {output_sdr}")
        except Exception as e:
            print(f"Encoding failed with error: {e}")

        sdrs.append(copy.deepcopy(output_sdr))

        assert sdrs[i].get_sparse() != []

        assert sdrs[i].get_sparse() == output_sdr.get_sparse()

        output_sdr.zero()

        assert sdrs[i].get_sparse() != []
