"""Test cases for EncoderHandler to build union SDRs"""

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

    encoders = [
        ScalarEncoder(parameters, [2, 5])
        # ScalarEncoder(parameters, [2, 5])
    ]

    return EncoderHandler([encoder for encoder in encoders])


def test_handler_singleton(handler: EncoderHandler):
    """Test that EncoderHandler enforces singleton pattern"""

    # Arrange

    TestHandler = EncoderHandler([])

    # Act & Assert
    with pytest.raises(Exception):
        another_handler = EncoderHandler([])

    assert isinstance(TestHandler, EncoderHandler)
    assert not isinstance(another_handler, EncoderHandler)


def test_build_sdr(handler: EncoderHandler):
    """Test building a composite SDR from multiple encoders"""
    data = pd.DataFrame({"scalarOne": [25.0], "scalarTwo": [75.0]})

    rows = []
    rows.append(data.iloc[0, 0])
    rows.append(data.iloc[0, 1])

    sdrs = []

    encoders: List[BaseEncoder] = handler._encoders

    for i, encoder in enumerate(encoders):
        input_value = rows[i]
        output_sdr = SDR(encoder.dimensions)
        output_sdr.zero()

        print(f"Encoding value {input_value} with encoder {i}")
        print(f"Encoder dimensions: {encoder.dimensions}, size: {encoder.size}")
        print(f"Output SDR before encoding: {output_sdr}")

        encoder.encode(input_value, output_sdr)

        print(f"Output SDR after encoding: {output_sdr}")

        sdrs.append(output_sdr)
