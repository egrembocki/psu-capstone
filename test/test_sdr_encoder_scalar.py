"""Test suite for the SDR Encoder-Scalar."""

import pytest
import logging as looger

from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.encoder_layer.sdr import SDR


@pytest.fixture
def scalar_encoder_instance():
    parameters = ScalarEncoderParameters(
        minimum=0.0,
        maximum=100.0,
        clip_input=True,
        periodic=False,
        active_bits=5,
        sparsity=0.0,
        member_size=50,
        radius=0.0,
        category=False,
        resolution=0.0
    )
    dimensions = [1]

    encoder = ScalarEncoder(parameters, dimensions)
    
    return encoder


def test_scalar_encoder_initialization(scalar_encoder_instance):
    encoder = scalar_encoder_instance
    looger.info("Testing ScalarEncoder Initialization")
    assert isinstance(encoder, ScalarEncoder)
    assert encoder.size == 1
    assert encoder.dimensions == [1]


