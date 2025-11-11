"""Test suite for the SDR Encoder-Scalar."""

import pytest


from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.encoder_layer.sdr import SDR


@pytest.fixture
def scalar_encoder_instance():
    parameters = ScalarEncoderParameters(
        minimum=0.0,
        maximum=100.0,
        clipInput=True,
        periodic=False,
        category=False,
        activeBits=5,
        sparsity=0.02,
        memberSize=50,
        radius=1.0,
        resolution=0.5
    )
    dimensions = [50]

    encoder = ScalarEncoder(parameters, dimensions)
    
    return encoder


def test_scalar_encoder_initialization(scalar_encoder_instance):
    encoder = scalar_encoder_instance
    assert encoder.size == 50
    assert encoder.dimensions == [50]


