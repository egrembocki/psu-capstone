import pandas as pd
import pytest

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.sdr import SDR
from psu_capstone.input_layer.input_handler import InputHandler


class DummyEncoder(BaseEncoder):

    def __init__(self, dimensions=None):
        if dimensions is None:
            dimensions = [32]  # arbitrary SDR size for the test
        super().__init__(dimensions)

    def attach_input(self, df: pd.DataFrame):
        self.input_df = df

    def encode(self, input_value, output_sdr):
        return True


# Note: This is expected to fail because we do not have an HTM interface yet
# from psu_capstone.htm.interface import HTMinterface


class HTMinterface:
    """
    Mock Class for HTM interface
    """


def test_encoder_to_htm_receives_sdr_object():

    # Arrange
    fib_sequence = [0, 1, 1, 2, 3, 5, 8, 13]

    handler = InputHandler()
    df = handler.to_dataframe(fib_sequence)
    assert isinstance(df, pd.DataFrame)

    encoder = DummyEncoder()
    encoder.attach_input(df)

    # HTM interface (not implemented yet)
    htm = HTMinterface()

    # Act
    # Encode a single value
    last_value = df.iloc[-1, 0]
    sdr = SDR([1, 8])
    sdr = encoder.encode(last_value, sdr)

    # Hypothetical interface call: Would accept an SDR instance as input.
    htm.consume_sdr(sdr)

    # Assert
    # Once HTMinterface is implemented, give it some observable state
    assert isinstance(sdr, SDR)
    assert htm.last_received_sdr is sdr
