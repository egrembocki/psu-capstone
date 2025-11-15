"""Encoder Handler to build composite SDRs"""

import copy
from typing import List, Self

import pandas as pd

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.encoder_layer.sdr import SDR

# four methods to set parameters? for each encoder?
# or just set them when initializing the encoder and not have a separate method?


class EncoderHandler:
    """Handles multiple encoders to create composite SDRs"""

    def __new__(cls, encoders: List[BaseEncoder]) -> Self:
        """Singleton pattern implementation -- only one instance of EncoderHandler allowed"""
        if not hasattr(cls, "instance"):
            cls.__instance = super(EncoderHandler, cls).__new__(cls)

            return cls.__instance
        else:
            raise Exception("Only one instance of EncoderHandler allowed")

    def __init__(self, encoders: List[BaseEncoder]):
        self._encoders = encoders

    def build_sdr(self, data: pd.DataFrame) -> SDR:
        """Builds a composite SDR from the provided data using the configured encoders

        Args:
            data (pd.DataFrame): DataFrame of data points to encode"""
        if len(data) != len(self._encoders):
            raise ValueError("Data length does not match number of encoders")
