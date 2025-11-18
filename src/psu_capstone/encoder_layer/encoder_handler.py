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

    __instance = None

    def __new__(cls, encoders: List) -> "EncoderHandler":
        """Singleton pattern implementation -- only one instance of EncoderHandler allowed"""

        if cls.__instance is None:
            cls.__instance = super(EncoderHandler, cls).__new__(cls)

        return cls.__instance

    def __init__(self, encoders: List[BaseEncoder]):
        self._encoders = encoders

    def build_sdr(self, data: pd.DataFrame) -> SDR:
        """Builds a composite SDR from the provided data using the configured encoders

        Args:
            data (pd.DataFrame): DataFrame of data points to encode"""
        if len(data) != len(self._encoders):
            raise ValueError("Data length does not match number of encoders")

        composite_sdr = SDR([])

        return composite_sdr


# helper parameters setting methods could go here
