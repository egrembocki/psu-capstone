"""Encoder Handler to build composite SDRs"""

from typing import List, Self

from psu_capstone.encoder_layer.base_encoder import BaseEncoder

# from psu_capstone.encoder_layer.categorical_encoder import CategoricalEncoder
from psu_capstone.encoder_layer.date_encoder import DateEncoder
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder
from psu_capstone.encoder_layer.sdr import SDR


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

    def build_sdr(self, data: List) -> SDR:
        """Builds a composite SDR from the provided data using the configured encoders

        Args:
            data (List): List of data points to encode"""

        if len(data) != len(self._encoders):
            raise ValueError("Data length does not match number of encoders")

        sdrs = []

        for i, encoder in enumerate(self._encoders):
            sdr = SDR(data[i])
            encoder.encode(data[i])
            sdrs.append(sdr)

        composite_sdr = SDR.concatenate(sdrs)
        return composite_sdr
