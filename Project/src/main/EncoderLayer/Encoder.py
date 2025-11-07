from abc import ABC
from numenta import htm

"""Encoder module :: contains the Encoder class to encode input data."""


class Encoder(ABC):
    """Class to encode input data into SDR format."""

    def __init__(self, data: object):
        
        self.data = data

    def encode(self) -> str:
        """ Encode the input data to SDR format """
        # Implement encoding logic here
        return str(self.data)
