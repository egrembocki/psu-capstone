"""Utilities for defining abstract SDR encoders compatible with pdoc."""

from abc import ABC, abstractmethod
from SDR_Encoder_Temp.SDR import SDR
from typing import List


class BaseEncoder(ABC):
    """
    Abstract base class for converting inputs into Sparse Distributed Representations (SDRs).

    Attributes:
        _dimensions (List[int]): Dimensions of the encoded SDR output.
        _size (int): Total number of bits produced by the encoder.
    """

    def __init__(self):  # , dimensions: List[int] = None
        """
        Initialize the encoder metadata to empty defaults.

        Subclasses can configure the concrete dimensions later via :meth:`initialize`.

        Attributes:
            _dimensions (List[int]): List of integers representing the shape of the
                encoded SDR output. This defines the number of bits along each dimension.
            _size (int): The total size (number of bits) of the encoded SDR output.
        """
        self._dimensions: List[int] = []
        self._size: int = 0

        # if dimensions is not None:
        #    self.initialize(dimensions)

    @property
    def dimensions(self) -> List[int]:
        """
        Return the current SDR output dimensions.

        Returns:
            List[int]: Dimensions representing the shape of the encoded SDR.
        """
        return self._dimensions

    @property
    def size(self) -> int:
        """
        Return the total number of bits in the encoded SDR.

        Returns:
            int: Total bit count of the encoded SDR.
        """
        return self._size

    def initialize(self, dimensions: List[int]):
        """
        Configure the encoder with the provided dimensions.

        Args:
            dimensions (List[int]): Shape of the SDR output.

        Returns:
            None
        """
        self._dimensions = list(dimensions)
        self._size = SDR(dimensions).size

    def reset(self):
        """
        Reset any subclass-specific internal state.

        Subclasses can override this to clear caches or running statistics.
        """
        pass

    @abstractmethod
    def encode(self, input_value, output):
        """
        Encode an input value into the provided SDR container.

        Args:
            input_value: Data to be encoded.
            output: SDR instance that receives the encoded bits.

        Returns:
            None

        Raises:
            NotImplementedError: Always, to enforce subclass implementation.
        """
        raise NotImplementedError("Subclasses must implement this method")
