from abc import ABC, abstractmethod
from SDR import SDR
from typing import List


class BaseEncoder(ABC):
    """
    Abstract base class for implementing an encoder that converts input values into
    Sparse Distributed Representations (SDRs). This class provides the foundational
    functionality for managing the dimensions and size of the encoded outputs but
    leaves the implementation of the actual encoding process as an abstract method.
    """
    def __init__(self):#, dimensions: List[int] = None
        """
        Initialize the BaseEncoder with optional dimensions.

        Args:
            dimensions (List[int], optional): The dimensions of the encoded SDR output.
                If provided, the `initialize` method will be called to configure the encoder.

        Attributes:
            _dimensions (List[int]): List of integers representing the shape of the
                encoded SDR output. This defines the number of bits along each dimension.
            _size (int): The total size (number of bits) of the encoded SDR output.
        """
        self._dimensions: List[int] = []
        self._size: int = 0

        #if dimensions is not None:
        #    self.initialize(dimensions)

    @property
    def dimensions(self) -> List[int]:
        """
        Get the dimensions of the encoded SDR output.

        Returns:
            List[int]: A list representing the dimensions (shape) of the encoded SDR.
        """
        return self._dimensions

    @property
    def size(self) -> int:
        """
        Get the total size (number of bits) of the encoded SDR output.

        Returns:
            int: The total number of bits in the encoded SDR.
        """
        return self._size

    def initialize(self, dimensions: List[int]):
        """
        Configure the encoder using a list of dimensions.

        This method sets the dimensions and calculates the total size of the
        Sparse Distributed Representation (SDR) based on these dimensions.

        Args:
            dimensions (List[int]): A list of integers representing the shape of the SDR output.
        """
        self._dimensions = list(dimensions)
        self._size = SDR(dimensions).size

    def reset(self):
        """
        Reset the encoder's state.
        """
        pass

    @abstractmethod
    def encode(self, input_value, output):
        """
        Abstract method to encode an input value into an SDR.

        Subclasses must implement this method to define how an input value is transformed
        into a Sparse Distributed Representation (SDR).

        Args:
            input_value: The input data to be encoded.
            output: The SDR object where the encoded result is stored.

        Raises:
            NotImplementedError: If this method is not implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")