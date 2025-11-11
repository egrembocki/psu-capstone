from abc import ABC, abstractmethod
from SDR import SDR
from typing import List

class BaseEncoder(ABC):

    def __init__(self, dimensions: List[int] = None):
        self._dimensions: List[int] = []
        self._size: int = 0

        if dimensions is not None:
            self.initialize(dimensions)
     #
     # Members dimensions & size describe the shape of the encoded output SDR.
     # This is the total number of bits in the result.
     #
    @property
    def dimensions(self) -> List[int]:
        return self._dimensions

    @property
    def size(self) -> int:
        return self._size

    def initialize(self, dimensions: List[int]):
        self._dimensions = list(dimensions)
        self._size = SDR(dimensions).size

    def reset(self):
        pass

    @abstractmethod
    def encode(self, input_value, output):
        raise NotImplementedError("Subclasses must implement this method")