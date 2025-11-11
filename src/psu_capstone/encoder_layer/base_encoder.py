from abc import ABC, abstractmethod
from typing_extensions import Self
from psu_capstone.encoder_layer.sdr import SDR
from typing import List


class BaseEncoder(ABC):

   # def __new__(cls) -> Self:
       # raise TypeError("Cannot instantiate abstract class")
 

    def __init__(self, dimensions: List[int] = []):
        self._dimensions: List[int] = []
        self._size: int = 0

        if dimensions is not None:
            self.initialize(dimensions)

    @property
    def dimensions(self) -> List[int]:
        return self._dimensions

    @property
    def size(self) -> int:
        return self._size

    def initialize(self, dimensions: List[int]):
        self._dimensions = list(dimensions)
        self._size = SDR(dimensions).__size

    def reset(self):
        raise NotImplementedError("Not implemented yet")

    @abstractmethod
    def encode(self, input_value, output):
        raise NotImplementedError("Subclasses must implement this method")