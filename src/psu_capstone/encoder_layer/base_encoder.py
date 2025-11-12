"""Base class for all encoder types."""

from abc import ABC, abstractmethod
from typing import List

from sympy import prod
from typing_extensions import Self

from psu_capstone.encoder_layer.sdr import SDR


class BaseEncoder(ABC):

    # private members
    __size: int
    """Scalar size of the encoder output."""

    __dimensions: List[int]
    """Input space of the encoder."""

    __sdr: SDR
    """Internal SDR representation."""

    def __new__(cls) -> Self:
        cls.__sdr = SDR([1])
        cls.__size = 0
        cls.__dimensions = []
        return super().__new__(cls)

    def __init__(self, dimensions: List[int]):
        self.__dimensions = dimensions
        self.__size = prod(int(dim) for dim in self.__dimensions)

        print(
            f"Initialized BaseEncoder with dimensions: {self.__dimensions}"
            f" and size: {self.__size}"
        )

    @property
    def dimensions(self) -> List[int]:
        return self.__dimensions

    @property
    def size(self) -> int:
        return self.__size

    def reset(self):
        raise NotImplementedError("Not implemented yet")

    @abstractmethod
    def encode(self, input_value, output):
        raise NotImplementedError("Subclasses must implement this method")
