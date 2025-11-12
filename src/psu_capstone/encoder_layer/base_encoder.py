from abc import ABC, abstractmethod
import operator
from typing_extensions import Self
from functools import reduce

from typing import List


class BaseEncoder(ABC):

    # private members
    __size: int = 0

    __dimensions: List[int] = []

    def __new__(cls) -> Self:
        return super().__new__(cls)

    def __init__(self, dimensions: List[int]):
        self.__dimensions = dimensions
        self.__size = reduce(operator.mul, dimensions, 1)

        print(f"Initialized BaseEncoder with dimensions: {self.__dimensions}"
              f" and size: {self.__size}")

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