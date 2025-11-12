"""Sparse Distributed Representation (SDR) class implementation fist pass
The new Sdr.py is in another branch. This is just a placeholder for now."""

from typing import Callable, List, Optional


class SDR:

    elem_dense = int  #: Dense element type used for storing SDR bits.
    elem_sparse = int  #: Sparse index type mirroring the C++ implementation.
    sdr_dense_t = List[elem_dense]  #: Alias for the dense SDR container type.
    sdr_sparse_t = List[elem_sparse]  #: Alias for the sparse SDR container type.
    sdr_coordinate_t = List[List[int]]  #: Alias representing coordinates grouped per dimension.
    sdr_callback_t = Callable[[], None]  #: Callback signature invoked on SDR state changes.

    def __init__(self, dimensions: list[int]):
        self.size = 1
        self.dimensions_ = dimensions
        for dim in dimensions:
            self.size *= dim
        self.sparse = []

        # I think we need these for the other types of encoders like RDSE
        self.dense = []
        self.coordinates = []
        self.dense_valid = False
        self.sparse_invalid = False
        self.coordinates_valid = False
        self.__callbacks: List[Optional[sdr_callback_t]] = []
        self.__destroy_callbacks: List[Optional[sdr_callback_t]] = []

    def zero(self):
        self.sparse = []

    def get_sparse(self):
        return self.sparse

    def set_sparse(self, dimensions: List[int]):
        self.sparse = dimensions

    def set_dense(self, value):
        assert len(value) == self.size
        self.dense, value = value, self.dense
        self.set_dense_inplace()

    def set_dense_inplace(self):
        assert len(self.dense) == self.size
        self.clear()
        self.dense_valid = True
        self.sparse = [i for i, v in enumerate(self.dense) if v != 0]
        self.do_callbacks()

    def clear(self):
        self.dense_valid = False
        self.sparse_valid = False
        self.coordinates_valid = False

    def do_callbacks(self) -> None:
        """Notify registered watchers that the SDR value has changed."""

        for callback in self.__callbacks:
            if callback is not None:
                callback()

    def get_dense(self):
        return self.dense
