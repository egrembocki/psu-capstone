#C:\Users\chris\repos\psu-capstone\Project\src\main\EncoderLayer\Sdr.py

"""This module contains the implementation of the SDR (Sparse Distributed Representation) encoder layer."""

import os
import EncoderLayer.Types as t


from sklearn.naive_bayes import abstractmethod
from typing import Callable, List



# defined types for SDR objects
elem_dense = t.Byte
elem_sparse = t.UInt32
sdr_dense_t = List[elem_dense]
sdr_sparse_t = List[elem_sparse]
sdr_coordinate_t = List[t.UInt32]

# function pointer type for SDR callbacks, not sure how to do this yet
void = t.Handle
def callback(value: void) -> None: ...
sdr_callback_t = Callable[[void], None]


class SdrArray:
    """Class representing a Sparse Distributed Representation (SDR) array."""

    #private members
    __dimensions: list[t.UInt32]
    __size: t.UInt32

    #hooks for function callbacks
    __on_changes: List[sdr_callback_t]
    __destroy_callbacks: List[sdr_callback_t]

    #protected members
    _dense:sdr_dense_t
    _sparse:sdr_sparse_t
    _coordinates:sdr_coordinate_t

    def __init__(self, dimensions: List[t.UInt32], size: t.UInt32) -> None:
        """Initialize the SDR array with given dimensions."""
        self.__dimensions = dimensions
        self.__size = size
        for dim in dimensions:
            self.__size = t.UInt32(int(self.__size) * int(dim))

        self._dense = [elem_dense(0)] * int(self.__size)
        self._sparse = []
        self._coordinates = []

        self.__on_changes = []
        self.__destroy_callbacks = []



    # protected methods

    @abstractmethod
    def clear(self) -> None:
        """Clear the SDR array."""
        self._dense = [elem_dense(0)] * int(self.__size)
        self._sparse.clear()
        self._coordinates.clear()

    
    def do_callbacks(self, callbacks: List[sdr_callback_t]) -> None:
        """Execute the registered callbacks."""
        for callback in callbacks:
            callback(void(id(self)))
        
    @abstractmethod
    def setDenseInplace(self, dense: sdr_dense_t) -> None:
        """Set the dense representation of the SDR array in place."""
        if len(dense) != int(self.__size):
            raise ValueError("Input dense array size does not match SDR size.")
        
        self._dense = dense
        self.updateSparseFromDense()
        self.do_callbacks(self.__on_changes)

    @abstractmethod
    def setCoordinatesInplace(self, coordinates: sdr_coordinate_t) -> None:
        """Set the coordinates representation of the SDR array in place."""
        self._coordinates = coordinates
        self.updateSparseFromCoordinates()
        self.do_callbacks(self.__on_changes) 

    @abstractmethod
    def updateSparseFromDense(self) -> None:
        """Update the sparse representation from the dense representation."""
        self._sparse.clear()
        for index, value in enumerate(self._dense):
            if value != elem_dense(0):
                self._sparse.append(elem_sparse(index))

    @abstractmethod
    def updateSparseFromCoordinates(self) -> None:
        """Update the sparse representation from the coordinates representation."""
        self._sparse.clear()
        for coord in self._coordinates:
            if coord < t.UInt32(int(self.__size)):
                self._sparse.append(elem_sparse(int(coord))) 