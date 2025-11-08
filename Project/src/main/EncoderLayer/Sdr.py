#C:\Users\chris\repos\psu-capstone\Project\src\main\EncoderLayer\Sdr.py

"""This module contains the implementation of the SDR (Sparse Distributed Representation) encoder layer."""

import os
import random
import EncoderLayer.types as t


from abc import ABC, abstractmethod
from typing import Callable, List, TypeAlias


# defined types for SDR objects
elem_dense = t.Byte
elem_sparse = t.UInt32
sdr_dense_t = List[elem_dense]
sdr_sparse_t = List[elem_sparse]
sdr_coordinate_t = List[t.UInt32]

# function pointer type for SDR callbacks, not sure how to do this yet
void: TypeAlias = t.Handle

def do_callbacks() -> None: ...
sdr_callback_t = Callable[[], None]

class SdrArray():
    """Class representing a Sparse Distributed Representation (SDR) array."""
    """Class representing a Sparse Distributed Representation (SDR) array."""

    #private members
    __dimensions: list[t.UInt32]
    __size: t.UInt32

    #hooks for function callbacks
    __callbacks: List[sdr_callback_t]
    __destroy_callbacks: List[sdr_callback_t]

    #protected members
    _dense:sdr_dense_t
    _sparse:sdr_sparse_t
    _coordinates:sdr_coordinate_t

    def __init__(self, *args, **kwargs) -> None:
        """Constructor -- Initialize the SDR array with given dimensions."""

        self.__dimensions = args[0] if args else kwargs.get('size', t.UInt32(1))
        self.__size = args[1] if len(args) > 1 else t.UInt32(1)
       

        for dim in self.__dimensions:
            self.__size = t.UInt32(int(self.__size) * int(dim))

        self._dense = [elem_dense(0)] * int(self.__size)
        self._sparse = []
        self._coordinates = []

        self.__on_changes = []
        self.__destroy_callbacks = []



    # protected methods

    @abstractmethod
    def clear(self) -> None:
        """ 
     Remove the value from this SDR by clearing all of the valid flags.  Does
     not actually change any of the data.  Attempting to get the SDR's value
     immediately after this operation will raise an exception.
     """
        self._dense = [elem_dense(0)] * int(self.__size)
        self._sparse.clear()
        self._coordinates.clear()

    
    def do_callbacks(self) -> None:
        """Notify everyone that this SDR's value has officially changed."""
        for callback in self.__callbacks:
            callback()
        
    @abstractmethod
    def set_dense_inplace(self, dense: sdr_dense_t) -> None:
        """ Update the SDR to reflect the value currently inside of the dense array.
     Use this method after modifying the dense buffer inplace, in order to
     propagate any changes to the sparse & coordinate formats."""
        if len(dense) != int(self.__size):
            raise ValueError("Input dense array size does not match SDR size.")
        
        self._dense = dense
        
        self.do_callbacks()

    @abstractmethod
    def set_sparse_inplace(self, sparse: sdr_sparse_t) -> None:
        """ Update the SDR to reflect the value currently inside of the flatSparse
     vector. Use this method after modifying the flatSparse vector inplace, in
     order to propagate any changes to the dense & coordinate formats."""
        
        self._sparse = sparse
        
        self.do_callbacks()
    
    @abstractmethod
    def set_coordinates_inplace(self, coordinates: sdr_coordinate_t) -> None:
        """Update the SDR to reflect the value currently inside of the sparse
     vector. Use this method after modifying the sparse vector inplace, in
     order to propagate any changes to the dense & sparse formats."""
        self._coordinates = coordinates

        self.do_callbacks()


    @abstractmethod
    def destroy(self) -> None:
     
     """Destroy this SDR.  Makes SDR unusable, should error or clearly fail if
     used.  Also sends notification to all watchers via destroyCallbacks.
     This is a separate method from ~SDR so that SDRs can be destroyed long
     before they're deallocated."""

     self.do_callbacks()
     self._dense.clear()
     self._sparse.clear()
     self._coordinates.clear()


    # public methods
    def get_dimensions(self) -> List[t.UInt32]:
        """ Get the dimensions of the SDR array. """
        return self.__dimensions  

    def reshape(self, new_dimensions: List[t.UInt32]) -> None:
        """ Reshape the SDR array to new dimensions. """
        new_size = t.UInt32(1)
        for dim in new_dimensions:
            new_size = t.UInt32(int(new_size) * int(dim))
        
        if new_size != self.__size:
            raise ValueError("New dimensions do not match the total size of the SDR.")
        
        self.__dimensions = new_dimensions


    def zero(self) -> None:
        """ Set all elements of the SDR to zero. """
        self._dense = [elem_dense(0)] * int(self.__size)
        self._sparse.clear()
        self._coordinates.clear()
        
        self.do_callbacks()  


    def set_dense(self, dense: sdr_dense_t) -> None:
        """ Set the SDR's value from a dense array. """
        self.set_dense_inplace(dense)

    @abstractmethod
    def get_dense(self) -> sdr_dense_t:
        """ Get the SDR's value as a dense array. """
        return self._dense

    @abstractmethod
    def at_byte(self, coordinates: List[t.UInt32]) -> t.Byte:
        """ Get the byte value at the specified coordinates. """
        pass

    def set_sparse(self, sparse: sdr_sparse_t) -> None:
        """ Set the SDR's value from a sparse array. """
        self.set_sparse_inplace(sparse)

    @abstractmethod
    def get_sparse(self) -> sdr_sparse_t:
        """ Get the SDR's value as a sparse array. """
        return self._sparse
    
    def set_coordinates(self, coordinates: sdr_coordinate_t) -> None:
        """ Set the SDR's value from a coordinate array. """
        self.set_coordinates_inplace(coordinates)

    @abstractmethod
    def get_coordinates(self) -> sdr_coordinate_t:
        """ Get the SDR's value as a coordinate array. """
        return self._coordinates
    
    @abstractmethod
    def set_sdr(self, other: 'SdrArray') -> None:
        """ Set this SDR's value to be the same as another SDR's value.
          Deep Copy the given SDR to this SDR.  This overwrites the current value of
          this SDR.  This SDR and the given SDR will have no shared data and they
          can be modified without affecting each other.
        @param value An SDR to copy the value of. """

        pass

    def get_sum(self) -> t.UInt32:
        """ Get the sum of all elements in the SDR. """
        return t.UInt32(len(self.get_sparse()))
    
    def get_sparsity(self) -> float:
        """ Get the sparsity of the SDR. """
        return float(len(self.get_sparse())) / float(int(self.__size))
    
    def get_overlap(self, other: 'SdrArray') -> t.UInt32:
        """ Get the overlap between this SDR and another SDR. """
        set_self = set(self.get_sparse())
        set_other = set(other.get_sparse())
        overlap = set_self.intersection(set_other)
        return t.UInt32(len(overlap))
    
    def randomize(self, *args) -> None:
        """ Randomize the SDR with the given sparsity. """

        num_active = int(args[0] * float(int(self.__size)))
        indices = random.sample(range(int(self.__size)), num_active) if args else args[1]

        self._dense = [elem_dense(0)] * int(self.__size)
        for index in indices:
            self._dense[index] = elem_dense(1)

        self._sparse = [elem_sparse(index) for index in indices]
        self._coordinates = [elem_sparse(index) for index in indices]

        self.do_callbacks()


    def add_noise(self, *args) -> None:
        """ Add noise to the SDR by flipping bits based on the noise level. """
        num_flips = int(args[0] * float(int(self.__size)))
        indices = random.sample(range(int(self.__size)), num_flips) if args else args[1]

        for index in indices:
            self._dense[index] = elem_dense(1) if self._dense[index] == elem_dense(0) else elem_dense(0)

        self._sparse = [elem_sparse(i) for i, val in enumerate(self._dense) if val != elem_dense(0)]
        self._coordinates = [elem_sparse(i) for i, val in enumerate(self._dense) if val != elem_dense(0)]

        self.do_callbacks()


    def kill_cells(self, fraction: float, seed: t.UInt32 = t.UInt32(0)) -> None:
        """ Kill a fraction of the active cells in the SDR. """
        active_indices = [i for i, val in enumerate(self._dense) if val != elem_dense(0)]
        num_to_kill = int(fraction * len(active_indices))

        random.seed(int(seed))
        indices_to_kill = random.sample(active_indices, num_to_kill)

        for index in indices_to_kill:
            self._dense[index] = elem_dense(0)

        self._sparse = [elem_sparse(i) for i, val in enumerate(self._dense) if val != elem_dense(0)]
        self._coordinates = [elem_sparse(i) for i, val in enumerate(self._dense) if val != elem_dense(0)]

        self.do_callbacks()


    def intersection(self, other: List['SdrArray']) -> 'SdrArray':
        """ Get the intersection of this SDR with other SDRs. """
        result_sparse = set(self.get_sparse())
        for sdr in other:
            result_sparse = result_sparse.intersection(set(sdr.get_sparse()))

        result = SdrArray(self.__dimensions)
        result.set_sparse(list(result_sparse))
        return result
    

    def union(self, other: List['SdrArray']) -> 'SdrArray':
        """ Get the union of this SDR with other SDRs. """
        result_sparse = set(self.get_sparse())
        for sdr in other:
            result_sparse = result_sparse.union(set(sdr.get_sparse()))

        result = SdrArray(self.__dimensions)
        result.set_sparse(list(result_sparse))
        return result
    

    def concat(self, others: List['SdrArray'], axis: t.UInt32) -> 'SdrArray':
        if axis != t.UInt32(0):
            raise NotImplementedError("Only axis=0 is supported.")
        new_dimensions = self.__dimensions.copy()
        for sdr in others:
            new_dimensions.extend(sdr.get_dimensions())

        result = SdrArray(new_dimensions)
        result_sparse = list(self.get_sparse())
        for sdr in others:
            result_sparse.extend(sdr.get_sparse())

        result.set_sparse(result_sparse)
        return result

    def add_on_change_callback(self, callback: sdr_callback_t) -> t.UInt32:
        """ Add a callback to be called when the SDR changes. """
        self.__on_changes.append(callback)
        return t.UInt32(len(self.__on_changes) - 1)
    
    def remove_on_change_callback(self, index: t.UInt32) -> None:
        """ Remove an on-change callback by index. """
        if int(index) < len(self.__on_changes):
            del self.__on_changes[int(index)]

    def add_destroy_callback(self, callback: sdr_callback_t) -> t.UInt32:
        """ Add a callback to be called when the SDR is destroyed. """
        self.__destroy_callbacks.append(callback)
        return t.UInt32(len(self.__destroy_callbacks) - 1)
    
    def remove_destroy_callback(self, index: t.UInt32) -> None:
        """ Remove a destroy callback by index. """
        if int(index) < len(self.__destroy_callbacks):
            del self.__destroy_callbacks[int(index)]