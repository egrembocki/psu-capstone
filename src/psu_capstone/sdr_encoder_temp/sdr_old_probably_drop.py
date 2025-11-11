import math
import random
from typing import Any, Callable, List, Optional

import numpy as np



class SparseDistributedRepresentation:
    # they had two types of constructors, one with dimensions and one without
    def __init__(self, dimensions: Optional[List[int]] = None):
        self.size_ = 0
        self.dimensions_ = []

        # Internal representations in each data format. Not all
        # match at a time, see *_valid below.
        self.dense_ = []
        self.sparse_ = []
        self.coordinates_ = []

        # These flags remember which data formats are up-to-data and which
        # formats need to be updated.
        self.dense_valid = False
        self.sparse_valid = False
        self.coordinates_valid = False

        # These hooks are called every time the SDR's value changes. These can be NUll
        # Pointers! See methods addCallback & removeCallback for API details
        self.callbacks: List[Callable[[], None]] = []
        # These hooks are called when the SDR is destroyed. These can be NULL pointers!
        # See methods addDestroyCallback and removeDestroyCallback for API details.
        self.destroy_callbacks: List[Callable[[], None]] = []
        if dimensions is not None:
            self.initialize(dimensions)

    def initialize(self, dimensions: List[int]):
        if not dimensions:
            raise Exception("SDR has no dimensions!")

        self.dimensions_ = dimensions
        self.size_ = 1
        for d in dimensions:
            self.size_ *= d

        if dimensions != [0]:
            if self.size_ <= 0:
                raise Exception("SDR: all dimensions must be > 0")

        self.dense_valid = False
        self.sparse_valid = True
        self.coordinates_valid = True
        self.coordinates_ = [[] for _ in dimensions]

    def clear(self):
        self.dense_valid = False
        self.sparse_valid = False
        self.coordinates_valid = False

    def do_callbacks(self):
        for func in self.callbacks:
            if func is not None:
                func()

    def setDenseInplace(self):
        if len(self.dense_) != self.size_:
            raise Exception("SDR: dense dimensions do not match with input dimensions!")
        self.clear()
        self.dense_valid = True
        self.do_callbacks()

    def setSparseInplace(self):
        pass

    def setCoordinatesInplace(self):
        pass

    def deconstruct(self):
        self.clear()
        self.size_ = 0
        self.dimensions_.clear()
        for func in self.destroy_callbacks:
            if func is not None:
                func()
        self.callbacks.clear()
        self.destroy_callbacks.clear()

    def reshape(self, new_dims: List[int]):
        pass

    def zero(self):
        pass

    def setDense(self, dense: List[int] | np.ndarray):
        pass

    def getDense(self) -> list[Any]:
        return self.dense_

    def at(self, coordinates: List[int]) -> int:
        pass

    def setSparse(self, value: List[int]):
        pass

    def getSparse(self) -> List[int]:
        return self.sparse_

    def setCoordinates(self, coordinates: List[List[int]]):
        pass

    def getCoordinates(self) -> List[List[int]]:
        return self.coordinates_

    def setSDR(self, value: "SparseDistributedRepresentation"):
        self.reshape(value.dimensions_)
        pass

    # This is a method in C++ from HTM.core, I am not sure how to convert it
    # SparseDistributedRepresentation& SparseDistributedRepresentation::operator=(const SparseDistributedRepresentation& value) {
    #    if( dimensions.empty() ) {
    #        initialize( value.dimensions );
    #    }
    #    set_sdr( value );
    #    return *this;
    # }
    def getOverlap(self, sdr: "SparseDistributedRepresentation") -> int:
        if self.dimensions_ != sdr.dimensions_:
            raise Exception("SDR: dimensions mismatch in overlap!")
        a_sparse = self.get_sparse()
        b_sparse = self.get_sparse()
        i, j, ovlp = 0
        while i < len(a_sparse) and j < len(b_sparse):
            a = a_sparse[i]
            b = b_sparse[j]
            if a == b:
                ovlp += 1
                i += 1
                j += 1
            elif a > b:
                j += 1
            else:
                i += 1
        return ovlp

    # they have multiple randomize methods
    # void SparseDistributedRepresentation::randomize(Real sparsity) {
    #    Random rng( 0 );
    #    randomize( sparsity, rng );
    # }

    # void SparseDistributedRepresentation::randomize(Real sparsity, Random &rng) {
    #    NTA_ASSERT( sparsity >= 0.0f and sparsity <= 1.0f );
    #    UInt nbits = (UInt) std::round( size * sparsity );

    #    SDR_sparse_t range( size );
    #    iota( range.begin(), range.end(), 0u );
    #    sparse_ = rng.sample( range, nbits);
    #    sort( sparse_.begin(), sparse_.end() );
    #    setSparseInplace();
    # }

    # they have multiple of the addNoise methods
    # void SparseDistributedRepresentation::addNoise(Real fractionNoise) {
    #    Random rng( 0 );
    #    addNoise( fractionNoise, rng );
    # }

    # void SparseDistributedRepresentation::addNoise(Real fractionNoise, Random &rng) {
    #    NTA_ASSERT( fractionNoise >= 0. and fractionNoise <= 1. );
    #    NTA_CHECK( ( 1 + fractionNoise) * getSparsity() <= 1. );

    #    const UInt num_move_bits = (UInt) std::round( fractionNoise * getSum() );
    #    const auto& turn_off = rng.sample(get_sparse(), num_move_bits);

    #    auto& dns = get_dense();

    #    vector<UInt> off_pop;
    #    for(UInt idx = 0; idx < size; idx++) {
    #        if( dns[idx] == 0 )
    #            off_pop.push_back( idx );
    #    }
    #    const vector<UInt> turn_on = rng.sample(off_pop, num_move_bits);

    #    for( auto idx : turn_on )
    #        dns[ idx ] = 1;
    #    for( auto idx : turn_off )
    #        dns[ idx ] = 0;

    #    setDenseInplace();
    # }

    def killCells(self, fraction: float, seed: int):
        if not 0 <= fraction <= 1:
            raise Exception("SDR: fraction must be between 0 and 1")
        nkill = int(round(self.size_ * fraction))
        rng = random.Random(seed)
        dns = self.get_dense()
        indices = list(range(self.size_))
        to_kill = rng.sample(indices, nkill)
        for i in to_kill:
            dns[i] = 0
        self.setDense(dns)

    # they have multiple intersection methods
    # void SparseDistributedRepresentation::intersection(
    #        const SDR &input1,
    #        const SDR &input2) {
    #    intersection( { &input1, &input2 } );
    # }

    # void SparseDistributedRepresentation::intersection(vector<const SDR*> inputs) {
    #    NTA_CHECK( inputs.size() >= 2u );
    #    bool inplace = false;
    #    for( size_t i = 0; i < inputs.size(); i++ ) {
    #        NTA_CHECK( inputs[i] != nullptr );
    #        NTA_CHECK( inputs[i]->dimensions == dimensions );
    #        // Check for modifying this SDR inplace.
    #        if( inputs[i] == this ) {
    #            inplace = true;
    #            inputs[i--] = inputs.back();
    #            inputs.pop_back();
    #        }
    #    }
    #    if( inplace ) {
    #        get_dense(); // Make sure that the dense data is valid.
    #    }
    #    if( not inplace ) {
    #        // Copy one of the SDRs over to the output SDR.
    #        const auto &denseIn = inputs.back()->get_dense();
    #        dense_.assign( denseIn.begin(), denseIn.end() );
    #        inputs.pop_back();
    #        // inplace = true; // Now it's an inplace operation.
    #    }
    #    for(const auto &sdr_ptr : inputs) {
    #        const auto &data = sdr_ptr->get_dense();
    #        for(auto z = 0u; z < data.size(); ++z) {
    #            dense_[z] = dense_[z] && data[z];
    #        }
    #    }
    #    SDR::setDenseInplace();
    # }

    # they have multiple set_union methods
    # void
    # SparseDistributedRepresentation::set_union(
    #    const
    # SDR & input1, const
    # SDR & input2) {
    #    set_union({ & input1, & input2} );
    # }

    # void
    # SparseDistributedRepresentation::set_union(vector < const
    # SDR * > inputs) {
    #    NTA_CHECK(inputs.size() >= 2u);
    # bool
    # inplace = false;
    # for (size_t i = 0; i < inputs.size(); i++ ) {
    # NTA_CHECK( inputs[i] != nullptr );
    # NTA_CHECK( inputs[i]->dimensions == dimensions );
    # // Check for modifying this SDR inplace.
    # if ( inputs[i] == this ) {
    # inplace = true;
    # inputs[i--] = inputs.back();
    # inputs.pop_back();
    # }
    # }
    # if ( inplace ) {
    # getDense(); // Make sure that the dense data is valid.
    # }
    # if ( not inplace ) {
    # // Copy one of the SDRs over to the output SDR.
    # const auto & denseIn = inputs.back()->getDense();
    # dense_.assign( denseIn.begin(), denseIn.end() );
    # inputs.pop_back();
    # // inplace = true; // Now it's an inplace operation.
    # }
    # for (const auto & sdr_ptr: inputs) {
    #    const
    # auto & data = sdr_ptr->getDense();
    # for (auto z = 0u; z < data.size(); ++z) {
    # dense_[z] = dense_[z] | | data[z];
    # }
    # }
    # SDR::
    #    setDenseInplace();
    # }

    def concatenate(self, inputs: List["SparseDistributedRepresentation"], axis: int):
        if len(inputs) < 2:
            raise Exception("Need at least two SDRs to concatenate.")
        if axis >= len(self.dimensions_):
            raise Exception("Invalid concatenation axis.")
        concat_axis_size = 0
        for sdr in inputs:
            if len(sdr.dimensions_) != len(self.dimensions_):
                raise Exception("All inputs must have same number of dimensions.")
            for dim in range(len(self.dimensions_)):
                if dim == axis:
                    concat_axis_size += sdr.dimensions_[axis]
                elif sdr.dimensions_[dim] != self.dimensions_[dim]:
                    raise Exception("Dimension mismatch on non-concat axis.")
            if concat_axis_size != self.dimensions_[axis]:
                raise Exception("Axis sizes do not match for concatenation.")
            dense_out = []
            for sdr in inputs:
                dense_out.extend(sdr.get_dense())
            self.dense_ = dense_out
            self.setDenseInplace()
            # messy and not complete

    # They override an operator ==
    # bool SparseDistributedRepresentation::operator==(const SparseDistributedRepresentation &sdr) const {
    #    // Check attributes
    #    if( sdr.size != size or dimensions.size() != sdr.dimensions.size() )
    #        return false;
    #    for( UInt i = 0; i < dimensions.size(); i++ ) {
    #        if( dimensions[i] != sdr.dimensions[i] )
    #            return false;
    #    }
    #    // Check data
    #    return std::equal(
    #        getDense().begin(),
    #        getDense().end(),
    #        sdr.getDense().begin());
    # }

    def add_callback(self, callback: Callable):
        for i, f in enumerate(self.callbacks):
            if f is None:
                self.callbacks[i] = callback
                return i
        self.callbacks.append(callback)
        return len(self.callbacks) - 1

    def removeCallback(self, index: int):
        if index >= len(self.callbacks) or self.callbacks[index] is None:
            raise Exception("Invalid callback.")
        self.callbacks[index] = None

    def addDestroyCallback(self, callback: Callable):
        for i, f in enumerate(self.destroy_callbacks):
            if f is None:
                self.destroy_callbacks[i] = callback
                return i
        self.destroy_callbacks.append(callback)
        return len(self.destroy_callbacks) - 1

    def removeDestroyCallback(self, index: int):
        if (
            index >= len(self.destroy_callbacks)
            or self.destroy_callbacks[index] is None
        ):
            raise Exception("Invalid destroy callback.")
        self.destroy_callbacks[index] = None
