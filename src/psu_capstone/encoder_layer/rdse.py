import copy
import math
import random
import struct
from dataclasses import dataclass
from typing import List, Optional

import mmh3

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.sdr import SDR
from psu_capstone.utils import Parameters

"""
 * Parameters for the RandomDistributedScalarEncoder (RDSE)
 *
 * Members "activeBits" & "sparsity" are mutually exclusive, specify exactly one
 * of them.
 *
 * Members "radius", "resolution", & "category" are mutually exclusive, specify
 * exactly one of them.
"""


@dataclass
class RDSEParameters:
    """Parameters for the Random Distributed Scalar Encoder (RDSE)."""

    size: int = 1024
    """
    * Member "size" is the total number of bits in the encoded output SDR.
    """
    active_bits: int = 0  # xor with sparsity
    """
    * Member "activeBits" is the number of true bits in the encoded output SDR.
    """
    sparsity: float = 0.02  # xor with active_bits
    """
    * Member "sparsity" is the fraction of bits in the encoded output which this
    * encoder will activate. This is an alternative way to specify the member
    * "activeBits".
    """
    # xor radius, category, resolution
    radius: float = 0.0
    """
    * Member "radius" Two inputs separated by more than the radius have
    * non-overlapping representations. Two inputs separated by less than the
    * radius will in general overlap in at least some of their bits. You can
    * think of this as the radius of the input.
    """
    resolution: float = 1.0
    """
    * Member "resolution" Two inputs separated by greater than, or equal to the
    * resolution will in general have different representations.
    """
    category: bool = False
    """
    * Member "category" means that the inputs are enumerated categories.
    * If true then this encoder will only encode unsigned integers, and all
    * inputs will have unique / non-overlapping representations.
    """
    seed: int = 0
    """
    * Member "seed" forces different encoders to produce different outputs, even
    * if the inputs and all other parameters are the same.  Two encoders with the
    * same seed, parameters, and input will produce identical outputs.
    *
    * The seed 0 is special.  Seed 0 is replaced with a random number.
    """


"""
 * Encodes a real number as a set of randomly generated activations.
 *
 * Description:
 * The RandomDistributedScalarEncoder (RDSE) encodes a numeric scalar (floating
 * point) value into an SDR.  The RDSE is more flexible than the ScalarEncoder.
 * This encoder does not need to know the minimum and maximum of the input
 * range.  It does not assign an input->output mapping at construction.  Instead
 * the encoding is determined at runtime.
"""


class RandomDistributedScalarEncoder(BaseEncoder):
    """Random Distributed Scalar Encoder (RDSE) implementation."""

    def __init__(self, parameters: RDSEParameters, dimensions: List[int] | None = None):
        """Initializes the RandomDistributedScalarEncoder with given parameters."""
        self._parameters = copy.deepcopy(parameters)
        self._parameters = self.check_parameters(self._parameters)

        self._size = self._parameters.size
        self._active_bits = self._parameters.active_bits
        self._sparsity = self._parameters.sparsity
        self._radius = self._parameters.radius
        self._resolution = self._parameters.resolution
        self._category = self._parameters.category
        self._seed = self._parameters.seed

        self._dimensions = dimensions if dimensions is not None else [self._size]

        super().__init__(self._dimensions, self._size)  # pass dimensions and size to BaseEncoder

    def encode(self, input_value: float, output: SDR) -> None:
        """
        Encodes an input value into an SDR with a random distributed scalar encoder.
        We employ the murmur hashing.
        """
        assert output.size == self._size, "Output SDR size does not match encoder size."
        if math.isnan(input_value):
            output.zero()
            return
        if self._category:
            if input_value != int(input_value) or input_value < 0:
                raise ValueError("Input to category encoder must be an unsigned integer")

        data = [0] * self.size

        index = int(input_value / self._resolution)

        for offset in range(self._active_bits):
            hash_buffer = index + offset
            bucket = mmh3.hash(struct.pack("I", hash_buffer), self._seed, signed=False)
            bucket = bucket % self.size
            """
                Don't worry about hash collisions.  Instead measure the critical
                properties of the encoder in unit tests and quantify how significant
                the hash collisions are.  This encoder can not fix the collisions
                because it does not record past encodings.  Collisions cause small
                deviations in the sparsity or semantic similarity, depending on how
                they're handled.
            """
            data[bucket] = 1

        output.set_dense(data)

    # After encode we may need a check_parameters method since most of the encoders have this
    def check_parameters(self, parameters: RDSEParameters):

        num_active_args = 0

        # Ensure size, max and min are valid
        assert parameters.size > 0

        # X-OR toggle for active_bits and sparsity
        if parameters.active_bits > 0:
            parameters.sparsity = 0.0
            num_active_args += 1
        elif parameters.sparsity > 0.0:
            parameters.active_bits = 0
            num_active_args += 1

        # This case may not be needed but adding for safety
        if parameters.active_bits > 0 and parameters.sparsity > 0.0:
            print("Both active_bits and sparsity were set, resetting to default sparsity=0.02")
            parameters.active_bits = 0
            parameters.sparsity = 0.02

        # Assert X-OR toggle for active_bits and sparsity
        assert num_active_args != 0, "Missing argument, need one of: 'activeBits' or 'sparsity'."
        assert (parameters.active_bits == 0) != (
            parameters.sparsity == 0
        ), "Exactly one of 'active_bits' or 'sparsity' must be set."

        # Set active_bits if sparsity is provided
        if parameters.sparsity > 0.0 and parameters.active_bits == 0:
            assert 0 <= parameters.sparsity <= 1
            parameters.active_bits = int(round(parameters.size * parameters.sparsity))
            assert parameters.active_bits > 0

        # Set default for radius/resolution/category if all are unset
        num_resolution_args = 0

        if parameters.radius > 0.0:
            parameters.category = False
            parameters.resolution = 0.0
            num_resolution_args += 1
        elif parameters.resolution > 0.0:
            parameters.category = False
            parameters.radius = 0.0
            num_resolution_args += 1
        elif parameters.category:
            parameters.radius = 0.0
            parameters.resolution = 0.0
            num_resolution_args += 1

        # Assert X-OR toggle for radius, resolution, category

        assert (
            num_resolution_args != 0
        ), "Missing argument, need one of: 'radius', 'resolution', 'category'."
        assert (
            num_resolution_args == 1
        ), "Too many arguments, choose only one of: 'radius', 'resolution', 'category'."
        assert (parameters.radius > 0.0) + (parameters.resolution > 0.0) + (
            parameters.category
        ) == 1, "Exactly one of 'radius', 'resolution', or 'category' must be set."

        # Set radius/resolution based on which is provided
        args = parameters

        if args.category:
            args.radius = 1
            args.resolution = 0.0

        if args.radius > 0.0:
            args.resolution = args.radius / args.active_bits
        elif args.resolution > 0:
            args.radius = args.active_bits * args.resolution

        while args.seed == 0:
            args.seed = random.getrandbits(32)

        return args


# Smoke Tests
if __name__ == "__main__":

    params = RDSEParameters(size=1000, active_bits=20, resolution=1.5)
    encoder = RandomDistributedScalarEncoder(params)
    output = SDR([encoder.size])
    encoder.encode(1, output)
    print(output.get_sparse())
    output2 = SDR([encoder.size])
    encoder.encode(1, output2)
    print(output2.get_sparse())
