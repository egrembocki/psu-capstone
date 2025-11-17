import copy
import math
import random
import struct
from dataclasses import dataclass
from typing import List

import mmh3

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.sdr import SDR

# from SDR import SDR
# from SDR_Encoder_Temp.BaseEncoder import BaseEncoder


@dataclass
class RDSEParameters:
    size: int
    active_bits: int
    sparsity: float
    radius: float
    resolution: float
    category: bool
    seed: int


class RandomDistributedScalarEncoder(BaseEncoder):
    def __init__(self, parameters: RDSEParameters, dimensions: List[int]):
        super().__init__(dimensions)
        self.parameters = copy.deepcopy(parameters)
        self.parameters = self.check_parameters(self.parameters)

        self._size = self.parameters.size
        self._active_bits = self.parameters.active_bits
        self._sparsity = self.parameters.sparsity
        self._radius = self.parameters.radius
        self._resolution = self.parameters.resolution
        self._category = self.parameters.category
        self._seed = self.parameters.seed

    def encode(self, input_value: float, output: SDR) -> None:
        assert output.size == self.size, "Output SDR size does not match encoder size."
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
        # output.setSparse(data) #we may need setDense implemented for SDR class

    # After encode we may need a check_parameters method since most of the encoders have this
    def check_parameters(self, parameters: RDSEParameters):
        assert parameters.size > 0

        num_active_args = 0
        if parameters.active_bits > 0:
            num_active_args += 1
        if parameters.sparsity > 0:
            num_active_args += 1

        assert num_active_args != 0, "Missing argument, need one of: 'activeBits' or 'sparsity'."
        assert (
            num_active_args == 1
        ), "Too many arguments, choose only one of: 'activeBits' or 'sparsity'."

        num_resolution_args = 0
        if parameters.radius > 0:
            num_resolution_args += 1
        if parameters.category:
            num_resolution_args += 1
        if parameters.resolution > 0:
            num_resolution_args += 1

        assert (
            num_resolution_args != 0
        ), "Missing argument, need one of: 'radius', 'resolution', 'category'."
        assert (
            num_resolution_args == 1
        ), "Too many arguments, choose only one of: 'radius', 'resolution', 'category'."

        args = parameters

        if args.sparsity > 0:
            assert 0 <= args.sparsity <= 1
            args.active_bits = int(round(args.size * args.sparsity))
            assert args.active_bits > 0

        if args.category:
            args.radius = 1

        if args.radius > 0:
            args.resolution = args.radius / args.active_bits
        elif args.resolution > 0:
            args.radius = args.active_bits * args.resolution

        while args.seed == 0:
            args.seed = random.getrandbits(32)

        return args


# Tests
"""params = RDSEParameters(
    size=1000, activeBits=0, sparsity=0.10, radius=10, resolution=0, category=False, seed=0
)
encoder = RandomDistributedScalarEncoder(params, dimensions=[params.size])
output = SDR(dimensions=[params.size])
encoder.encode(66, output)
print(output.sparse)"""
