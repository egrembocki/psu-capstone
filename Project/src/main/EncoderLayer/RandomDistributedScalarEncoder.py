import math
import struct
import random

import mmh3

from BaseEncoder import BaseEncoder
from dataclasses import dataclass

from SDR import SDR


@dataclass
class RDSEParameters:
    """
    Parameters for the RandomDistributedScalarEncoder (RDSE)

    Members "activeBits" & "sparsity" are mutually exclusive, specify exactly one
    of them.

    Members "radius", "resolution", & "category" are mutually exclusive, specify
    exactly one of them.

    Member "size" is the total number of bits in the encoded output SDR.

    Member "activeBits" is the number of true bits in the encoded output SDR.

    Member "sparsity" is the fraction of bits in the encoded output which this
    encoder will activate. This is an alternative way to specify the member
    "activeBits".

    Member "radius" Two inputs separated by more than the radius have
    non-overlapping representations. Two inputs separated by less than the
    radius will in general overlap in at least some of their bits. You can
    think of this as the radius of the input.

    Member "resolution" Two inputs separated by greater than, or equal to the
    resolution will in general have different representations.

    Member "category" means that the inputs are enumerated categories.
    If true then this encoder will only encode unsigned integers, and all
    inputs will have unique / non-overlapping representations.

    Member "seed" forces different encoders to produce different outputs, even
    if the inputs and all other parameters are the same.  Two encoders with the
    same seed, parameters, and input will produce identical outputs.

    The seed 0 is special.  Seed 0 is replaced with a random number.
    """

    size: int
    active_bits: int
    sparsity: float
    radius: float
    resolution: float
    category: bool
    seed: int


class RandomDistributedScalarEncoder(BaseEncoder):
    """
    Encodes a real number as a set of randomly generated activations.

    Description:
    The RandomDistributedScalarEncoder (RDSE) encodes a numeric scalar (floating
    point) value into an SDR.  The RDSE is more flexible than the ScalarEncoder.
    This encoder does not need to know the minimum and maximum of the input
    range.  It does not assign an input->output mapping at construction.  Instead
    the encoding is determined at runtime.

    Note: This implementation differs from Numenta's original RDSE.  The original
    RDSE saved all associations between inputs and active bits for the lifetime
    of the encoder.  This allowed it to guarantee a good set of random
    activations which didn't conflict with any previous encoding.  It also allowed
    the encoder to decode an SDR into the input value which likely created it.
    This RDSE does not save the association between inputs and active bits.  This
    is faster and uses less memory.  It relies on the random & distributed nature
    of SDRs to prevent conflicts between different encodings.  This method does
    not allow for decoding SDRs into the inputs which likely created it.
    """

    memberSize: int
    """Total number of bits in the output SDR."""
    activeBits: int
    """Count of active bits assigned to every encoding."""
    sparsity: float
    """Configured sparsity ratio used to derive the active bit count."""
    radius: float
    """Encoding radius controlling overlap between neighbour values."""
    resolution: float
    """Minimum representable separation between distinct inputs."""
    category: bool
    """Indicates whether the encoder treats inputs as categorical."""
    seed: int
    """Seed value passed to the hashing function for reproducible outputs."""

    def __init__(self, parameters: RDSEParameters):
        """
        Initializes the RandomDistributedScalarEncoder with parameters.

        Args:
            parameters (RDSEParameters): The configuration object containing encoder parameters like
                                          size, activeBits, sparsity, etc.
        """
        super().__init__()
        parameters = self.check_parameters(parameters)

        self.memberSize = parameters.size
        self.activeBits = parameters.active_bits
        self.sparsity = parameters.sparsity
        self.radius = parameters.radius
        self.resolution = parameters.resolution
        self.category = parameters.category
        self.seed = parameters.seed

    def encode(self, input_value: float, output: SDR) -> None:
        """
        Encodes a scalar value into a Sparse Distributed Representation (SDR).

        Args:
            value (float): The scalar value to encode.

        Returns:
            list[int]: A sparse distributed representation (SDR) of the input value,
                       where indices of active bits are returned as a list.
        """
        assert output.size == self.size, "Output SDR size does not match encoder size."
        if math.isnan(input_value):
            output.zero()
            return
        if self.category and (input_value != int(input_value) or input_value < 0):
            raise ValueError(
                "Input to category encoder must be an unsigned integer"
            )

        data = [0] * self.size

        index = int(input_value / self.resolution)

        for offset in range(self.activeBits):
            hash_buffer = index + offset
            bucket = mmh3.hash(struct.pack("I", hash_buffer), self.seed, signed=False)
            bucket = bucket % self.size
            data[bucket] = 1

        output.setDense(data)
        # output.setSparse(data) #we may need setDense implemented for SDR class

    # After encode we may need a check_parameters method since most of the encoders have this
    def check_parameters(self, parameters: RDSEParameters) -> RDSEParameters:
        """
        Verifies the validity and consistency of the encoder's parameter configuration.

        Raises:
            ValueError: If any of the parameters are invalid or inconsistent.
        """
        assert parameters.size > 0

        num_active_args = 0
        if parameters.active_bits > 0:
            num_active_args += 1
        if parameters.sparsity > 0:
            num_active_args += 1

        assert (
            num_active_args != 0
        ), "Missing argument, need one of: 'activeBits' or 'sparsity'."
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
params = RDSEParameters(
    size=1000,
    active_bits=0,
    sparsity=0.10,
    radius=10,
    resolution=0,
    category=False,
    seed=0,
)
encoder = RandomDistributedScalarEncoder(params)
output = SDR(dimensions=[params.size])
encoder.encode(66, output)
print(output.sparse)
