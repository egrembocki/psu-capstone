"""Scalar encoder utilities for converting scalars into Sparse Distributed Representations."""

from typing import List
from SDR_Encoder_Temp.BaseEncoder import BaseEncoder
from SDR import SDR
from dataclasses import dataclass
import math
import numpy as np

"""
Define the ScalarEncoder class
"""


@dataclass
class ScalarEncoderParameters:
    """
    Configuration class for defining the parameters of a :class:`ScalarEncoder`.

    This class holds all the necessary attributes to manage the configuration of a scalar
    encoder, which converts scalar values into sparse distributed representations (SDRs).

    These four (4) members define the total number of bits in the output: size, radius,
    category, resolution. These are mutually exclusive and only one of them should be
    non-zero when constructing the encoder.

    Members ``minimum`` and ``maximum`` define the range of the input signal. These
    endpoints are inclusive.

    Member ``clipInput`` determines whether to allow input values outside the range
    [minimum, maximum]. If true, the input will be clipped into the range [minimum,
    maximum]. If false, inputs outside of the range will raise an error.

    Member ``periodic`` controls what happens near the edges of the input range. If true,
    then the minimum & maximum input values are the same and the first and last bits of
    the output SDR are adjacent. The contiguous block of 1's wraps around the end back to
    the beginning. If false, then minimum & maximum input values are the endpoints of the
    input range, are not adjacent, and activity does not wrap around.

    Member ``category`` means that the inputs are enumerated categories. If true then this
    encoder will only encode unsigned integers, and all inputs will have unique /
    non-overlapping representations.

    Member ``activeBits`` is the number of true bits in the encoded output SDR. The output
    encodings will have a contiguous block of this many 1's.

    Member ``sparsity`` is an alternative way to specify the member ``activeBits``.
    Sparsity requires that the size to also be specified. Specify only one of:
    ``activeBits`` or ``sparsity``.

    Member ``size`` is the total number of bits in the encoded output SDR.

    Member ``radius``: Two inputs separated by more than the radius have non-overlapping
    representations. Two inputs separated by less than the radius will in general overlap
    in at least some of their bits. You can think of this as the radius of the input.

    Member ``resolution``: Two inputs separated by greater than, or equal to the
    resolution are guaranteed to have different representations.

    Attributes:
        minimum (float): Inclusive lower bound on the scalar input domain.
        maximum (float): Inclusive upper bound on the scalar input domain.
        clipInput (bool): When True, clip values outside the domain instead of erroring.
        periodic (bool): Treat the domain as circular and wrap encodings around.
        category (bool): Encode unsigned integer categories with non-overlapping bits.
        activeBits (int): Number of simultaneously active bits in the output SDR.
        sparsity (float): Fractional sparsity used to derive ``activeBits`` when size is set.
        memberSize (int): Total number of bins in the encoder output.
        radius (float): Separation guaranteeing non-overlapping encodings.
        resolution (float): Minimum difference that yields distinct encodings.
    """

    minimum: float = 0.0
    maximum: float = 0.0
    clip_input: bool = False
    periodic: bool = False
    category: bool = False
    active_bits: int = 0
    sparsity: float = 0.0
    member_size: int = 0
    radius: float = 0.0
    resolution: float = 0.0


class ScalarEncoder(BaseEncoder):
    """
    Encodes a real number as a contiguous block of 1's within an SDR.

    The ScalarEncoder converts a numeric (floating point) value into an array of bits.
    The output is 0's except for a contiguous block of 1's whose position varies
    continuously with the input value.
    """

    def __init__(self, parameters: ScalarEncoderParameters):
        """
        Initialize a :class:`ScalarEncoder` using the provided parameters.

        Args:
            parameters (ScalarEncoderParameters): Configuration object containing
                range limits, sparsity, and geometric options for the encoder.
        """
        super().__init__()
        parameters = self.check_parameters(parameters)

        self.minimum = parameters.minimum
        self.maximum = parameters.maximum
        self.clipInput = parameters.clip_input
        self.periodic = parameters.periodic
        self.category = parameters.category
        self.activeBits = parameters.active_bits
        self.sparsity = parameters.sparsity
        self._size = parameters.member_size
        self.radius = parameters.radius
        self.resolution = parameters.resolution

    def encode(self, input_value: float, output: SDR) -> None:
        """
        Encode a scalar input value into a sparse distributed representation (SDR).

        This method processes a single scalar value, applies the encoding logic, and
        populates ``output`` with the corresponding sparse binary array.

        Args:
            input_value (float): The scalar value to encode.
            output (SDR): Destination SDR that receives the activated indices.
            TODO The c++ version is passing output by reference. Are we accurately modeling that?

        Returns:
            None

        Raises:
            ValueError: If the input value is outside the valid range and clipping is
                disabled while ``clipInput`` is False.
        """
        assert output.size == self.size, "Output SDR size does not match encoder size."

        if math.isnan(input_value):
            output.zero()
            return

        if self.clipInput:
            if self.periodic:
                raise NotImplementedError("Periodic input clipping not implemented.")
            else:
                input_value = max(input_value, self.minimum)
                input_value = min(input_value, self.maximum)
        else:
            if self.category and input_value != float(int(input_value)):
                raise ValueError(
                    "Input to category encoder must be an unsigned integer!"
                )
            if not (self.minimum <= input_value <= self.maximum):
                raise ValueError(
                    f"Input must be within range [{self.minimum}, {self.maximum}]! "
                    f"Received {input_value}"
                )

        start = int(round((input_value - self.minimum) / self.resolution))

        if not self.periodic:
            start = min(start, output.size - self.activeBits)

        sparse = list(range(start, start + self.activeBits))

        if self.periodic:
            sparse = [bit % output.size for bit in sparse]
            sparse.sort()

        output.setSparse(sparse)

    # After encode we may need a check_parameters method since most of the encoders have this
    def check_parameters(self, parameters: ScalarEncoderParameters):
        """
        TODO CCG is too High - SonarQube is flagging this method as too complex.

        Validate encoder parameters and derive dependent configuration values.

        Ensures that all parameters meet the expected constraints and raises errors for
        invalid configurations. This includes checking proper ranges, sparsity requirements,
        and consistency between attributes.

        Args:
            parameters (ScalarEncoderParameters): Candidate encoder configuration.

        Returns:
            ScalarEncoderParameters: Normalized parameter set ready for encoder use.

        Raises:
            AssertionError: If mutually exclusive options are combined or invalid ranges
                are supplied.
            ValueError: Propagated from downstream checks when invariants are violated.
        """

        assert parameters.minimum <= parameters.maximum
        num_active_args = sum([parameters.active_bits > 0, parameters.sparsity > 0.0])
        assert (
            num_active_args != 0
        ), "Missing argument, need one of: 'activeBits', 'sparsity'."
        assert (
            num_active_args == 1
        ), "Specified both: 'activeBits', 'sparsity'. Specify only one of them."
        num_size_args = sum(
            [
                parameters.member_size > 0,
                parameters.radius > 0.0,
                bool(parameters.category),
                parameters.resolution > 0.0,
            ]
        )
        assert (
            num_size_args != 0
        ), "Missing argument, need one of: 'size', 'radius', 'resolution', 'category'."
        assert (
            num_size_args == 1
        ), "Too many arguments specified: 'size', 'radius', 'resolution', 'category'. Choose only"
        " one of them."
        if parameters.periodic:
            assert (
                not parameters.clip_input
            ), "Will not clip periodic inputs.  Caller must apply modulus."
        if parameters.category:
            assert (
                not parameters.clip_input
            ), "Incompatible arguments: category & clipInput."
            assert (
                not parameters.periodic
            ), "Incompatible arguments: category & periodic."
            assert parameters.minimum == float(
                int(parameters.minimum)
            ), "Minimum input value of category encoder must be an unsigned integer!"
            assert parameters.maximum == float(
                int(parameters.maximum)
            ), "Maximum input value of category encoder must be an unsigned integer!"

        args = parameters
        if args.category:
            args.radius = 1.0
        if args.sparsity:
            assert 0.0 <= args.sparsity <= 1.0
            assert (
                args.member_size > 0
            ), "Argument 'sparsity' requires that the 'size' also be given."
            args.active_bits = round(args.member_size * args.sparsity)
            assert (
                args.active_bits > 0
            ), "sparsity and size must be given so that sparsity * size > 0!"
        if args.periodic:
            extent_width = args.maximum - args.minimum
        else:
            max_inclusive = math.nextafter(args.maximum, math.inf)
            extent_width = max_inclusive - args.minimum
        if args.member_size > 0:
            if args.periodic:
                args.resolution = extent_width / args.member_size
            else:
                n_buckets = args.member_size - (args.active_bits - 1)
                args.resolution = extent_width / (n_buckets - 1)
        else:
            if args.radius > 0.0:
                args.resolution = args.radius / args.active_bits

            needed_bands = math.ceil(extent_width / args.resolution)
            if args.periodic:
                args.member_size = needed_bands
            else:
                args.member_size = needed_bands + (args.active_bits - 1)
        # Sanity check the parameters.
        assert args.member_size > 0
        assert args.active_bits > 0
        assert args.active_bits < args.member_size

        args.radius = args.active_bits * args.resolution
        assert args.radius > 0

        args.sparsity = args.active_bits / float(args.member_size)
        assert args.sparsity > 0
        return args


# Tests
params = ScalarEncoderParameters(
    minimum=0,
    maximum=100,
    clip_input=False,
    periodic=False,
    category=False,
    active_bits=21,
    sparsity=0,
    member_size=500,
    radius=0,
    resolution=0,
)
"""encoder = ScalarEncoder(params ,dimensions=[100])
sdr = SDR(dimensions=[10, 10])
print(sdr.size)
sdr.setSparse([0,5,22,99])
print(sdr.getSparse())
sdr.zero()
print(sdr.getSparse())

encoder2 = ScalarEncoder(params ,dimensions=[100])
print(encoder2.size)
print(encoder2.dimensions)"""

encoder3 = ScalarEncoder(params)
output = SDR(dimensions=[params.member_size])
encoder3.encode(7.3, output)
print(output.getSparse())
