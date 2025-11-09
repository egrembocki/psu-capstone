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
        A configuration class for defining the parameters of a ScalarEncoder.

        This class holds all the necessary attributes to manage the configuration
        of a scalar encoder, which converts scalar values into sparse distributed
        representations (SDRs).

        Attributes:
            These four (4) members define the total number of bits in the output:
                size,
                radius,
                category,
                resolution.
            These are mutually exclusive and only one of them should be non-zero when
            constructing the encoder.

            Members "minimum" and "maximum" define the range of the input signal.
            These endpoints are inclusive.

            Member "clipInput" determines whether to allow input values outside the
            range [minimum, maximum].
            If true, the input will be clipped into the range [minimum, maximum].
            If false, inputs outside of the range will raise an error.

            Member "periodic" controls what happens near the edges of the input
            range.

            If true, then the minimum & maximum input values are the same and the
            first and last bits of the output SDR are adjacent.  The contiguous
            block of 1's wraps around the end back to the beginning.

            If false, then minimum & maximum input values are the endpoints of the
            input range, are not adjacent, and activity does not wrap around.

            Member "category" means that the inputs are enumerated categories.
            If true then this encoder will only encode unsigned integers, and all
            inputs will have unique / non-overlapping representations.

            Member "activeBits" is the number of true bits in the encoded output SDR.
            The output encodings will have a contiguous block of this many 1's.

            Member "sparsity" is an alternative way to specify the member "activeBits".
            Sparsity requires that the size to also be specified.
            Specify only one of: activeBits or sparsity.

            Member "size" is the total number of bits in the encoded output SDR.

            Member "radius" Two inputs separated by more than the radius have
            non-overlapping representations. Two inputs separated by less than the
            radius will in general overlap in at least some of their bits. You can
            think of this as the radius of the input.

            Member "resolution" Two inputs separated by greater than, or equal to the
            resolution are guaranteed to have different representations.
    """
    minimum: float = 0.0
    maximum: float = 0.0
    clipInput: bool = False
    periodic: bool = False
    category: bool = False
    activeBits: int = 0
    sparsity: float = 0.0
    memberSize: int = 0
    radius: float = 0.0
    resolution: float = 0.0

class ScalarEncoder(BaseEncoder):
    """
    Encodes a real number as a contiguous block of 1's.
            Description:
            The ScalarEncoder encodes a numeric (floating point) value into an array
            of bits. The output is 0's except for a contiguous block of 1's. The
            location of this contiguous block varies continuously with the input value.
    """
    def __init__(self, parameters: ScalarEncoderParameters):
        """
            Initializes a ScalarEncoder instance using the provided parameters.

            Args:
                params (ScalarEncoderParameters): An instance of `ScalarEncoderParameters`
                                                   containing the configuration for this encoder.
        """
        super().__init__()
        parameters = self.check_parameters(parameters)

        self.minimum = parameters.minimum
        self.maximum = parameters.maximum
        self.clipInput = parameters.clipInput
        self.periodic = parameters.periodic
        self.category = parameters.category
        self.activeBits = parameters.activeBits
        self.sparsity = parameters.sparsity
        self._size = parameters.memberSize
        self.radius = parameters.radius
        self.resolution = parameters.resolution

    def encode(self, input_value: float, output: SDR) -> None:
        """
            Encodes a scalar input value into a sparse distributed representation (SDR).

            This method processes a single scalar value, applies the encoding logic,
            and generates the corresponding sparse binary array.

            Args:
                inputValue (float): The scalar value to encode.

            Returns:
                list[int]: A binary sparse representation (SDR) of the input value.

            Raises:
                ValueError: If the input value is outside the valid range and clipping
                            is disabled and `clipInput` is set to False.
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
            if self.category:
                if input_value != float(int(input_value)):
                    raise ValueError("Input to category encoder must be an unsigned integer!")
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

#After encode we may need a check_parameters method since most of the encoders have this
    def check_parameters(self, parameters: ScalarEncoderParameters):
        """
                Validates the parameters of the encoder.

                Ensures that all parameters meet the expected constraints and
                raises errors for invalid configurations. This includes checking
                proper ranges, sparsity requirements, and consistency between attributes.

                Raises:
                    ValueError: If one or more parameter values are invalid.
        """

        assert parameters.minimum <= parameters.maximum
        num_active_args = sum([
            parameters.activeBits >0,
            parameters.sparsity >0.0
        ])
        assert num_active_args != 0, "Missing argument, need one of: 'activeBits', 'sparsity'."
        assert num_active_args == 1, "Specified both: 'activeBits', 'sparsity'. Specify only one of them."
        num_size_args = sum([
            parameters.memberSize > 0,
            parameters.radius > 0.0,
            bool(parameters.category),
            parameters.resolution > 0.0
        ])
        assert num_size_args != 0, "Missing argument, need one of: 'size', 'radius', 'resolution', 'category'."
        assert num_size_args == 1, "Too many arguments specified: 'size', 'radius', 'resolution', 'category'. Choose only one of them."
        if parameters.periodic:
            assert not parameters.clipInput, "Will not clip periodic inputs.  Caller must apply modulus."
        if parameters.category:
            assert not parameters.clipInput, "Incompatible arguments: category & clipInput."
            assert not parameters.periodic, "Incompatible arguments: category & periodic."
            assert parameters.minimum == float(int(parameters.minimum)), "Minimum input value of category encoder must be an unsigned integer!"
            assert parameters.maximum == float(int(parameters.maximum)), "Maximum input value of category encoder must be an unsigned integer!"

        args = parameters
        if args.category:
            args.radius = 1.0
        if args.sparsity:
            assert 0.0 <= args.sparsity <= 1.0
            assert args.memberSize > 0, "Argument 'sparsity' requires that the 'size' also be given."
            args.activeBits = round(args.memberSize * args.sparsity)
            assert args.activeBits > 0, "sparsity and size must be given so that sparsity * size > 0!"
        if args.periodic:
            extentWidth = args.maximum - args.minimum
        else:
            maxInclusive = math.nextafter(args.maximum, math.inf)
            extentWidth = maxInclusive - args.minimum
        if args.memberSize > 0:
            if args.periodic:
                args.resolution = extentWidth / args.memberSize
            else:
                nBuckets = args.memberSize - (args.activeBits -1)
                args.resolution = extentWidth / (nBuckets-1)
        else:
            if args.radius > 0.0:
                args.resolution = args.radius / args.activeBits

            neededBands = math.ceil(extentWidth / args.resolution)
            if args.periodic:
                args.memberSize = neededBands
            else:
                args.memberSize = neededBands + (args.activeBits - 1)

        # Sanity check the parameters.
        assert args.memberSize > 0
        assert args.activeBits > 0
        assert args.activeBits < args.memberSize

        args.radius = args.activeBits * args.resolution
        assert args.radius > 0

        args.sparsity = args.activeBits / float(args.memberSize)
        assert args.sparsity > 0
        return args


#Tests
params = ScalarEncoderParameters(
    minimum = 0,
    maximum = 100,
    clipInput = False,
    periodic = False,
    category = False,
    activeBits = 21,
    sparsity = 0,
    memberSize = 500,
    radius = 0,
    resolution = 0
)
'''encoder = ScalarEncoder(params ,dimensions=[100])
sdr = SDR(dimensions=[10, 10])
print(sdr.size)
sdr.setSparse([0,5,22,99])
print(sdr.getSparse())
sdr.zero()
print(sdr.getSparse())

encoder2 = ScalarEncoder(params ,dimensions=[100])
print(encoder2.size)
print(encoder2.dimensions)'''

encoder3 = ScalarEncoder(params ,dimensions=[params.memberSize])
output = SDR(dimensions=[params.memberSize])
encoder3.encode(7.3, output)
print(output.getSparse())