"""Scalar Encoder implementation for encoding scalar values into Sparse Distributed Representations (SDRs)."""


from typing import List
from typing_extensions import Self
from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.sdr import SDR
from dataclasses import dataclass
import math

@dataclass
class ScalarEncoderParameters:
    """Parameters for the Scalar Encoder."""
    
    minimum: float
    maximum: float
    clipInput: bool
    periodic: bool
    category: bool
    activeBits: int
    sparsity: float
    memberSize: int
    radius: float
    resolution: float


class ScalarEncoder(BaseEncoder):

    def __new__(cls, parameters: ScalarEncoderParameters, dimensions: List[int]) -> Self:
        """Create a new instance of ScalarEncoder."""
        
        return super().__new__(cls)

    def __init__(self, parameters: ScalarEncoderParameters, dimensions: List[int]):
        super().__init__(dimensions)
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
        assert output.__size == self.size, "Output SDR size does not match encoder size."

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
            start = min(start, output.__size - self.activeBits)

        sparse = list(range(start, start + self.activeBits))

        if self.periodic:
            sparse = [bit % output.__size for bit in sparse]
            sparse.sort()

        output.set_sparse(sparse)







#After encode we may need a check_parameters method since most of the encoders have this
    def check_parameters(self, parameters: ScalarEncoderParameters) -> ScalarEncoderParameters:
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
