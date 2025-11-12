"""Scalar Encoder implementation for encoding scalar values into Sparse Distributed Representations (SDRs)."""


from typing import List, Union
from typing_extensions import Self
from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.sdr import SDR
from dataclasses import dataclass
import math

@dataclass
class ScalarEncoderParameters:
    """Parameters for the Scalar Encoder."""
    
    minimum: float
    """Minimum value for the input."""
    maximum: float
    """Maximum value for the input."""
    clip_input: bool
    """Whether to clip input values to the min/max range."""
    periodic: bool
    """Whether the encoder is periodic."""
    active_bits: int
    """Number of active bits in the output SDR."""
    sparsity: float
    """Sparsity of the output SDR."""
    member_size: int
    """Size of the output SDR."""
    radius: float
    """Radius of the encoder."""
    category: bool
    """Whether the encoder is categorical."""
    resolution: float
    """Resolution of the encoder."""

    #active_bits_or_sparsity: Union[int, float] = 0
    #member_size_or_radius_or_category_or_resolution: Union[int, float, bool, double] = 0

    """perfect use case for a union : either int or float : active_bits xor sparsity
    perfect use case for a union : int or float or bool or double : member_size xor radius xor category xor resolution"""


class ScalarEncoder(BaseEncoder):

    def __new__(cls, parameters: ScalarEncoderParameters, dimensions: List[int]) -> Self:
        """Create a new instance of ScalarEncoder."""
        cls.minimum = 0.0
        cls.maximum = 100.0
        cls.clipInput = True
        cls.periodic = False

        # Union implementation pending
        cls.active_bits = 5
        cls.sparsity = 0.0

        # Union implementation pending
        cls.member_size = 10
        cls.radius = 0.0
        cls.category = False
        cls.resolution = 0.0

        return super().__new__(cls)

    def __init__(self, parameters: ScalarEncoderParameters, dimensions: List[int]):
        """Initialize the Scalar Encoder with given parameters and dimensions."""
        super().__init__(dimensions)

        parameters = self.check_parameters(parameters)
        """Scalar Encoder parameters after validation."""

        self.minimum = parameters.minimum
        self.maximum = parameters.maximum
        self.clipInput = parameters.clip_input
        self.periodic = parameters.periodic
        self.category = parameters.category
        self.activeBits = parameters.active_bits
        self.sparsity = parameters.sparsity
        self.member_size = parameters.member_size
        self.radius = parameters.radius
        self.resolution = parameters.resolution

    def encode(self, input_value: float, output: SDR) -> bool:
        assert output._size == self.size, "Output SDR size does not match encoder size."

        self.__sdr = output

        if math.isnan(input_value):
            output.zero()
            return False

        if self.clipInput:
            if self.periodic:
                raise NotImplementedError("Periodic input clipping not implemented.")
            else:
                input_value = max(input_value, self.minimum)
                input_value = min(input_value, self.maximum)
        else:
            if self.category and input_value != float(int(input_value)):
                raise ValueError("Input to category encoder must be an unsigned integer!")
            if not (self.minimum <= input_value <= self.maximum):
                raise ValueError(
                    f"Input must be within range [{self.minimum}, {self.maximum}]! "
                    f"Received {input_value}"
                )

        start = int(round((input_value - self.minimum) / self.resolution))

        if not self.periodic:
            start = min(start, output._size - self.activeBits)

        sparse = list(range(start, start + self.activeBits))

        if self.periodic:
            sparse = [bit % output._size for bit in sparse]
            sparse.sort()

        output.set_sparse(sparse)
        
        self.__sdr = output

        return self.__sdr == output







#After encode we may need a check_parameters method since most of the encoders have this
    def check_parameters(self, parameters: ScalarEncoderParameters) -> ScalarEncoderParameters:
        """Validate and compute derived parameters for the Scalar Encoder. This may change if we can get the Union to work properly."""
        assert parameters.minimum <= parameters.maximum
        num_active_args = sum([
            parameters.active_bits > 0,
            parameters.sparsity > 0.0
        ])
        assert num_active_args != 0, "Missing argument:: need 'activeBits' or 'sparsity'."
        assert num_active_args == 1, (
            "Specified both:: 'activeBits' and 'sparsity'. "
            "Specify only one of them."
        )
        num_size_args = sum([
            parameters.member_size > 0,
            parameters.radius > 0.0,
            bool(parameters.category),
            parameters.resolution > 0.0
        ])
        assert num_size_args != 0, (
            "Missing argument, need one of: 'size', 'radius', 'resolution', 'category'."
        )
        assert num_size_args == 1, "Too many arguments specified: 'size', 'radius', 'resolution', 'category'. Choose only one of them."
        if parameters.periodic:
            assert not parameters.clip_input, "Will not clip periodic inputs.  Caller must apply modulus."
        if parameters.category:
            assert not parameters.clip_input, "Incompatible arguments: category & clipInput."
            assert not parameters.periodic, "Incompatible arguments: category & periodic."
            assert parameters.minimum == float(int(parameters.minimum)), "Minimum input value of category encoder must be an unsigned integer!"
            assert parameters.maximum == float(int(parameters.maximum)), "Maximum input value of category encoder must be an unsigned integer!"

        args = parameters
        if args.category:
            args.radius = 1.0
        if args.sparsity:
            assert 0.0 <= args.sparsity <= 1.0
            assert args.member_size > 0, "Argument 'sparsity' requires that the 'size' also be given."
            args.active_bits = round(args.member_size * args.sparsity)
            assert args.active_bits > 0, "sparsity and size must be given so that sparsity * size > 0!"
        if args.periodic:
            extent_width = args.maximum - args.minimum
        else:
            max_inclusive = math.nextafter(args.maximum, math.inf)
            extent_width = max_inclusive - args.minimum
        if args.member_size > 0:
            if args.periodic:
                args.resolution = (extent_width / args.member_size)
            else:
                n_buckets = args.member_size - (args.active_bits -1)
                args.resolution = (extent_width / (n_buckets-1))
        else:
            if args.radius > 0.0:
                args.resolution = (args.radius / args.active_bits)

            needed_bands = math.ceil(extent_width / args.resolution)
            if args.periodic:
                args.member_size = needed_bands
            else:
                args.member_size = needed_bands + (args.active_bits - 1)

        # Sanity check the parameters.
        assert args.member_size > 0
        assert args.active_bits > 0
        assert args.active_bits < args.member_size

        args.radius = args.active_bits * (args.resolution)
        assert args.radius > 0

        args.sparsity = args.active_bits / float(args.member_size)
        assert args.sparsity > 0
        
        return args
