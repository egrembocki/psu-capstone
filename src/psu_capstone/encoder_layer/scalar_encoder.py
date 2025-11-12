import math
from dataclasses import dataclass
from typing import List

from psu_capstone.encoder_layer.sdr import SDR
from psu_capstone.encoder_layer.base_encoder import BaseEncoder


@dataclass
class ScalarEncoderParameters:
    minimum: float
    maximum: float
    clip_input: bool
    periodic: bool
    category: bool
    active_bits: int
    sparsity: float
    member_size: int
    radius: float
    resolution: float


class ScalarEncoder(BaseEncoder):

    def __init__(self, parameters: ScalarEncoderParameters, dimensions: List[int]):
        super().__init__(dimensions)
        parameters = self.check_parameters(parameters)

        self.minimum = parameters.minimum
        self.maximum = parameters.maximum
        self.clip_input = parameters.clip_input
        self.periodic = parameters.periodic
        self.category = parameters.category
        self.active_bits = parameters.active_bits
        self.sparsity = parameters.sparsity
        self._size = parameters.member_size
        self.radius = parameters.radius
        self.resolution = parameters.resolution

    def encode(self, input_value: float, output: SDR) -> None:
        assert output.size == self.size, "Output SDR size does not match encoder size."

        if math.isnan(input_value):
            output.zero()
            return

        if self.clip_input:
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
            start = min(start, output.size - self.active_bits)

        sparse = list(range(start, start + self.active_bits))

        if self.periodic:
            sparse = [bit % output.size for bit in sparse]
            sparse.sort()

        output.set_sparse(sparse)

    # After encode we may need a check_parameters method since most of the encoders have this
    def check_parameters(self, parameters: ScalarEncoderParameters):

        assert parameters.minimum <= parameters.maximum
        num_active_args = sum([parameters.active_bits > 0, parameters.sparsity > 0.0])
        assert num_active_args != 0, "Missing argument, need one of: 'active_bits', 'sparsity'."
        assert (
            num_active_args == 1
        ), "Specified both: 'active_bits', 'sparsity'. Specify only one of them."
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
        ), "Too many arguments specified: 'size', 'radius', 'resolution', 'category'. Choose only one of them."
        if parameters.periodic:
            assert (
                not parameters.clip_input
            ), "Will not clip periodic inputs.  Caller must apply modulus."
        if parameters.category:
            assert not parameters.clip_input, "Incompatible arguments: category & clip_input."
            assert not parameters.periodic, "Incompatible arguments: category & periodic."
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
