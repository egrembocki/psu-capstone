"""Scalar Encoder Module implementation. From NuPic Numenta Cpp ported to python
/**
 * These four (4) members define the total number of bits in the output:
 *      size,
 *      radius,
 *      category,
 *      resolution.
 *
 * These are mutually exclusive and only one of them should be non-zero when
 * constructing the encoder. -- Need to refactor ScalarEncoder to take in only params
 */

"""

import copy
import math
from dataclasses import dataclass
from typing import List

import numpy as np

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.sdr import SDR


@dataclass
class ScalarEncoderParameters:

    minimum: float = 0.0
    """Minimum input value. Defaults to 0.0."""

    maximum: float = 10.0
    """Min and Max
     * Members "minimum" and "maximum" define the range of the input signal.
     * These endpoints are inclusive. Defaults to 10.0.

     */"""

    # xor clip_input, periodic
    clip_input: bool = True
    """Whether to clip inputs outside the min/max range. Defaults to True.
       /**
     * Member "clipInput" determines whether to allow input values outside the
     * range [minimum, maximum].
     * If true, the input will be clipped into the range [minimum, maximum].
     * If false, inputs outside of the range will raise an error.
     */"""

    periodic: bool = False
    """Whether the encoder is periodic (circular) or not. Defaults to False.
    /**
     * Member "periodic" controls what happens near the edges of the input
     * range.
     *
     * If true, then the minimum & maximum input values are the same and the
     * first and last bits of the output SDR are adjacent.  The contiguous
     * block of 1's wraps around the end back to the beginning.
     *
     * If false, then minimum & maximum input values are the endpoints of the
     * input range, are not adjacent, and activity does not wrap around.
     */
    """
    size: int = 1024
    """Total number of bits in the output SDR. Defaults to 1024.
     /**
     * Member "size" is the total number of bits in the encoded output SDR.
     */
    """

    # xor active_bits, sparsity
    active_bits: int = 0  # xor with sparsity
    """Number of active bits in the output SDR. Defaults to 0."""

    sparsity: float = 0.02  # xor with active_bits
    """Number of active bits or sparsity level. Defaults to 0.02.
     /**
     * Member "activeBits" is the number of true bits in the encoded output SDR.
     * The output encodings will have a contiguous block of this many 1's.
     */

      /**
     * Member "sparsity" is an alternative way to specify the member "activeBits".
     * Sparsity requires that the size to also be specified.
     * Specify only one of: activeBits or sparsity.
     */

    """

    # xor radius, category, resolution
    radius: float = 0.0
    """Approximate input range (width) covered by the active bits. Defaults to 0.0.
    /**
     * Member "radius" Two inputs separated by more than the radius have
     * non-overlapping representations. Two inputs separated by less than the
     * radius will in general overlap in at least some of their bits. You can
     * think of this as the radius of the input.
     */
    """
    resolution: float = 1.0
    """The smallest difference between two inputs that produces different outputs.
         Defaults to 1.0.
      /**
     * Member "resolution" Two inputs separated by greater than, or equal to the
     * resolution are guaranteed to have different representations.
     */"""

    category: bool = False
    """Whether the encoder is a category encoder. Defaults to False.
    /**
     * Member "category" means that the inputs are enumerated categories.
     * If true then this encoder will only encode unsigned integers, and all
     * inputs will have unique / non-overlapping representations.
     */
    """


class ScalarEncoder(BaseEncoder):
    """
    /**
     * Encodes a real number as a contiguous block of 1's.
     *
     * Description:
     * The ScalarEncoder encodes a numeric (floating point) value into an array
     * of bits. The output is 0's except for a contiguous block of 1's. The
     * location of this contiguous block varies continuously with the input value.
     *
     * To inspect this run:
     * $ python -m htm.examples.encoders.scalar_encoder --help
     */"""

    def __init__(self, parameters: ScalarEncoderParameters, dimensions: List[int] | None = None):
        self._parameters = copy.deepcopy(parameters)
        self._parameters = self.check_parameters(self._parameters)

        self._minimum = self._parameters.minimum
        self._maximum = self._parameters.maximum
        self._clip_input = self._parameters.clip_input
        self._periodic = self._parameters.periodic
        self._category = self._parameters.category
        self._active_bits = self._parameters.active_bits
        self._sparsity = self._parameters.sparsity
        self._size = self._parameters.size
        """Size of the ScalarEncoder"""
        self._radius = self._parameters.radius
        self._resolution = self._parameters.resolution

        super().__init__(dimensions, self._size)

    """
        Encodes an input value into an SDR with a block of 1's.

        Description:
        The encode method is responsible for transforming the supplied SDR data structure into
        an SDR that has the encoding of the input value.
    """

    def encode(self, input_value: float, output_sdr: SDR) -> None:
        """Encodes the input value into the output SDR."""

        assert output_sdr.size == self.size, "Output SDR size does not match encoder size."

        if math.isnan(input_value):
            output_sdr.zero()
            return

        elif self._clip_input:
            if self._periodic:

                input_value = input_value % self._maximum
            else:
                input_value = max(input_value, self._minimum)
                input_value = min(input_value, self._maximum)
        else:
            if self._category and input_value != float(int(input_value)):
                raise ValueError("Input to category encoder must be an unsigned integer!")
            if not (self._minimum <= input_value <= self._maximum):
                raise ValueError(
                    f"Input must be within range [{self._minimum}, {self._maximum}]! "
                    f"Received {input_value}"
                )

        start = int(round((input_value - self._minimum) / self._resolution))

        """Handle edge case where start + active_bits exceeds output size.
          // The endpoints of the input range are inclusive, which means that the
          // maximum value may round up to an index which is outside of the SDR. Correct
          // this by pushing the endpoint (and everything which rounds to it) onto the
          // last bit in the SDR.
        """
        if not self._periodic:
            start = min(start, output_sdr.size - self._active_bits)

        sparse = output_sdr.get_sparse()
        sparse[:] = range(start, start + self._active_bits)

        if self._periodic:
            for i, bit in enumerate(sparse):
                if bit >= output_sdr.size:
                    sparse[i] = bit - output_sdr.size
            sparse.sort()

        output_sdr.set_sparse(sparse)

    # After encode we may need a check_parameters method since most of the encoders have this

    def check_parameters(self, parameters: ScalarEncoderParameters) -> ScalarEncoderParameters:
        """
        Check parameters method is used to verify that the correct parameters were entered
        and reject the user when they are not.

        Description: This changes and transforms the input that the user has with the parameters
        dataclass. There are many aspects such as the active bit and sparsity being mutually exclusive
        and the size, radius, resolution, and category also being muturally exclusive with each other.
        The user will have an assert that rejects when these are violated.
        """
        # Ensure size is valid
        assert parameters.size > 0

        # Ensure min is less than max
        assert parameters.minimum <= parameters.maximum

        # X-OR toggle for clip_input and periodic
        num_clip_args = 0

        if parameters.periodic:
            parameters.clip_input = False
            num_clip_args += 1
        elif parameters.clip_input:  # defualt case
            parameters.periodic = False
            num_clip_args += 1
        else:
            print("Neither clip_input nor periodic set, defaulting to clip_input = True")
            parameters.clip_input = True
            parameters.periodic = False
            num_clip_args += 1

        assert (
            parameters.clip_input != parameters.periodic
        ), "Incompatible arguments: 'clip_input' and 'periodic'."
        assert num_clip_args == 1, "Exactly one of 'clip_input' or 'periodic' must be set."

        # X-OR toggle for active_bits and sparsity
        num_active_args = 0

        if parameters.active_bits > 0:  # given active_bits -- use it
            parameters.sparsity = 0.0  # reset sparsity
            num_active_args += 1
        elif parameters.sparsity > 0.0:  # given sparsity -- use it
            parameters.active_bits = 0  # reset active_bits
            num_active_args += 1
        else:
            print("Neither active_bits nor sparsity set, defaulting to sparsity = 0.02")
            parameters.active_bits = 0
            parameters.sparsity = 0.02
            num_active_args += 1

        # Assert X-OR toggle for active_bits and sparsity
        assert (parameters.active_bits == 0) != (
            parameters.sparsity == 0
        ), "Exactly one of 'active_bits' or 'sparsity' must be set."
        assert num_active_args == 1, "Exactly one of 'active_bits' or 'sparsity' must be set."

        # X-OR Radius / Category / Resolution
        num_resolution_args = 0

        if parameters.radius > 0.0:
            parameters.category = False
            parameters.resolution = 0.0
            num_resolution_args += 1
        elif parameters.category:
            parameters.radius = 0.0
            parameters.resolution = 0.0
            num_resolution_args += 1
        elif parameters.resolution > 0.0:
            parameters.category = False
            parameters.radius = 0.0
            num_resolution_args += 1
        else:
            print(
                "None of 'radius', 'resolution', or 'category' set, defaulting to resolution = 1.0"
            )
            parameters.radius = 0.0
            parameters.category = False
            parameters.resolution = 1.0
            num_resolution_args += 1

        # Assert X-OR toggle for radius, resolution, category
        assert (parameters.radius > 0.0) + (parameters.resolution > 0.0) + (
            parameters.category
        ) == 1, "Exactly one of 'radius', 'resolution', or 'category' must be set."
        assert (
            num_resolution_args != 0
        ), "Missing argument, need one of: 'radius', 'resolution', 'category'."
        assert (
            num_resolution_args == 1
        ), "Too many arguments, choose only one of: 'radius', 'resolution', 'category'."

        # Category specific checks
        if parameters.category:
            assert not parameters.clip_input, "Incompatible arguments: category & clip_input."
            assert not parameters.periodic, "Incompatible arguments: category & periodic."
            assert parameters.minimum == float(
                int(parameters.minimum)
            ), "Minimum input value of category encoder must be an integer!"
            assert parameters.maximum == float(
                int(parameters.maximum)
            ), "Maximum input value of category encoder must be an integer!"

        # Copy parameters to args for easier reference -- adjust final parameters
        args = parameters

        if args.category:
            args.radius = 1.0

        # Set active_bits if sparsity is provided
        if args.sparsity > 0.0 and args.active_bits == 0:
            assert 0.0 <= args.sparsity <= 1.0
            assert args.size > 0, "Argument 'sparsity' requires that the 'size' also be given."
            args.active_bits = round(args.size * args.sparsity)
            assert (
                args.active_bits > 0
            ), "sparsity and size must be given so that sparsity * size > 0!"

        # Determine resolution & size
        extent_width = (
            args.maximum - args.minimum
            if args.periodic
            else np.nextafter(args.maximum, float("inf")) - args.minimum
        )

        # Set default for radius/resolution/category if all are unset
        if args.size > 0:

            if args.periodic:
                args.resolution = extent_width / args.size

            else:
                n_buckets = args.size - (args.active_bits - 1)
                args.resolution = extent_width / (n_buckets - 1)

        else:

            if args.radius > 0.0:
                args.resolution = args.radius / args.active_bits

            needed_bands = math.ceil(extent_width / args.resolution)

            args.size = needed_bands if args.periodic else needed_bands + (args.active_bits - 1)

        # Sanity check the parameters.
        assert args.size > 0
        assert args.active_bits > 0
        assert args.active_bits < args.size

        # compute radius
        args.radius = args.active_bits * args.resolution
        assert args.radius > 0

        # compute sparsity
        args.sparsity = args.active_bits / float(args.size - 1)
        assert args.sparsity > 0

        return args


# Smoke Test
if __name__ == "__main__":
    p = ScalarEncoderParameters(
        minimum=10.0,
        maximum=20.0,
        sparsity=0.02,
        size=480,
    )
    default = ScalarEncoderParameters()

    enc_def = ScalarEncoder(default)

    encoder = ScalarEncoder(p)
    output = SDR([encoder.size])
    dsdr = SDR([enc_def.size])
    print(dsdr.size, enc_def.size)
    encoder.encode(5.0, output)
    enc_def.encode(8.0, dsdr)
    print(output.get_sparse())
    print(dsdr.get_sparse())
