"""Category Encoder implementation"""

import copy
import math
import random
import struct
from dataclasses import dataclass
from typing import List

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.sdr import SDR


@dataclass
class CategoryParameters:
    w: int
    category_list: List[str]
    name: str = "category"
    verbosity: int = 0
    forced: bool = False


class CategoryEncoder(BaseEncoder):
    """
    Encodes a list of discrete categories (described by strings), that aren't
    related to each other, so we never emit a mixture of categories.

    The value of zero is reserved for "unknown category"

    Internally we use a :class:`.ScalarEncoder` with a radius of 1, but since we
    only encode integers, we never get mixture outputs.

    The :class:`.SDRCategoryEncoder` uses a different method to encode categories.

    :param categoryList: list of discrete string categories
    :param forced: if True, skip checks for parameters' settings; see
                    :class:`.ScalarEncoder` for details. (default False)
    """

    def __init__(self, parameters: CategoryParameters, dimensions: List[int] = None):
        super().__init__(dimensions)

        self.parameters = copy.deepcopy(parameters)

    def encode(self, input_value: str, output_sdr):
        return super().encode(input_value, output_sdr)

    def check_parameters(self, parameters: CategoryParameters):

        args = parameters
        return args
