"""Test suite for the Category Encoder"""

import pytest

from psu_capstone.encoder_layer.category_encoder import CategoryEncoder, CategoryParameters
from psu_capstone.encoder_layer.sdr import SDR


@pytest.fixture
def category_instance():
    """Fixture to create a Category encoder instance for tests"""


def test_category_initialization():
    """
    This tests to make sure the Category Encoder can succesfully be created.
    Note: there is an optional dimensions parameter not being used here.
    """
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, category_list=categories, forced=True)
    e = CategoryEncoder(parameters=parameters)
    """Checking if the instance is correct."""
    assert isinstance(e, CategoryEncoder)


def test_encode_us():
    """
    This encodes the category "US" into an SDR of 1x12. That bit number is determined from
    3 categories and 1 unknown category. This is w or width of 3 times 4 which is 12 long.
    """
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, categoryList=categories, forced=True)
    e = CategoryEncoder(parameters=parameters)
    a = SDR([1, 12])
    e.encode("US", a)
    """This makes sure our encoding is accurate and matches a known SDR outcome."""
    assert a == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]


def test_unknown_category():
    """
    This encodes an unknown category. Here we use "NA" which as you can see is not one of
    the categories specified.
    """
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, categoryList=categories, forced=True)
    e = CategoryEncoder(parameters=parameters)
    a = SDR([1, 12])
    e.encode("NA", a)
    """This makes sure our encoding is accurate and matches a known SDR outcome."""
    assert a == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_encode_es():
    """
    This is almost idential to the "US" encoding, I am just deomonstrating that the encoding
    shows different active bits for different categories.
    """
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, categoryList=categories, forced=True)
    e = CategoryEncoder(parameters=parameters)
    a = SDR([1, 12])
    e.encode("ES", a)
    """This makes sure our encoding is accurate and matches a known SDR outcome."""
    assert a == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]


def test_with_width_one():
    """This test is used to show how SDR outputs look with a single w or width."""
    categories = ["cat1", "cat2", "cat3", "cat4", "cat5"]
    """Note: I think since width is 1, each category is 1 bit and there is the first bit that is the unknown category."""
    expected = [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
    parameters = CategoryParameters(w=1, categoryList=categories, forced=True)
    e = CategoryEncoder(parameters=parameters)
    i = 0
    """The respective category should equal their index of expected results."""
    for cat in categories:
        a = SDR([1, 6])
        e.encode(cat, a)
        assert a == expected[i]
        i = i + 1
