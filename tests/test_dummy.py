"""
File for making sure testing actions work and pre-commit stuff, will delete
"""

from psu_capstone.utils import add_numbers, greet


def test_dummy() -> None:
    assert 1 == 1


def test_add_numbers() -> None:

    assert add_numbers(2, 3) == 5
    assert add_numbers(-7, 7) == 0
    assert add_numbers(3, 10) == 13


def test_greet() -> None:
    assert greet("World") == "Hello, World!"
