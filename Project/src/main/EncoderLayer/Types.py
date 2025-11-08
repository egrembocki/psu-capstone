"""
Types.py

Python equivalents of basic HTM core types from the C++ NTA_Types.hpp.
This is a structural translation, not a drop-in FFI layer.
"""

from ctypes import (
    c_int8,
    c_int16,
    c_uint16,
    c_int32,
    c_uint32,
    c_int64,
    c_uint64,
    c_float,
    c_double,
    c_void_p,
    c_size_t,
)
from enum import IntEnum


# ----------------------------------------------------------------------
# Configuration flags (Python stand-ins for the C++ preprocessor macros)
# ----------------------------------------------------------------------

# Set these True/False to match your build configuration
NTA_DOUBLE_PRECISION = False
NTA_BIG_INTEGER = False


def un_used(*_args) -> None:
    """Marker for intentionally unused variables."""
    return None


# ----------------------------------------------------------------------
# Basic types
# ----------------------------------------------------------------------

# Represents a signed 8-bit byte
Byte = c_int8

# Represents a 16-bit signed integer
Int16 = c_int16

# Represents a 16-bit unsigned integer
UInt16 = c_uint16

# Represents a 32-bit signed integer
Int32 = c_int32

# Represents a 32-bit unsigned integer
UInt32 = c_uint32

# Represents a 64-bit signed integer
Int64 = c_int64

# Represents a 64-bit unsigned integer
UInt64 = c_uint64

# Represents a 32-bit real number (float)
Real32 = c_float

# Represents a 64-bit real number (double)
Real64 = c_double

# Represents an opaque handle / pointer
Handle = c_void_p

# Represents lengths of arrays, strings, and so on
Size = c_size_t


# ----------------------------------------------------------------------
# Flexible types (depend on NTA_DOUBLE_PRECISION and NTA_BIG_INTEGER)
# ----------------------------------------------------------------------

# Real: Real64 if double precision, otherwise Real32
Real = Real64 if NTA_DOUBLE_PRECISION else Real32

# Int: Int64 if big integer mode, otherwise Int32
Int = Int64 if NTA_BIG_INTEGER else Int32

# UInt: UInt64 if big integer mode, otherwise UInt32
UInt = UInt64 if NTA_BIG_INTEGER else UInt32


# ----------------------------------------------------------------------
# Basic types enumeration
# ----------------------------------------------------------------------

class NTABasicType(IntEnum):
    """Basic type enumeration, mirroring the C++ NTA_BasicType enum."""

    # Represents an 8-bit byte
    Byte = 0

    # Represents a 16-bit signed integer
    Int16 = 1

    # Represents a 16-bit unsigned integer
    UInt16 = 2

    # Represents a 32-bit signed integer
    Int32 = 3

    # Represents a 32-bit unsigned integer
    UInt32 = 4

    # Represents a 64-bit signed integer
    Int64 = 5

    # Represents a 64-bit unsigned integer
    UInt64 = 6

    # Represents a 32-bit real number
    Real32 = 7

    # Represents a 64-bit real number
    Real64 = 8

    # Represents an opaque handle/pointer (void*)
    Handle = 9

    # Represents a boolean
    Bool = 10

    # Represents an SDR object payload (not an array)
    SDR = 11

    # Represents a std::string-like object
    Str = 12

    # Not an actual type, just a marker for validation
    Last = 13

    # Default-sized unsigned integer
    UInt = 6 if NTA_BIG_INTEGER else 4  # UInt64 or UInt32

    # Default-sized real number
    Real = 8 if NTA_DOUBLE_PRECISION else 7  # Real64 or Real32
