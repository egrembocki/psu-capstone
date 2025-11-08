"""Exceptions for the LogLayer module."""



class HtmException(RuntimeError):
	"""Base exception mirroring the HTM C++ runtime error."""


class HtmAssertionError(HtmException):
	"""Raised when an HTM assertion fails."""