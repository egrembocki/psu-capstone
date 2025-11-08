



class HtmException(RuntimeError):
	"""Base exception mirroring the HTM C++ runtime error."""
	pass

class HtmAssertionError(HtmException):
	"""Raised when an HTM assertion fails."""
	pass