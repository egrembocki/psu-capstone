"""Lightweight logging helpers mirroring NuPIC's NTA logging macros."""

from __future__ import annotations

import inspect
import logging
import os
import threading
from enum import IntEnum
from typing import Callable, Optional

"""
//this code intentionally uses "if() dosomething" instead of "if() { dosomething }" 
// as the macro expects another "<< "my clever message";
// so it eventually becomes: `if() std::cout << "DEBUG:\t" << "users message";`
//
//Expected usage: 
//<your class>:
//Network::setLogLevel(LogLevel::LogLevel_Verbose);
//NTA_WARN << "Hello World!" << std::endl; //shows
//NTA_DEBUG << "more details how cool this is"; //not showing under "Normal" log level
//NTA_ERR << "You'll always see this, HAHA!";
//NTA_THROW << "crashing for a good cause";

"""


class LogLevel(IntEnum):
	"""Log verbosity levels compatible with NuPIC's C++ macros."""

	NONE = 0
	MINIMAL = 1
	NORMAL = 2
	VERBOSE = 3


DEFAULT_LOG_LEVEL = LogLevel.NORMAL
ASSERTIONS_ENABLED = True
_thread_state = threading.local()

_logger = logging.getLogger("htm")
if not _logger.handlers:
	handler = logging.StreamHandler()
	handler.setFormatter(logging.Formatter("%(message)s"))
	_logger.addHandler(handler)
_logger.setLevel(logging.DEBUG)


def set_log_level(level: LogLevel) -> None:
	"""Set the current thread's log level."""

	_thread_state.log_level = LogLevel(level)


def get_log_level() -> LogLevel:
	"""Return the current thread's log level."""

	return getattr(_thread_state, "log_level", DEFAULT_LOG_LEVEL)


def enable_assertions(enabled: bool) -> None:
	"""Toggle HTM-style assertions globally."""

	global ASSERTIONS_ENABLED
	ASSERTIONS_ENABLED = bool(enabled)


def _call_location(stack_depth: int = 2) -> tuple[str, int]:
	frame = inspect.currentframe()
	for _ in range(stack_depth):
		if frame is None:
			break
		frame = frame.f_back
	if frame is None:
		return "<unknown>", 0
	return os.path.basename(frame.f_code.co_filename), frame.f_lineno


def _format_message(prefix: str, filename: str, lineno: int, message: str) -> str:
	return f"{prefix}:\t{filename}:{lineno}: {message}"


def _log(
	required_level: LogLevel,
	emit: Callable[[str], None],
	prefix: str,
	message: str,
	*args: object,
) -> None:
	if get_log_level() < required_level:
		return
	filename, lineno = _call_location()
	formatted = message.format(*args) if args else message
	emit(_format_message(prefix, filename, lineno, formatted))


def nta_debug(message: str, *args: object) -> None:
	"""Log a verbose debugging message if the log level permits."""

	_log(LogLevel.VERBOSE, _logger.debug, "DEBUG", message, *args)


def nta_info(message: str, *args: object) -> None:
	"""Log an informational message."""

	_log(LogLevel.NORMAL, _logger.info, "INFO", message, *args)


def nta_warn(message: str, *args: object) -> None:
	"""Log a warning message."""

	_log(LogLevel.NORMAL, _logger.warning, "WARN", message, *args)


def nta_error(message: str, *args: object) -> None:
	"""Log an error message that should almost always be shown."""

	_log(LogLevel.MINIMAL, _logger.error, "ERROR", message, *args)


def nta_throw(message: str, *args: object) -> None:
	"""Raise the HTM runtime error after logging it."""

	formatted = message.format(*args) if args else message
	nta_error(formatted)
	raise HtmException(formatted)


def nta_check(condition: bool, message: Optional[str] = None) -> None:
	"""Raise if *condition* is false, mirroring NTA_CHECK."""

	if condition:
		return
	detail = f"CHECK FAILED: {message}" if message else "CHECK FAILED"
	nta_throw(detail)


def nta_assert(condition: bool, message: Optional[str] = None) -> None:
	"""Raise if *condition* is false and assertions are enabled."""

	if not ASSERTIONS_ENABLED or condition:
		return
	detail = f"ASSERT FAILED: {message}" if message else "ASSERT FAILED"
	nta_error(detail)
	raise HtmAssertionError(detail)


__all__ = [
	"HtmException",
	"HtmAssertionError",
	"LogLevel",
	"ASSERTIONS_ENABLED",
	"set_log_level",
	"get_log_level",
	"enable_assertions",
	"nta_debug",
	"nta_info",
	"nta_warn",
	"nta_error",
	"nta_throw",
	"nta_check",
	"nta_assert",
]
