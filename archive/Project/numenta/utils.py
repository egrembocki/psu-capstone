import logging

import randomname


def get_logger(name):
    """Return a project-level logger prefixed with the lidapy namespace."""
    return logging.getLogger(f"lidapy.{name}")


logger = get_logger(__name__)


def random_name():
    return randomname.get_name()
