import logging

import randomname


def get_logger(name):
    return logging.getLogger(f"lidapy.{name}")


logger = get_logger(__name__)

<<<<<<<< HEAD:numenta/utils.py
random_name = lambda: randomname.get_name()
========

def random_name():
    return randomname.get_name()
>>>>>>>> origin/main:archive/Project/numenta/utils.py
