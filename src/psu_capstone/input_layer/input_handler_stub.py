"""input handler stub - TDD"""

from typing_extensions import Self
import pandas as pd
import numpy as np
from typing import Union
import logging


class InputHandler:
    """Stub InputHandler class for TDD purposes."""

    # Set up logging for this module
    logging.basicConfig(level=logging.INFO)

    _instance = None
    """The single instance of the InputHandler class."""

    _data: pd.DataFrame = pd.DataFrame()
    """The input data as a DataFrame."""

    _raw_data: Union[str, bytearray, list, dict, np.ndarray, pd.DataFrame] = ""
    """The raw input data of supported type."""

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super(InputHandler, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        assert isinstance(self, InputHandler)
        self._data = pd.DataFrame()
        self._raw_data = ""
        logging.info("InputHandler initialized.")
    
    def load_data(self, filepath: str):
        """Load data from a file into a DataFrame based on file extension."""
        assert isinstance(filepath, str), "File path must be a string"
        if filepath.endswith(".csv"):
            return pd.read_csv(filepath)
        elif filepath.endswith(".xls") or filepath.endswith(".xlsx"):
            return pd.read_excel(filepath)
        elif filepath.endswith(".json"):
            return pd.read_json(filepath)
        elif filepath.endswith(".txt"):
            with open(filepath, "r") as file:
                lines = file.readlines()
            return pd.DataFrame(lines)
        else:
            raise ValueError("Unsupported file type")

    def get_data(self) -> pd.DataFrame:
        """Getter for the data attribute"""
        return self._data
     
