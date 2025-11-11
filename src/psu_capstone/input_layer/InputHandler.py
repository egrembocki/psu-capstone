"""InputHandler singleton to pass a Data object to the Encoder layer.
Implemented as a singleton  layer handler for now, with methods to convert raw data to
DataFrame, sequence, etc.
TODO : make a logger class to log messages instead of print statements.
"""


import datetime
import pandas as pd
import numpy as np
import os
from typing import Union


class InputHandler:
    """
    Singleton InputHandler class to handle input data.

    """

    _instance = None

    def __new__(cls) -> "InputHandler":
        """Constructor -- Singleton pattern implementation."""

        if cls._instance is None:
            cls._instance = super(InputHandler, cls).__new__(cls)

        return cls._instance

    def __init__(self):
        """Initialize the InputHandler singleton."""

        self._instance = None
        """The singleton instance."""

        self._data = pd.DataFrame()
        """The input data of any type."""

    # Getters, maybe use properties later
    def get_data(self) -> pd.DataFrame:
        """Getter for the data attribute"""
        return pd.DataFrame(self._data)

    # main methods to handle input data processing

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from a file with padas based on file extension.
        This will automatically create a dataframe."""

        assert os.path.exists(filepath), f"The file {filepath} does not exist."
        assert isinstance(filepath, str), "Filepath must be a string."
        assert len(filepath) > 0, "Filepath cannot be empty."
        assert isinstance(
            self, InputHandler
        ), "load_data must be called on an InputHandler instance."

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist.")

        file_extension = os.path.splitext(filepath)[1].lower()

        if file_extension == ".csv":
            print("Loading csv file:", file_extension, filepath)
            self._data = pd.read_csv(filepath)
            return self._data
        elif file_extension in [".xls", ".xlsx"]:
            print("Loading excel file:", file_extension, filepath)
            self._data = pd.read_excel(filepath)
            return self._data
        elif file_extension == ".json":
            print("Loading json file:", file_extension, filepath)
            self._data = pd.read_json(filepath)
            return self._data
        elif file_extension == ".txt":
            # setup context manager to read text file
            with open(filepath, "r") as file:
                self._data = file.readlines()
            return pd.DataFrame(self._data)

        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def to_dataframe(
        self, data: Union[pd.DataFrame, list, bytearray, np.ndarray]
    ) -> pd.DataFrame:
        """Explicitly convert input data to a pandas DataFrame"""

        assert isinstance(data, (pd.DataFrame, list, bytearray, np.ndarray)), (
            "Data must be a DataFrame, list, bytearray, or numpy ndarray."
        )
        temp_data: pd.DataFrame

        if isinstance(data, pd.DataFrame):
            print("Data is already a DataFrame.")
            temp_data = data
        elif isinstance(data, list):
            print("Converting data to DataFrame.")
            temp_data = pd.DataFrame(data)
        elif isinstance(data, bytearray):
            print("Converting bytearray to DataFrame.")
            temp_data = pd.DataFrame(list(data))
        elif isinstance(data, np.ndarray):
            print("Converting numpy array to DataFrame.")
            temp_data = pd.DataFrame(data)
        else:
            raise TypeError("Unsupported data type for conversion to DataFrame.")
        self._fill_missing_values(temp_data)
        return temp_data

    def raw_to_sequence(
        self, data: Union[list, bytearray, bytes, np.ndarray, str]
    ) -> list:
        """Convert raw data to a normalized sequence list with guaranteed date metadata."""

        assert isinstance(data, (list, bytearray, bytes, np.ndarray, str)), (
            "Data must be a list, bytearray, bytes, numpy ndarray, or string."
        )
        if isinstance(data, np.ndarray):
            iterable = data.tolist()
        elif isinstance(data, (bytearray, bytes)):
            iterable = list(data)
        elif isinstance(data, list):
            iterable = data[:]
        elif isinstance(data, str):
            iterable = [data]
        else:
            raise TypeError("Unsupported data type for conversion to sequence.")

        sequence: list = []
        contains_date = False

        for item in iterable:
            normalized, is_date = self._normalize_datetime_entry(item)
            sequence.append(normalized)
            if is_date:
                contains_date = True

        if not contains_date:
            sequence.insert(0, datetime.datetime.now().isoformat())

        return sequence

    def _normalize_datetime_entry(self, value: object) -> tuple[object, bool]:
        """Normalize datetime-like values to ISO strings and report detection."""
        if isinstance(value, datetime.datetime):
            return value.isoformat(), True
        if isinstance(value, datetime.date):
            return datetime.datetime.combine(value, datetime.time()).isoformat(), True
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime().isoformat(), True
        if isinstance(value, str):
            try:
                parsed = datetime.datetime.fromisoformat(value)
                return parsed.isoformat(), True
            except ValueError:
                return value, False
        return value, False

    def _fill_missing_values(self, data: pd.DataFrame) -> None:
        """Fill missing values in the input data"""

        # Placeholder implementation; actual logic will depend on data type and requirements

        if isinstance(data, pd.DataFrame):
            data.fillna(data.mean(numeric_only=True), inplace=True)

        # Add more cases as needed for different data types

    # validation methods

    def validate_data(self) -> bool:
        """Validate the input data"""

        assert isinstance(self._data, pd.DataFrame), "Data is not a DataFrame."
        if self._data.empty:
            print("DataFrame is empty.")
            return False
        return True  # Data is valid if not empty
