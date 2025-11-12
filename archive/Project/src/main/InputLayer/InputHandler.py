import os
from typing import Union

import numpy as np
import pandas as pd

"""InputHandler singleton to pass a Data object to the Encoder layer. Implemented as a singleton layer handler for now, with methods to convert raw data to DataFrame, sequence, etc."""


class InputHandler:
    """
    Singleton InputHandler class to handle input data.

    """

    _instance = None
    """The single instance of the InputHandler class."""

    _data: Union[pd.DataFrame, list] = pd.DataFrame()
    """The input data of any type."""

    _hyperparameters: dict = {}
    """The hyperparameters associated with the input data."""

    # Getters, maybe use properties later
    def get_data(self) -> Union[pd.DataFrame, list]:
        """Getter for the data attribute"""
        return self._data

    def get_hyperparameters(self) -> dict:
        """Getter for the hyperparameters attribute"""
        return self._hyperparameters

    # main methods to handle input data processing

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from a file with padas based on file extension. This will automatically create a dataframe."""

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

    def to_dataframe(self, data: Union[pd.DataFrame, list]) -> pd.DataFrame:
        """Explicitly convert input data to a pandas DataFrame"""

        # Placeholder implementation; actual conversion logic will depend on data type
        # the main goal is to get the dataset to list first for easy pandas conversion to DataFrame

        if isinstance(data, pd.DataFrame):
            print("Data is already a DataFrame.")
            return data
        elif isinstance(data, list):
            print("Converting data to DataFrame.")

            return pd.DataFrame(data)
        else:
            raise TypeError("Unsupported data type for conversion to DataFrame.")

    def raw_to_sequence(self) -> list:
        """Convert raw data to sequence (list)"""

        # Placeholder implementation; actual conversion logic will depend on data type
        # could be list, numpy array, pandas series, etc.

        if isinstance(self._data, list):
            return self._data
        else:
            # simple conversion to list, more logic is needed based on what the data is
            return [self._data]

    # validation methods

    def validate_data(self) -> bool:
        """Validate the input data"""

        # Placeholder implementation; actual validation logic will depend on data type and requirements

        is_valid = isinstance(self._data, (pd.DataFrame, list, np.ndarray, pd.Series, dict, str))

        return is_valid

    def fill_missing_values(self, data: pd.DataFrame) -> None:
        """Fill missing values in the input data"""

        # Placeholder implementation; actual logic will depend on data type and requirements

        if isinstance(data, pd.DataFrame):
            data.fillna(data.mean(numeric_only=True), inplace=True)

        # Add more cases as needed for different data types

    def __new__(cls, *args, **kwargs):
        """Constructor -- Singleton pattern implementation."""

        if cls._instance is None:
            cls._instance = super(InputHandler, cls).__new__(cls)

        return cls._instance
