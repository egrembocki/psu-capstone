"""InputData module :: contains the InputData class to handle input data of any type.
    Some of this needs to get divided into smaller classes later on.
"""

from __future__ import annotations

import os
import pandas as pd

class InputData:
    """Class to handle input data :: of any type."""
    
    _data = None
    """The input data of any type."""

    _hyperparameters: dict = {}
    """The hyperparameters associated with the input data."""

    _dataframe: pd.DataFrame = pd.DataFrame()
    """The output data as a pandas DataFrame."""

    def __new__(cls, *args, **kwargs) -> InputData:
        return super(InputData, cls).__new__(cls)

    def __init__(self, data: object, hyperparameters: dict = {}) -> None:
        self._data = data
        self._hyperparameters: dict = hyperparameters


    # Getters, maybe use properties later
    def get_data(self) -> pd.DataFrame:
        """ Getter for the data attribute """
        return self._dataframe  
    
    def get_hyperparameters(self) -> dict:
        """ Getter for the hyperparameters attribute """
        return self._hyperparameters
   

   # main methods to handle input data processing

    def raw_to_sequence(self) -> list:
        """ Convert raw data to sequence (list) """
        # Placeholder implementation; actual conversion logic will depend on data type 
        # could be list, numpy array, pandas series, etc.

        if isinstance(self._data, list):
            return self._data
        else:
            # simple conversion to list, more logic is needed based on what the data is
            return [self._data] 
   
    def raw_to_dataframe(self) -> pd.DataFrame:
        """ Convert raw data to pandas DataFrame """

        # Placeholder implementation; actual conversion logic will depend on data type
        # convert based on file extension, data structure, etc. -> use pandas read functions
        # might be a need for strategy pattern here for different data types

        if isinstance(self._data, pd.DataFrame):
            self._dataframe = self._data
        else:
            self._dataframe = pd.DataFrame([self._data])
        return self._dataframe
    
    
    def sequence_to_dataframe(self, sequence: list) -> pd.DataFrame:
        """ Convert a sequence (list) to pandas DataFrame """
        self._dataframe = pd.DataFrame(sequence)
        return self._dataframe
    
    
    
    
    
    
    
    
    
    # Helpers for file handling
    def check_path_validity(self, path: str) -> bool:
        """ Check if the given path is valid """
        return os.path.exists(path)
    

    def find_file_type(self, filename: str) -> str:
        """ Find the file type based on the file extension """
        _, file_extension = os.path.splitext(filename)
        return file_extension.lower()