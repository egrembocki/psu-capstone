


import InputLayer.InputHandler as ih
#import EncoderLayer.Sdr as sdr
import pathlib as path
import os


ROOT_PATH = path.Path(__file__).parent.parent.parent.parent

DATA_PATH = ROOT_PATH / "Data"

"""Driver code to test InputHandler functionality."""

def main():
    """Main function to demonstrate InputHandler usage."""
    # Create an instance of InputHandler
    handler = ih.InputHandler()
    
    # Set some raw data, will need more abstraction later
    data_set = handler.load_data(os.path.join(DATA_PATH, "concat_ESData.xlsx"))

    print("Raw Data Loaded.", type(data_set), "\n", DATA_PATH)

    # Explicitly convert raw data to DataFrame
    data_frame = handler.to_dataframe(data_set)

    handler.fill_missing_values(data_frame)

    print("Data Frame Created.", type(data_frame), "\n", data_frame.head())
    print("Data Validation:", handler.validate_data())
    print(data_frame.info())

if __name__ == "__main__":
      
    main()