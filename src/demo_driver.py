"""
demo_driver.py

"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from psu_capstone.encoder_layer.encoder_handler import EncoderHandler
from psu_capstone.encoder_layer.sdr import SDR
from psu_capstone.input_layer.input_handler import InputHandler


def visualize_sdr_all_rows(sdr: SDR, title: str = "Composite SDR – All Rows"):
    """
    Visualize a multi-row SDR: each row in the SDR's first dimension is one demo row.
    """
    dense = np.array(sdr.get_dense())

    # reshape according to SDR dimensions (e.g., [num_rows, bits_per_row])
    if len(sdr.dimensions) == 1:
        arr2d = dense.reshape(1, -1)
        row_labels = ["Row 0"]
    else:
        arr2d = dense.reshape(sdr.dimensions)
        num_rows = sdr.dimensions[0]
        row_labels = [f"Row {i}" for i in range(num_rows)]

    cmap = ListedColormap(["white", "blue"])

    plt.figure(figsize=(12, 3 + arr2d.shape[0]))
    plt.imshow(arr2d, cmap=cmap, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.xlabel("Bit Index")
    plt.title(title)
    plt.show()


def build_demo_dataframe() -> pd.DataFrame:

    rows = [
        {
            "temp_c": 21.5,
            "visits": 3,
            "country": "US",
            "timestamp": datetime(2023, 12, 25, 8, 30),
        },
        {
            "temp_c": 4.5,
            "visits": 12,
            "country": "US",
            "timestamp": datetime(2015, 3, 25, 8, 30),
        },
    ]
    return pd.DataFrame(rows)


def main():

    ih = InputHandler()

    excel_path = (
        r"C:\Users\alexb\Desktop\SWENG 480-481 Final Project\psu-capstone\data\concat_ESData.xlsx"
    )
    full_df = ih.load_data(excel_path)

    print("Raw DataFrame from Excel:")
    print(full_df.head())

    # 👇 pick as many rows as you want here
    # demo_df = full_df.iloc[0:10]      # first 10 rows
    # demo_df = full_df                # or ALL rows
    demo_df = full_df.sample(5)  # or a random 5 rows

    df = ih.to_dataframe(demo_df)

    handler = EncoderHandler(df)
    composite: SDR = handler.build_composite_sdr(df)

    print("Composite SDR dimensions:", composite.dimensions)
    print("Composite SDR size:", composite.size)

    visualize_sdr_all_rows(composite, title="Composite SDR – All Rows")


if __name__ == "__main__":
    main()
