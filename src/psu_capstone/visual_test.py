"""Driver code to see SDR visualizations."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from psu_capstone.encoder_layer.encoder_handler import EncoderHandler
from psu_capstone.input_layer.input_handler import InputHandler

if __name__ == "__main__":

    input_handler = InputHandler()

    input_handler.load_data("data/easyData.xlsx")

    assert input_handler.get_data() is not None

    df = input_handler.get_data()

    int_columns = df.select_dtypes(include=["int64"]).columns
    df[int_columns] = df[int_columns].astype("float64")

    print("Input DataFrame:")
    print(df.head(), "\n", df.dtypes)

    encoder_handler = EncoderHandler(df)

    sdr_one = encoder_handler.build_composite_sdr(df.iloc[[0]])
    sdr_two = encoder_handler.build_composite_sdr(df.iloc[[1]])

    print("Composite SDR sparse representation:", sdr_one.get_sparse())
    print("Composite SDR size:", sdr_one.size)

    dense_one = np.array(sdr_one.get_dense())

    # compute square grid size
    n = dense_one.size
    side = int(np.ceil(np.sqrt(n)))  # smallest square that fits SDR

    # pad if needed
    padded = np.zeros(side * side, dtype=int)
    padded[:n] = dense_one

    grid = padded.reshape(side, side)

    # colormap: white for 0, blue for 1
    cmap = ListedColormap(["white", "blue"])

    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap=cmap, interpolation="nearest")
    title = "SDR Visualization"
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.show(block=True)

    dense_two = np.array(sdr_two.get_dense())
    padded_two = np.zeros(side * side, dtype=int)
    padded_two[:n] = dense_two
    grid_two = padded_two.reshape(side, side)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_two, cmap=cmap, interpolation="nearest")
    title = "SDR Visualization"
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.show(block=True)
