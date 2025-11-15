"""Visual tests for SDR class."""

from time import time
from matplotlib.colors import ListedColormap
import pytest
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from psu_capstone.encoder_layer.sdr import SDR


@pytest.fixture
def sdr_visualization(debug=False):
    """Fixture for SDR visualization tests."""
    pass  # No setup needed for this simple visualization test


def test_sdr_visualization(sdr_visualization):
    """Test the visualization of an SDR."""

    # sdr.get_dense() should return a list or 1D array of 0/1
    sdr = SDR([64, 32])
    sdr.randomize(0.02)
    sdr.add_noise(0.01)
    dense = np.array(sdr.get_dense())

    # compute square grid size
    n = dense.size
    side = int(np.ceil(np.sqrt(n)))  # smallest square that fits SDR

    # pad if needed
    padded = np.zeros(side * side, dtype=int)
    padded[:n] = dense

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


def test_sdr_one_row_visual():
    """Test visualization of a single row SDR."""
    # Arrange
    sdr = SDR([100])
    sdr.randomize(0.05)

    dense = np.array(sdr.get_dense())
    arr2d = dense.reshape(1, -1)  # one row, N columns

    # ON bits = blue, OFF bits = white
    cmap = ListedColormap(["white", "blue"])

    plt.figure(figsize=(12, 2))
    plt.imshow(arr2d, cmap=cmap, aspect="auto", interpolation="nearest")
    plt.yticks([])  # Remove y-axis
    plt.xlabel("Bit Index")
    plt.title("SDR (1D One-Row Visual)")
    plt.show()


def test_sdr_union_visual():
    """Visualize multiple SDRs as a stacked view."""
