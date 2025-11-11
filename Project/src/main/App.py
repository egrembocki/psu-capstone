"""Main application module for the project."""

import pathlib as path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from typing import Tuple
from EncoderLayer.Sdr import SDR

ROOT_PATH = path.Path(__file__).parent.parent.parent.parent
DATA_PATH = ROOT_PATH / "Data"
ENCODER_LAYER_PATH = ROOT_PATH / "src" / "main" / "EncoderLayer"


def _sdr_to_grid(sdr: SDR) -> list[list[int]]:
    """Return SDR data reshaped into a 2D grid for plotting."""

    dims = sdr.get_dimensions()
    dense = sdr.get_dense()

    if not dims:
        return [[]]

    if len(dims) == 1:
        rows, cols = 1, dims[0]
    else:
        rows = dims[0]
        cols = len(dense) // rows if rows else len(dense)

    if rows <= 0 or cols <= 0 or rows * cols != len(dense):
        rows, cols = 1, len(dense)

    return [dense[row * cols : (row + 1) * cols] for row in range(rows)]


def plot_sdrs(*named_sdrs: Tuple[str, SDR]) -> None:
    """Visualise SDRs as heatmaps where active bits are highlighted."""

    if plt is None:
        print("Matplotlib not installed; skipping SDR plots.")
        return

    if not named_sdrs:
        return

    total = len(named_sdrs)
    cols = min(3, total)
    rows = (total + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))

    if hasattr(axes, "flat"):
        axes_list = list(axes.flat)
    elif isinstance(axes, (list, tuple)):
        axes_list = []
        for entry in axes:
            if isinstance(entry, (list, tuple)):
                axes_list.extend(entry)
            else:
                axes_list.append(entry)
    else:
        axes_list = [axes]

    cmap = ListedColormap(["#f5f5f5", "#1f77b4"]) if ListedColormap else "Blues"

    for ax, (title, sdr) in zip(axes_list, named_sdrs):
        grid = _sdr_to_grid(sdr)
        ax.imshow(grid, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    for extra_axis in axes_list[len(named_sdrs) :]:
        extra_axis.axis("off")

    fig.tight_layout()
    plt.show()


def main() -> None:
    """Main function to demonstrate InputHandler usage."""
    # Create an SDR instance demoing the encoder layer
    sdr_one = SDR([10, 10])
    sdr_two = SDR([10, 10])
    sdr_three = SDR([10, 10])
    sdr_cat = SDR([30, 10])

    sdr_one.randomize(0.02)
    sdr_two.randomize(0.02)
    sdr_three.randomize(0.02)

    print("SDR One:")
    print(sdr_one)
    print("SDR Two:")
    print(sdr_two)
    print("SDR Three:")
    print(sdr_three)

    sdr_cat.concatenate([sdr_two, sdr_one, sdr_three], axis=0)
    print("Union of SDR One,SDR Two, and SDR Three:")
    print(sdr_cat)

    sdr_sparse = SDR([32, 64])
    sdr_sparse.randomize(0.02)

    print("Sparse SDR:")
    print(sdr_sparse)

    plot_sdrs(
        ("SDR One", sdr_one),
        ("SDR Two", sdr_two),
        ("SDR Three", sdr_three),
        ("Union", sdr_cat),
        ("Sparse", sdr_sparse),
    )


if __name__ == "__main__":

    main()
