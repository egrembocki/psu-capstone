import os

# import EncoderLayer.Sdr as sdr
import pathlib as path

from InputLayer.InputHandler import InputHandler


def plot_sdrs(*named_sdrs: Tuple[str, SDR]) -> None:
    """Visualise SDRs as heatmaps where active bits are highlighted."""

    if plt is None:
        print("Matplotlib not installed; skipping SDR plots.")
        return

    if not named_sdrs:
        return


def main():
    """Main function to demonstrate InputHandler usage."""
    # Create an instance of InputHandler
    handler = InputHandler()

    # Set some raw data, will need more abstraction later
    data_set = handler.load_data(os.path.join(DATA_PATH, "concat_ESData.xlsx"))

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
