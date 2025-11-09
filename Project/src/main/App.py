"""Main application module to test InputHandler and SDR functionality."""

from __future__ import annotations

import os
import pathlib as path

try:  # Package-relative imports when executed via ``python -m Project.src.main.App``
    from .EncoderLayer import SDR
    from .InputLayer import inputHandler as ih
except ImportError:  # Fallback for running ``python Project/src/main/App.py`` directly
    if __package__ is None or __package__ == "":
        import sys
        PROJECT_PACKAGE_PARENT = path.Path(__file__).resolve().parents[3]
        sys.path.append(str(PROJECT_PACKAGE_PARENT))
        from Project.src.main.EncoderLayer import SDR  # type: ignore[no-redef]
        from Project.src.main.InputLayer import inputHandler as ih  # type: ignore[no-redef]
    else:
        raise



ROOT_PATH = path.Path(__file__).parent.parent.parent.parent

DATA_PATH = ROOT_PATH / "Data"


def main() -> None:
    """Main function to demonstrate InputHandler usage."""
    # Create an SDR instance demoing the encoder layer
    sdr_test = SDR([50, 50])
    sdr_test.set_sparse([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48])
    print("SDR Sparse Test:", sdr_test)

    sdr_test_dense = SDR([2,2])
    sdr_test_dense.set_dense([0, 0,1,1])
    print("SDR Dense Test:", sdr_test_dense)


    

    






if __name__ == "__main__":

    main()
