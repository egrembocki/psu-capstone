# htm_core/demo.py
"""Demo application for Spatial Pooler and Temporal Memory integration."""

from __future__ import annotations

import numpy as np
from ModelLayer.HTM_Model.spatial_pooler import SpatialPooler
from ModelLayer.HTM_Model.temporal_memory import TemporalMemory


def main() -> None:
    input_space_size = 100
    column_count = 256
    cells_per_column = 8
    initial_synapses_per_column = 20
    steps = 5
    inhibition_radius = 10.0

    sp = SpatialPooler(input_space_size, column_count, initial_synapses_per_column)
    tm = TemporalMemory(sp.columns, cells_per_column)
    rng = np.random.default_rng(seed=42)

    for t in range(steps):
        input_vector = rng.integers(0, 2, size=input_space_size)
        _, active_cols = sp.compute_active_columns(input_vector, inhibition_radius)
        tm.step(active_cols)

        if t == 0:
            # Example: run one SP learning step at t=0
            sp.learning_phase(active_cols, input_vector)

    print("SP + TM demo completed.")


if __name__ == "__main__":
    main()
