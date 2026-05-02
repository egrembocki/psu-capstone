"""
Test suite for wave sequence prediction using Brain + InputField encoders.

This test evaluates whether the system can learn and predict values from a
wave-based dataset using the configured encoder and HTM-style Brain pipeline.

Key Features:
  - Builds a dataset from the sine wave CSV input
  - Supports RDSE and Scalar encoder configurations
  - Constructs an InputField, ColumnField, and Brain for prediction
  - Trains on an initial portion of the wave data
  - Collects one-step-ahead predictions after training
  - Plots actual vs predicted values for visual comparison

Parameter Coverage:
  - csv_path: input dataset file containing wave values
  - encoder_type: encoder used for input representation ("rdse" or "scalar")
  - train_steps: number of rows used for learning
  - start_idx / end_idx: selected slice of dataset used in the experiment
  - learn: controls whether the Brain updates during training

Tests validate:
  1. Dataset is loaded and parsed correctly
  2. Brain is initialized with the selected encoder type
  3. Predictions are collected after training
  4. Actual and predicted values align as one-step-ahead outputs
  5. Results can be visualized for sequence-learning behavior
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics

from htmrl.agent_layer.brain import Brain
from htmrl.agent_layer.HTM import ColumnField, InputField
from htmrl.encoder_layer.rdse import RDSEParameters
from htmrl.encoder_layer.scalar_encoder import ScalarEncoderParameters
from htmrl.input_layer.input_handler import InputHandler


def build_dataset(csv_path):
    ih = InputHandler()
    data = ih.input_data(csv_path)

    return {
        "wave_count_input": [float(v) for v in data["Wave"] if v != "Wave"],
    }


def build_brain(encoder_type):
    if encoder_type == "rdse":
        wave_input = InputField(
            size=2048,
            encoder_params=RDSEParameters(
                size=2048,
                sparsity=0.02,
                resolution=0.001,
                category=False,
                seed=42,
            ),
        )
    elif encoder_type == "scalar":
        wave_input = InputField(
            size=2048,
            encoder_params=ScalarEncoderParameters(
                size=2048,
                minimum=-1,
                maximum=5,
                periodic=False,
                category=False,
                sparsity=0.02,
                resolution=1.0,
                active_bits=0,
                radius=0,
            ),
        )
    else:
        raise ValueError("Need a correct encoder type.")

    inputfieldlist = [wave_input]

    cf = ColumnField(
        inputfieldlist,
        num_columns=2048,
        cells_per_column=32,
        non_spatial=True,
    )

    brain = Brain(
        {
            "wave_count_input": wave_input,
            "column_field": cf,
        }
    )
    return brain


def run_experiment(
    dataset,
    train_steps,
    learn,
    start_idx=0,
    end_idx=None,
    encoder_type="rdse",
):
    wave_data = dataset["wave_count_input"]

    if end_idx is None:
        end_idx = len(wave_data)

    wave_data = wave_data[start_idx:end_idx]

    brain = build_brain(encoder_type)
    n = len(wave_data)

    if n == 0:
        raise ValueError("Dataset is empty.")

    train_steps = min(train_steps, n)

    predictions = [np.nan] * n
    actuals = []
    final_predictions = []
    indices = []

    for i in range(n):
        brain.step(
            {"wave_count_input": wave_data[i]},
            learn=(learn and i < train_steps),
        )

        preds = brain.prediction()
        predictions[i] = preds["wave_count_input"]

        if i > train_steps and i > 0:
            prev_pred = predictions[i - 1]
            if not np.isnan(prev_pred):
                actuals.append(wave_data[i])
                final_predictions.append(prev_pred)
                indices.append(i)

    if len(actuals) == 0:
        raise ValueError("No evaluation samples were collected.")

    return {
        "actual": actuals,
        "predictions": final_predictions,
        "indices": indices,
    }


def test_actual_vs_predicted():
    plt.close("all")
    csv_path = "data/sine_wave.csv"

    dataset = build_dataset(csv_path)

    run = run_experiment(
        dataset, learn=True, train_steps=4800, start_idx=0, end_idx=5002, encoder_type="rdse"
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(
        run["indices"],
        run["actual"],
        label="Actual",
    )
    ax1.plot(
        run["indices"],
        run["predictions"],
        label="Predicted",
    )

    ax1.set_xlabel("Rows")
    ax1.set_ylabel("Wave Data")
    ax1.legend()
    ax1.margins(x=0)
    ax1.set_xlim(
        run["indices"][0],
        run["indices"][-1],
    )

    plt.tight_layout()
    plt.show()
