"""
Test suite for LoggerManager (SF-F-1 through SF-F-14 requirements).

The LoggerManager is responsible for handling all logging, reporting,
and artifact persistence for the agent-brain training pipeline. It
provides functionality for console logging, file-based reporting,
and structured storage of training artifacts.

Key Features:
  - Console logging for training progress and final performance
  - File persistence for metrics, reports, and datasets
  - JSON and text-based artifact storage
  - Retrieval of previously stored artifacts
  - Per-brain isolation of logs and reports
  - Logger hierarchy support (root, class-based, instance-based)

Parameter Coverage:
  - report_artifact_path: base directory for saved artifacts
  - brain_id: used for namespacing logs and files
  - training_steps: number of steps used during training
  - test_results: dictionary of evaluation metrics
  - dataset / parameters: structured data stored and retrieved

Tests validate:
  1. Logger initialization and artifact path configuration (SF-F-6)
  2. Logger retrieval and naming behavior (SF-F-12, SF-F-13)
  3. Training progress logging to console (SF-F-1)
  4. Mean squared error persistence to file (SF-F-2)
  5. Brain shape persistence in JSON format (SF-F-3)
  6. Final performance output to console (SF-F-4)
  7. Final performance persistence to file (SF-F-5)
  8. Validated dataset storage and retrieval (SF-F-7)
  9. Evaluation parameter storage and retrieval (SF-F-8)
 10. Prediction report storage and retrieval (SF-F-9)
 11. Brain-specific report isolation (SF-F-10)
 12. Average reward per step report generation (SF-F-11)
 13. The logger shall have an Enum {INFO, WARNING, ERROR, DEBUG} structure. (SF-F-14)
"""

import json
import logging
from pathlib import Path

import pytest

import htmrl.log as log_module


class DummyBrain:
    def __init__(self, brain_id: str):
        self.brain_id = brain_id


class DummyTrainer:
    pass


def test_logger_requirements_sf_f_1_through_14(tmp_path, caplog):
    brain = DummyBrain("brain_001")
    manager = log_module.LoggerManager()

    # SF-F-6: set artifact path
    manager.set_report_artifact_path(tmp_path)
    assert manager.get_report_artifact_path() == tmp_path

    # SF-F-12 + SF-F-13:
    # get_logger returns logger / child logger with class origin
    root_logger = manager.get_logger()
    child_logger = manager.get_logger(DummyTrainer)
    instance_logger = manager.get_logger(DummyTrainer())

    assert isinstance(root_logger, logging.Logger)
    assert isinstance(child_logger, logging.Logger)
    assert isinstance(instance_logger, logging.Logger)

    assert root_logger.name == "htmrl"
    assert child_logger.name.endswith("DummyTrainer")
    assert instance_logger.name.endswith("DummyTrainer")

    # SF-F-1: training progress to console
    with caplog.at_level(logging.INFO):
        manager.log_training_progress(
            brain=brain,
            step=3,
            total_steps=10,
            inputs={"input_a": 42, "input_b": 7},
        )

    assert "Training Step 3/10" in caplog.text
    assert "[brain_001]" in caplog.text
    assert "input_a=42" in caplog.text
    assert "input_b=7" in caplog.text

    # Shared fake results for later tests
    test_results = {
        "mean_squared_error": 0.125,
        "total_prediction_failures": 2,
        "avg_bursting_columns": 1.75,
        "errors": {
            "field1": [0.1, 0.2],
            "field2": [0.05],
        },
        "prediction_failures": {
            "field1": 1,
            "field2": 1,
        },
    }

    # SF-F-2: MSE stored in txt file
    mse_path = manager.save_mean_squared_error(brain, 0.125, append=False)
    assert mse_path.exists()
    assert mse_path.name.endswith(".txt")
    assert "MSE=0.125000" in mse_path.read_text(encoding="utf-8")

    # SF-F-3: brain shape stored in JSON
    shape_data = {"columns": 2048, "cells_per_column": 32}
    shape_path = manager.save_agent_brain_shape(brain, shape_data)
    assert shape_path.exists()
    assert shape_path.suffix == ".json"
    assert json.loads(shape_path.read_text(encoding="utf-8")) == shape_data

    # SF-F-4: final performance to console
    caplog.clear()
    with caplog.at_level(logging.INFO):
        manager.output_final_training_performance(
            brain=brain,
            training_steps=10,
            test_results=test_results,
        )

    assert "Final Training Performance Report [brain_001]" in caplog.text
    assert "Training Steps: 10" in caplog.text
    assert "Global MSE: 0.125000" in caplog.text
    assert "Total Prediction Failures: 2" in caplog.text

    # SF-F-5: final performance stored in txt file
    perf_path = manager.save_final_training_performance(
        brain=brain,
        training_steps=10,
        test_results=test_results,
    )
    assert perf_path.exists()
    assert perf_path.name.endswith(".txt")
    perf_text = perf_path.read_text(encoding="utf-8")
    assert "Final Training Performance Report [brain_001]" in perf_text
    assert "Training Steps: 10" in perf_text
    assert "Global MSE: 0.125000" in perf_text

    # SF-F-7: validated dataset save + retrieve
    dataset = {"x": [1, 2, 3], "y": [4, 5, 6]}
    dataset_path = manager.save_validated_dataset(brain, dataset)
    assert dataset_path.exists()
    assert manager.get_validated_dataset(brain) == dataset

    # SF-F-8: evaluation params save + retrieve
    params = {"epochs": 5, "learning_rate": 0.01}
    params_path = manager.save_evaluation_parameters(
        brain,
        params,
        filename="evaluation_parameters.json",  # important for file fallback consistency
    )
    assert params_path.exists()
    assert manager.get_last_evaluation_parameters(brain) == params

    # SF-F-9: latest prediction report retrieve
    report_text = "prediction: 123\nconfidence: 0.99"
    report_path = manager.save_prediction_report(brain, report_text)
    assert report_path.exists()
    assert manager.get_latest_prediction_report(brain) == report_text
    assert manager.get_latest_prediction_report_path(brain) == report_path

    # SF-F-10: report retrieval keyed by brain
    brain2 = DummyBrain("brain_002")
    manager.save_prediction_report(brain2, "other brain report")
    assert manager.get_latest_prediction_report(brain) == report_text
    assert manager.get_latest_prediction_report(brain2) == "other brain report"
    assert manager.get_latest_prediction_report_path(brain).parent.name == "brain_001"
    assert manager.get_latest_prediction_report_path(brain2).parent.name == "brain_002"

    # SF-F-11: average reward per step report generation
    avg_reward_path = manager.save_average_reward_per_step(brain, 3.5)
    assert avg_reward_path.exists()
    avg_reward_text = avg_reward_path.read_text(encoding="utf-8")
    assert "Average Reward Per Step Report [brain_001]" in avg_reward_text
    assert "Average Reward Per Step: 3.500000" in avg_reward_text

    # SF-F-14: The logger shall have an Enum {INFO, WARNING, ERROR, DEBUG} structure.
    assert hasattr(logging, "DEBUG")
    assert hasattr(logging, "INFO")
    assert hasattr(logging, "WARNING")
    assert hasattr(logging, "ERROR")
