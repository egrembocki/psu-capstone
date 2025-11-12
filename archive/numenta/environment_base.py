"""Common environment abstraction shared across NuPIC-inspired modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from numenta.utils import get_logger


class Environment(ABC):
    """Abstract base class for environments."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def execute(self, motor_commands: Dict[str, Any]) -> Dict[str, Any]:
        """Run *motor_commands* against the environment and return sensory stimuli."""

        if not motor_commands:
            self.logger.warning("No motor commands provided to execute.")

        if motor_commands and len(motor_commands):
            self.run_commands(motor_commands)

        current_stimuli = self.receive_sensory_stimuli()
        return current_stimuli

    @abstractmethod
    def run_commands(self, motor_commands: Dict[str, Any]) -> None:
        """Execute the provided commands in the concrete environment."""

    @abstractmethod
    def receive_sensory_stimuli(self) -> Dict[str, Any]:
        """Return the latest sensory input from the environment."""


__all__ = ["Environment"]
