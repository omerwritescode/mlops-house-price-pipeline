"""Exceptions that stop the pipeline cleanly.

Gate failures carry a human-readable explanation and a structured report so the
CLI, the orchestrator and CI can all show *why* the pipeline stopped.
"""

from __future__ import annotations

from typing import Any


class PipelineError(Exception):
    """Base class for expected, explainable pipeline stops."""

    exit_code = 1


class ConfigError(PipelineError):
    exit_code = 4


class StageInputError(PipelineError):
    """A stage was run on its own but the outputs of an upstream stage are missing."""

    exit_code = 5


class GateError(PipelineError):
    gate = "gate"

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report or {}


class DataGateError(GateError):
    gate = "data"
    exit_code = 2


class ModelGateError(GateError):
    gate = "model"
    exit_code = 3
