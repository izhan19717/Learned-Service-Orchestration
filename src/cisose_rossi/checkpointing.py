"""Checkpoint helpers for Rossi/RLAD model-based controllers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cisose_rossi.actions import DEFAULT_ACTIONS
from cisose_rossi.config import DEFAULT_CONFIG, RossiConfig
from cisose_rossi.controllers import ModelBasedController
from cisose_rossi.state import RladState


def save_model_based_checkpoint(
    path: Path,
    controller: ModelBasedController,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        q=controller.q,
        transition_counts=controller.transition_counts,
        transition_prob=controller.transition_prob,
        unknown_cost=controller.unknown_cost,
        state=np.asarray(
            [controller.state.replicas, controller.state.util_bucket, controller.state.cpu],
            dtype=np.int64,
        ),
        previous_action=np.asarray([controller.previous_action.index], dtype=np.int64),
        lambda_tps=np.asarray([controller.lambda_tps], dtype=np.float64),
        learning_enabled=np.asarray([int(controller.learning_enabled)], dtype=np.int64),
        metadata=np.asarray([json.dumps(metadata or {}, sort_keys=True, default=str)]),
    )


def load_model_based_checkpoint(
    path: Path,
    *,
    config: RossiConfig = DEFAULT_CONFIG,
    freeze: bool = True,
) -> tuple[ModelBasedController, dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    controller = ModelBasedController(config, learning_enabled=not freeze)
    controller.q = np.asarray(data["q"], dtype=np.float64)
    controller.transition_counts = np.asarray(data["transition_counts"], dtype=np.float64)
    controller.transition_prob = np.asarray(data["transition_prob"], dtype=np.float64)
    controller.unknown_cost = np.asarray(data["unknown_cost"], dtype=np.float64)
    state = [int(x) for x in data["state"]]
    controller.state = RladState(replicas=state[0], util_bucket=state[1], cpu=state[2])
    controller.previous_action = DEFAULT_ACTIONS[int(data["previous_action"][0])]
    controller.lambda_tps = float(data["lambda_tps"][0])
    controller.learning_enabled = not freeze
    metadata = json.loads(str(data["metadata"][0]))
    return controller, metadata
