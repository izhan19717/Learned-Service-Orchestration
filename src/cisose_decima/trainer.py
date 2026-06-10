"""Rollout-compatible actor update for the Decima PyTorch port."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cisose_decima.model import DecimaPolicy
from cisose_decima.rollout import DecimaRollout
from cisose_decima.training import actor_loss, discount


@dataclass(frozen=True)
class DecimaTrainStepResult:
    loss: float
    mean_return: float
    total_reward: float
    num_steps: int


def train_on_rollout(
    policy: DecimaPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: DecimaRollout,
    *,
    gamma: float = 1.0,
    entropy_weight: float = 1.0,
) -> DecimaTrainStepResult:
    """Apply one actor update to a Decima-shaped rollout.

    This mirrors the official actor objective over already-collected experience.
    The official multi-agent simulator remains the reproduction gate; this
    helper exists so the PyTorch port has a tested training contract before we
    launch any Decima experiments.
    """

    if not rollout.steps:
        raise ValueError("rollout must contain at least one step")
    returns = torch.as_tensor(discount([step.reward for step in rollout.steps], gamma), dtype=torch.float32)
    baseline = returns.mean()
    losses: list[torch.Tensor] = []
    for idx, step in enumerate(rollout.steps):
        observation = step.observation
        output = policy.predict(
            observation.node_features,
            observation.adjacency,
            job_features=observation.job_features,
            node_valid_mask=observation.node_valid_mask,
            job_valid_mask=observation.job_valid_mask,
            dag_summ_backward_map=observation.dag_summ_backward_map,
            dag_summary_mat=observation.dag_summary_mat,
            running_dag_mat=observation.running_dag_mat,
        )
        loss, _ = actor_loss(
            node_probs=output.node_probs,
            job_probs=output.job_probs,
            node_action=step.action.node_index,
            job_index=0,
            executor_level_index=step.action.executor_level_index,
            advantage=returns[idx] - baseline,
            entropy_weight=entropy_weight,
        )
        losses.append(loss)
    batch_loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()
    return DecimaTrainStepResult(
        loss=float(batch_loss.detach().item()),
        mean_return=float(returns.mean().item()),
        total_reward=rollout.total_reward,
        num_steps=len(rollout.steps),
    )
