"""Source-shaped Decima rollout data structures.

These helpers intentionally stop short of claiming reproduction results. They
convert official TPC-H templates into the observation tensors consumed by the
source-aligned actor, which gives us a tested interface for later official-env
rollouts without starting Decima experiments before DeepRM is closed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cisose_decima.config import DEFAULT_CONFIG, DecimaConfig
from cisose_decima.model import DecimaPolicy
from cisose_decima.tpch import TpchDagTemplate


@dataclass(frozen=True)
class DecimaObservation:
    template: TpchDagTemplate
    node_features: torch.Tensor
    adjacency: torch.Tensor
    job_features: torch.Tensor
    node_valid_mask: torch.Tensor
    job_valid_mask: torch.Tensor
    dag_summ_backward_map: torch.Tensor
    dag_summary_mat: torch.Tensor
    running_dag_mat: torch.Tensor


@dataclass(frozen=True)
class DecimaAction:
    node_index: int
    executor_level_index: int
    executor_level: int
    node_probability: float
    executor_probability: float


@dataclass(frozen=True)
class DecimaExperienceStep:
    observation: DecimaObservation
    action: DecimaAction
    reward: float


@dataclass(frozen=True)
class DecimaRollout:
    steps: tuple[DecimaExperienceStep, ...]

    @property
    def total_reward(self) -> float:
        return float(sum(step.reward for step in self.steps))


def build_template_observation(
    template: TpchDagTemplate,
    *,
    config: DecimaConfig = DEFAULT_CONFIG,
) -> DecimaObservation:
    num_nodes = template.num_nodes
    node_features = torch.zeros(num_nodes, config.node_input_dim, dtype=torch.float32)
    node_features[:, 0] = 0.0
    node_features[:, 1] = 2.0
    node_features[:, 2] = float(config.exec_cap) / 20.0
    node_features[:, 3] = torch.as_tensor(_node_work(template), dtype=torch.float32) / 100000.0
    node_features[:, 4] = torch.as_tensor(_task_counts(template), dtype=torch.float32) / 200.0

    adjacency = torch.as_tensor(template.adjacency, dtype=torch.float32)
    job_features = torch.tensor([[0.0, 2.0, float(config.exec_cap) / 20.0]], dtype=torch.float32)
    dag_summary_mat = torch.ones(1, num_nodes, dtype=torch.float32)
    dag_summ_backward_map = torch.ones(num_nodes, 1, dtype=torch.float32)
    running_dag_mat = torch.ones(1, 1, dtype=torch.float32)
    node_valid_mask = torch.as_tensor(_source_node_mask(template.adjacency), dtype=torch.float32)
    job_valid_mask = torch.ones(1, len(config.actor_executor_levels), dtype=torch.float32)
    return DecimaObservation(
        template=template,
        node_features=node_features,
        adjacency=adjacency,
        job_features=job_features,
        node_valid_mask=node_valid_mask,
        job_valid_mask=job_valid_mask,
        dag_summ_backward_map=dag_summ_backward_map,
        dag_summary_mat=dag_summary_mat,
        running_dag_mat=running_dag_mat,
    )


def select_greedy_action(policy: DecimaPolicy, observation: DecimaObservation) -> DecimaAction:
    with torch.no_grad():
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
    node_index = int(torch.argmax(output.node_probs).item())
    executor_level_index = int(torch.argmax(output.job_probs[0]).item())
    return DecimaAction(
        node_index=node_index,
        executor_level_index=executor_level_index,
        executor_level=policy.executor_levels[executor_level_index],
        node_probability=float(output.node_probs[node_index].item()),
        executor_probability=float(output.job_probs[0, executor_level_index].item()),
    )


def rollout_smoke(policy: DecimaPolicy, templates: tuple[TpchDagTemplate, ...], *, count: int) -> DecimaRollout:
    steps: list[DecimaExperienceStep] = []
    for template in templates[:count]:
        observation = build_template_observation(template, config=policy.config)
        action = select_greedy_action(policy, observation)
        reward = -float(template.total_work) / max(float(policy.config.exec_cap), 1.0)
        steps.append(DecimaExperienceStep(observation=observation, action=action, reward=reward))
    return DecimaRollout(tuple(steps))


def _node_work(template: TpchDagTemplate) -> np.ndarray:
    if template.node_work is None:
        return np.ones(template.num_nodes, dtype=np.float32)
    return np.asarray(template.node_work, dtype=np.float32)


def _task_counts(template: TpchDagTemplate) -> np.ndarray:
    if template.task_counts is None:
        return np.ones(template.num_nodes, dtype=np.float32)
    return np.asarray(template.task_counts, dtype=np.float32)


def _source_node_mask(adjacency: np.ndarray) -> np.ndarray:
    if adjacency.size == 0:
        return np.zeros(0, dtype=np.float32)
    parent_counts = np.count_nonzero(adjacency, axis=0)
    mask = (parent_counts == 0).astype(np.float32)
    if not np.any(mask):
        mask[:] = 1.0
    return mask
