"""Actor-critic training primitives for the Decima PyTorch port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


def discount(rewards: Sequence[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        out[idx] = running
    return out


@dataclass
class AveragePerStepReward:
    """Moving average reward per unit time, matching Decima's baseline helper."""

    size: int

    def __post_init__(self) -> None:
        self.reward_record: list[float] = []
        self.time_record: list[float] = []
        self.reward_sum = 0.0
        self.time_sum = 0.0

    def add(self, reward: float, time: float) -> None:
        if len(self.reward_record) >= self.size:
            self.reward_sum -= self.reward_record.pop(0)
            self.time_sum -= self.time_record.pop(0)
        self.reward_record.append(float(reward))
        self.time_record.append(float(time))
        self.reward_sum += float(reward)
        self.time_sum += float(time)

    def add_list_filter_zero(self, rewards: Sequence[float], times: Sequence[float]) -> None:
        if len(rewards) != len(times):
            raise ValueError("rewards and times must have the same length")
        for reward, time in zip(rewards, times, strict=True):
            if time != 0:
                self.add(float(reward), float(time))
            elif reward != 0:
                raise ValueError("zero-duration reward must be zero")

    def get_avg_per_step_reward(self) -> float:
        if self.time_sum == 0:
            return 0.0
        return self.reward_sum / self.time_sum


def actor_loss(
    *,
    node_probs: torch.Tensor,
    job_probs: torch.Tensor,
    node_action: int,
    job_index: int,
    executor_level_index: int,
    advantage: torch.Tensor,
    entropy_weight: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Decima actor loss for one decision.

    The official loss is `-adv * log(p_node * p_job) + entropy_weight * entropy`,
    where the "entropy" term is implemented as sum p log p and is therefore
    non-positive. This function preserves that sign convention.
    """

    selected_node_prob = node_probs[node_action]
    selected_job_prob = job_probs[job_index, executor_level_index]
    adv_loss = torch.log(selected_node_prob * selected_job_prob + eps) * (-advantage)
    node_entropy = torch.sum(node_probs * torch.log(node_probs + eps))
    prob_each_job = torch.zeros(job_probs.shape[0], dtype=job_probs.dtype, device=job_probs.device)
    prob_each_job[job_index] = selected_node_prob.detach()
    job_entropy = torch.sum(prob_each_job * torch.sum(job_probs * torch.log(job_probs + eps), dim=1))
    entropy_loss = node_entropy + job_entropy
    normalizer = torch.log(torch.tensor(float(node_probs.numel()), device=node_probs.device)) + torch.log(
        torch.tensor(float(job_probs.shape[1]), device=node_probs.device)
    )
    entropy_loss = entropy_loss / normalizer.clamp_min(eps)
    loss = adv_loss + float(entropy_weight) * entropy_loss
    return loss, {
        "adv_loss": adv_loss.detach(),
        "entropy_loss": entropy_loss.detach(),
        "selected_node_prob": selected_node_prob.detach(),
        "selected_job_prob": selected_job_prob.detach(),
    }
