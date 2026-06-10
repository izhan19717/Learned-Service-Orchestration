"""PyTorch Decima actor components aligned with the official source."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cisose_decima.config import DEFAULT_CONFIG, DecimaConfig


def _leaky_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(x)


class MLP(nn.Module):
    """Dense MLP matching Decima's TensorFlow fully-connected blocks."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int):
        super().__init__()
        dims = (input_dim, *hidden_dims, output_dim)
        layers: list[nn.Module] = []
        for idx, (din, dout) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            layers.append(nn.Linear(din, dout))
            if idx < len(dims) - 2:
                layers.append(nn.LeakyReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphCNN(nn.Module):
    """Decima GraphCNN: prepare, process, aggregate, repeated by depth."""

    def __init__(self, config: DecimaConfig = DEFAULT_CONFIG):
        super().__init__()
        self.config = config
        self.prepare = MLP(config.node_input_dim, config.hid_dims, config.output_dim)
        self.process = MLP(config.output_dim, config.hid_dims, config.output_dim)
        self.aggregate = MLP(config.output_dim, config.hid_dims, config.output_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        *,
        masks: tuple[torch.Tensor, ...] | None = None,
        adj_mats: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if node_features.ndim != 2:
            raise ValueError("node_features must have shape [num_nodes, feature_dim]")
        if adjacency.shape[0] != adjacency.shape[1] or adjacency.shape[0] != node_features.shape[0]:
            raise ValueError("adjacency must be square and match node count")
        if adj_mats is None:
            adj_mats = tuple(adjacency for _ in range(self.config.max_depth))
        if masks is None:
            masks = tuple(torch.ones(node_features.shape[0], 1, device=node_features.device) for _ in adj_mats)
        if len(adj_mats) != len(masks):
            raise ValueError("adj_mats and masks must have the same length")

        x = self.prepare(node_features)
        for adj, mask in zip(adj_mats, masks, strict=True):
            y = self.process(x)
            y = adj.to(device=x.device, dtype=x.dtype) @ y
            y = self.aggregate(y)
            y = y * mask.to(device=x.device, dtype=x.dtype)
            x = x + y
        return x


class GraphSNN(nn.Module):
    """Decima GraphSNN with DAG-level and global-level summaries."""

    def __init__(self, input_dim: int, config: DecimaConfig = DEFAULT_CONFIG):
        super().__init__()
        self.config = config
        self.dag_summary = MLP(input_dim, config.hid_dims, config.output_dim)
        self.global_summary = MLP(config.output_dim, config.hid_dims, config.output_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        dag_summary_mat: torch.Tensor,
        running_dag_mat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dag_hidden = self.dag_summary(inputs)
        dag_summary = dag_summary_mat.to(device=inputs.device, dtype=inputs.dtype) @ dag_hidden
        global_hidden = self.global_summary(dag_summary)
        global_summary = running_dag_mat.to(device=inputs.device, dtype=inputs.dtype) @ global_hidden
        return dag_summary, global_summary


@dataclass(frozen=True)
class DecimaActorOutput:
    node_probs: torch.Tensor
    job_probs: torch.Tensor
    node_scores: torch.Tensor
    job_scores: torch.Tensor


class DecimaPolicy(nn.Module):
    """Source-aligned Decima actor.

    The official TensorFlow actor emits two distributions: one over schedulable
    nodes, and one over executor-limit levels per running DAG. This module keeps
    that split while exposing a backward-compatible `forward()` that returns
    node probabilities for simple callers.
    """

    def __init__(
        self,
        config: DecimaConfig = DEFAULT_CONFIG,
        *,
        executor_levels: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.config = config
        self.executor_levels = tuple(executor_levels or config.actor_executor_levels)
        self.gcn = GraphCNN(config)
        self.gsn = GraphSNN(config.node_input_dim + config.output_dim, config)
        node_score_dim = config.node_input_dim + 3 * config.output_dim
        job_score_dim = config.job_input_dim + 2 * config.output_dim + 1
        self.node_score = MLP(node_score_dim, config.actor_hidden_dims, 1)
        self.job_score = MLP(job_score_dim, config.actor_hidden_dims, 1)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.predict(node_features, adjacency).node_probs

    def predict(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        *,
        job_features: torch.Tensor | None = None,
        node_valid_mask: torch.Tensor | None = None,
        job_valid_mask: torch.Tensor | None = None,
        dag_summ_backward_map: torch.Tensor | None = None,
        dag_summary_mat: torch.Tensor | None = None,
        running_dag_mat: torch.Tensor | None = None,
        adj_mats: tuple[torch.Tensor, ...] | None = None,
        masks: tuple[torch.Tensor, ...] | None = None,
    ) -> DecimaActorOutput:
        if job_features is None:
            job_features = torch.zeros(1, self.config.job_input_dim, device=node_features.device, dtype=node_features.dtype)
        if dag_summary_mat is None:
            dag_summary_mat = torch.ones(job_features.shape[0], node_features.shape[0], device=node_features.device)
        if running_dag_mat is None:
            running_dag_mat = torch.ones(1, job_features.shape[0], device=node_features.device)
        if dag_summ_backward_map is None:
            dag_summ_backward_map = torch.ones(node_features.shape[0], job_features.shape[0], device=node_features.device)
        if node_valid_mask is None:
            node_valid_mask = torch.ones(node_features.shape[0], device=node_features.device)
        if job_valid_mask is None:
            job_valid_mask = torch.ones(job_features.shape[0], len(self.executor_levels), device=node_features.device)

        gcn_outputs = self.gcn(node_features, adjacency, masks=masks, adj_mats=adj_mats)
        dag_summary, global_summary = self.gsn(
            torch.cat([node_features, gcn_outputs], dim=1),
            dag_summary_mat,
            running_dag_mat,
        )
        dag_summary_by_node = dag_summ_backward_map.to(device=node_features.device, dtype=node_features.dtype) @ dag_summary
        global_by_node = global_summary.expand(node_features.shape[0], -1)
        node_state = torch.cat([node_features, gcn_outputs, dag_summary_by_node, global_by_node], dim=1)
        node_scores = self.node_score(node_state).squeeze(-1)
        node_scores = node_scores + (node_valid_mask.to(node_scores.dtype) - 1.0) * 10000.0
        node_probs = torch.softmax(node_scores, dim=-1)

        global_by_job = global_summary.expand(job_features.shape[0], -1)
        job_state = torch.cat([job_features, dag_summary, global_by_job], dim=1)
        levels = torch.as_tensor(
            [level / 50.0 for level in self.executor_levels],
            dtype=job_state.dtype,
            device=job_state.device,
        )
        expanded_job = job_state[:, None, :].expand(-1, len(self.executor_levels), -1)
        expanded_levels = levels[None, :, None].expand(job_state.shape[0], -1, -1)
        job_with_levels = torch.cat([expanded_job, expanded_levels], dim=2)
        job_scores = self.job_score(job_with_levels).squeeze(-1)
        job_scores = job_scores + (job_valid_mask.to(job_scores.dtype) - 1.0) * 10000.0
        job_probs = torch.softmax(job_scores, dim=-1)
        return DecimaActorOutput(
            node_probs=node_probs,
            job_probs=job_probs,
            node_scores=node_scores,
            job_scores=job_scores,
        )


def parameter_count(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def fgsm_node_features(
    policy: DecimaPolicy,
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    epsilon: float,
    perturb_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Untargeted FGSM on Decima's clean preferred node action.

    The official node feature schema contains one categorical channel
    (`source_job`, encoded as 2 or -2). The primary Decima attack perturbs only
    continuous observation channels and leaves this categorical flag unchanged.
    """

    x = node_features.detach().clone().requires_grad_(True)
    probs = policy(x, adjacency)
    action = int(torch.argmax(probs).item())
    loss = -torch.log(probs[action].clamp_min(1e-12))
    loss.backward()
    if perturb_mask is None:
        perturb_mask = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0], dtype=x.dtype, device=x.device)
    mask = perturb_mask.to(device=x.device, dtype=x.dtype).reshape(1, -1)
    if mask.shape[1] != x.shape[1]:
        raise ValueError("perturb_mask must have one entry per node feature")
    perturbed = x + epsilon * torch.sign(x.grad) * mask
    lower = torch.tensor([0.0, -2.0, 0.0, 0.0, 0.0], dtype=x.dtype, device=x.device)
    upper = torch.tensor(
        [
            float(policy.config.exec_cap) / 20.0,
            2.0,
            float(policy.config.exec_cap) / 20.0,
            float("inf"),
            float("inf"),
        ],
        dtype=x.dtype,
        device=x.device,
    )
    perturbed = torch.minimum(torch.maximum(perturbed, lower.reshape(1, -1)), upper.reshape(1, -1))
    perturbed[:, 1] = node_features[:, 1]
    return perturbed.detach()
