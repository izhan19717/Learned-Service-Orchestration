"""PyTorch DeepRM policy model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cisose_deeprm.protocol import DeepRMConfig
from cisose_deeprm.schedulers import Scheduler
from cisose_deeprm.simulator import DeepRMEnv


class DeepRMPolicy(nn.Module):
    def __init__(self, config: DeepRMConfig | None = None, hidden_dim: int = 20):
        super().__init__()
        self.config = config or DeepRMConfig()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.config.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.config.action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 3:
            state = state.unsqueeze(0)
        return self.net(state.float())

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class DeepRMScheduler:
    policy: DeepRMPolicy
    deterministic: bool = False
    generator: torch.Generator | None = None
    name: str = "DeepRM"

    def act(self, env: DeepRMEnv) -> int:
        state = torch.from_numpy(env.observe()).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(state)
            if self.deterministic:
                return int(torch.argmax(logits, dim=-1).item())
            probs = torch.softmax(logits.squeeze(0), dim=-1)
            return int(torch.multinomial(probs, 1, generator=self.generator).item())


def save_checkpoint(
    policy: DeepRMPolicy,
    path: Path,
    *,
    metadata: dict[str, object],
    optimizer_state: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": policy.state_dict(),
        "metadata": metadata,
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    torch.save(payload, path)


def load_checkpoint(path: Path, config: DeepRMConfig | None = None) -> DeepRMPolicy:
    checkpoint = torch.load(path, map_location="cpu")
    policy = DeepRMPolicy(config=config)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy


def state_to_tensor(state: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(state).unsqueeze(0).float()
