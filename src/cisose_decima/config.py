"""Official Decima simulator configuration anchors."""

from __future__ import annotations

from dataclasses import dataclass


DECIMA_REPO_URL = "https://github.com/hongzimao/decima-sim"
DECIMA_COMMIT = "c010dd74ff4b7566bd0ac989c90a32cfbc630d84"


@dataclass(frozen=True)
class DecimaConfig:
    exec_cap: int = 50
    num_init_dags: int = 1
    train_num_stream_dags: int = 200
    test_num_stream_dags: int = 5000
    stream_interval_ms: int = 25_000
    reset_prob: float = 5e-7
    reset_prob_min: float = 5e-8
    reset_prob_decay: float = 4e-10
    diff_reward_enabled: int = 1
    num_agents: int = 16
    model_save_interval: int = 100
    model_folder: str = "./models/stream_200_job_diff_reward_reset_5e-7_5e-8/"
    reference_model_epoch: int = 10_000
    node_input_dim: int = 5
    job_input_dim: int = 3
    hidden_dims: tuple[int, int] = (16, 8)
    actor_hidden_dims: tuple[int, int, int] = (32, 16, 8)
    output_dim: int = 8
    max_depth: int = 8
    executor_data_points: tuple[int, ...] = (5, 10, 20, 40, 50, 60, 80, 100)
    tpch_sizes: tuple[str, ...] = ("2g", "5g", "10g", "20g", "50g", "80g", "100g")
    tpch_num_queries: int = 22

    @property
    def hid_dims(self) -> tuple[int, int]:
        return self.hidden_dims

    @property
    def actor_executor_levels(self) -> tuple[int, ...]:
        """Official actor action levels used by Decima train/test.

        The coarse executor data points above are for task-duration lookup in
        the simulator. The actor itself chooses an executor limit from every
        integer level from 1 through `exec_cap`, as in the official
        `ActorAgent(..., range(1, args.exec_cap + 1))` calls.
        """

        return tuple(range(1, self.exec_cap + 1))


DEFAULT_CONFIG = DecimaConfig()
