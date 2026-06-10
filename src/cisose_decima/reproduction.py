"""Reproduction command records for the official Decima simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cisose_decima.config import DEFAULT_CONFIG, DecimaConfig


@dataclass(frozen=True)
class DecimaReferenceCommands:
    train: tuple[str, ...]
    test: tuple[str, ...]
    notes: tuple[str, ...]


def official_reference_commands(config: DecimaConfig = DEFAULT_CONFIG) -> DecimaReferenceCommands:
    train = (
        "python3",
        "train.py",
        "--exec_cap",
        str(config.exec_cap),
        "--num_init_dags",
        str(config.num_init_dags),
        "--num_stream_dags",
        str(config.train_num_stream_dags),
        "--reset_prob",
        str(config.reset_prob),
        "--reset_prob_min",
        str(config.reset_prob_min),
        "--reset_prob_decay",
        str(config.reset_prob_decay),
        "--diff_reward_enabled",
        str(config.diff_reward_enabled),
        "--num_agents",
        str(config.num_agents),
        "--model_save_interval",
        str(config.model_save_interval),
        "--model_folder",
        config.model_folder,
    )
    test = (
        "python3",
        "test.py",
        "--exec_cap",
        str(config.exec_cap),
        "--num_init_dags",
        str(config.num_init_dags),
        "--num_stream_dags",
        str(config.test_num_stream_dags),
        "--canvs_visualization",
        "0",
        "--test_schemes",
        "dynamic_partition",
        "learn",
        "--num_exp",
        "1",
        "--saved_model",
        f"{config.model_folder}model_ep_{config.reference_model_epoch}",
    )
    return DecimaReferenceCommands(
        train=train,
        test=test,
        notes=(
            "Commands mirror the official decima-sim README reference behavior.",
            "Do not run perturbations until this reproduction gate passes.",
            "Graphene validation is separate from the README learn-vs-dynamic_partition gate.",
        ),
    )


def reference_command_payload(config: DecimaConfig = DEFAULT_CONFIG) -> dict[str, object]:
    commands = official_reference_commands(config)
    return {
        "config": asdict(config),
        "train_command": list(commands.train),
        "test_command": list(commands.test),
        "notes": list(commands.notes),
    }
