from pathlib import Path

import torch

from cisose_decima.config import DEFAULT_CONFIG
from cisose_decima.gates import current_readiness
from cisose_decima.graphene import GrapheneStyleComparator
from cisose_decima.model import DecimaPolicy, fgsm_node_features, parameter_count
from cisose_decima.official import configure_official_simulator, run_official_episode, translate_state
from cisose_decima.preflight import decima_preflight_report
from cisose_decima.reproduction import reference_command_payload
from cisose_decima.rollout import build_template_observation, rollout_smoke, select_greedy_action
from cisose_decima.sampling import sampling_probabilities
from cisose_decima.trainer import train_on_rollout
from cisose_decima.training import AveragePerStepReward, actor_loss, discount
from cisose_decima.tpch import TpchDagTemplate, load_tpch_templates, template_summary


def test_official_decima_reference_config_is_locked():
    assert DEFAULT_CONFIG.exec_cap == 50
    assert DEFAULT_CONFIG.num_init_dags == 1
    assert DEFAULT_CONFIG.train_num_stream_dags == 200
    assert DEFAULT_CONFIG.test_num_stream_dags == 5000
    assert DEFAULT_CONFIG.model_folder == "./models/stream_200_job_diff_reward_reset_5e-7_5e-8/"
    assert DEFAULT_CONFIG.reference_model_epoch == 10_000
    assert DEFAULT_CONFIG.executor_data_points == (5, 10, 20, 40, 50, 60, 80, 100)
    assert DEFAULT_CONFIG.actor_executor_levels == tuple(range(1, 51))


def test_tpch_template_loader_reads_official_pool():
    templates = load_tpch_templates(Path("external/decima-sim/spark_env/tpch"))
    summary = template_summary(templates)
    assert summary["count"] == 154
    assert summary["max_nodes"] >= summary["mean_nodes"] > 0
    assert summary["max_edges"] >= summary["mean_edges"] > 0
    assert summary["max_work"] >= summary["mean_work"] > 0
    assert templates[0].task_counts is not None
    assert templates[0].node_work is not None


def test_alibaba_style_weighting_favors_larger_tpch_dags():
    templates = load_tpch_templates(Path("external/decima-sim/spark_env/tpch"))
    p0 = sampling_probabilities(templates, w=0.0)
    p9 = sampling_probabilities(templates, w=0.9)
    scores = torch.tensor([template.size_score for template in templates], dtype=torch.float64)
    mean0 = float((torch.tensor(p0) * scores).sum())
    mean9 = float((torch.tensor(p9) * scores).sum())
    assert abs(float(p0.sum()) - 1.0) < 1e-12
    assert abs(float(p9.sum()) - 1.0) < 1e-12
    assert mean9 > mean0


def test_decima_policy_forward_and_fgsm_shapes():
    torch.manual_seed(1)
    templates = load_tpch_templates(Path("external/decima-sim/spark_env/tpch"))
    template = templates[0]
    policy = DecimaPolicy()
    features = torch.rand(template.num_nodes, DEFAULT_CONFIG.node_input_dim)
    adjacency = torch.as_tensor(template.adjacency, dtype=torch.float32)
    probs = policy(features, adjacency)
    assert probs.shape == (template.num_nodes,)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    output = policy.predict(features, adjacency)
    assert output.job_probs.shape == (1, len(DEFAULT_CONFIG.actor_executor_levels))
    assert torch.isclose(output.job_probs.sum(), torch.tensor(1.0), atol=1e-6)
    assert parameter_count(policy) > 0
    perturbed = fgsm_node_features(policy, features, adjacency, epsilon=0.05)
    assert perturbed.shape == features.shape
    assert torch.all(perturbed >= 0)


def test_decima_fgsm_keeps_categorical_source_flag_fixed():
    torch.manual_seed(2)
    template = load_tpch_templates(Path("external/decima-sim/spark_env/tpch"))[0]
    policy = DecimaPolicy()
    features = torch.rand(template.num_nodes, DEFAULT_CONFIG.node_input_dim)
    features[:, 1] = torch.tensor([2.0 if idx % 2 == 0 else -2.0 for idx in range(template.num_nodes)])
    adjacency = torch.as_tensor(template.adjacency, dtype=torch.float32)
    perturbed = fgsm_node_features(policy, features, adjacency, epsilon=0.10)
    assert torch.equal(perturbed[:, 1], features[:, 1])
    assert torch.all(perturbed[:, 0] >= 0)
    assert torch.all(perturbed[:, 2] >= 0)


def test_decima_actor_uses_source_architecture_dimensions():
    policy = DecimaPolicy()
    assert policy.executor_levels == tuple(range(1, DEFAULT_CONFIG.exec_cap + 1))
    assert policy.gcn.prepare.net[0].in_features == DEFAULT_CONFIG.node_input_dim
    assert policy.gcn.prepare.net[0].out_features == 16
    assert policy.gcn.prepare.net[2].out_features == 8
    assert policy.gcn.prepare.net[4].out_features == DEFAULT_CONFIG.output_dim
    assert policy.node_score.net[0].out_features == 32
    assert policy.node_score.net[2].out_features == 16
    assert policy.node_score.net[4].out_features == 8


def test_decima_readiness_gates_are_explicitly_closed_before_reproduction():
    readiness = current_readiness()
    assert not readiness.ready_for_perturbations
    assert readiness.official_readme_reproduction.details["status"] == "not_run"
    assert readiness.graphene_validation.details["status"] == "not_run"
    assert readiness.graphene_validation.details["current_scaffold"] == "GrapheneStyleComparator"
    assert readiness.graphene_validation.details["paper_evidence_allowed"] is False


def test_decima_training_primitives_match_expected_shapes_and_signs():
    returns = discount([1.0, 2.0, 3.0], gamma=1.0)
    assert returns.tolist() == [6.0, 5.0, 3.0]
    avg = AveragePerStepReward(size=4)
    avg.add_list_filter_zero([2.0, 0.0, 4.0], [2.0, 0.0, 2.0])
    assert avg.get_avg_per_step_reward() == 1.5
    node_probs = torch.tensor([0.25, 0.75], requires_grad=True)
    job_probs = torch.tensor([[0.4, 0.6]], requires_grad=True)
    loss, parts = actor_loss(
        node_probs=node_probs,
        job_probs=job_probs,
        node_action=1,
        job_index=0,
        executor_level_index=1,
        advantage=torch.tensor(2.0),
        entropy_weight=0.1,
    )
    assert loss.ndim == 0
    assert parts["selected_node_prob"] > 0


def test_decima_reference_commands_record_official_readme_gate():
    payload = reference_command_payload()
    assert payload["train_command"][0:2] == ["python3", "train.py"]
    assert "--exec_cap" in payload["train_command"]
    assert "--model_folder" in payload["train_command"]
    assert "./models/stream_200_job_diff_reward_reset_5e-7_5e-8/" in payload["train_command"]
    assert payload["test_command"][0:2] == ["python3", "test.py"]
    assert "--canvs_visualization" in payload["test_command"]
    assert "--num_exp" in payload["test_command"]
    assert "dynamic_partition" in payload["test_command"]
    assert "learn" in payload["test_command"]
    assert "./models/stream_200_job_diff_reward_reset_5e-7_5e-8/model_ep_10000" in payload["test_command"]


def test_decima_rollout_observation_matches_actor_interface():
    torch.manual_seed(7)
    templates = load_tpch_templates(Path("external/decima-sim/spark_env/tpch"))
    policy = DecimaPolicy()
    observation = build_template_observation(templates[0])
    assert observation.node_features.shape == (templates[0].num_nodes, DEFAULT_CONFIG.node_input_dim)
    assert observation.job_features.shape == (1, DEFAULT_CONFIG.job_input_dim)
    assert observation.node_valid_mask.sum() >= 1
    action = select_greedy_action(policy, observation)
    assert action.executor_level in DEFAULT_CONFIG.actor_executor_levels
    assert observation.node_valid_mask[action.node_index] == 1
    rollout = rollout_smoke(policy, templates, count=3)
    assert len(rollout.steps) == 3
    assert rollout.total_reward < 0


def test_decima_trainer_updates_from_rollout_contract():
    torch.manual_seed(9)
    templates = load_tpch_templates(Path("external/decima-sim/spark_env/tpch"))
    policy = DecimaPolicy()
    rollout = rollout_smoke(policy, templates, count=4)
    before = [param.detach().clone() for param in policy.parameters()]
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    result = train_on_rollout(policy, optimizer, rollout, gamma=1.0, entropy_weight=0.01)
    assert result.num_steps == 4
    assert result.total_reward < 0
    assert any(not torch.equal(old, new) for old, new in zip(before, policy.parameters(), strict=True))


def test_graphene_comparator_respects_dependencies_and_downstream_work():
    template = TpchDagTemplate(
        size="unit",
        query_id=1,
        adjacency=torch.tensor(
            [
                [0, 1, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=torch.float32,
        ).numpy(),
        task_counts=(1, 1, 1, 1),
        node_work=(1.0, 10.0, 1.0, 1.0),
        node_duration=(1.0, 10.0, 1.0, 1.0),
    )
    comparator = GrapheneStyleComparator()
    schedule = comparator.preferred_schedule(template)
    assert schedule.node_order[0] == 0
    assert schedule.node_order[1] == 1
    assert schedule.node_order[-1] == 3
    assert comparator.choose_node(template, (1, 2)) == 1


def test_decima_preflight_records_graphene_reference_gap():
    report = decima_preflight_report(Path.cwd())
    assert report["official_readme_reference"]["passed"] is True
    assert report["graphene_reference"]["multi_resource_test_references_graphene"] is True
    assert report["graphene_reference"]["passed"] is False
    assert report["graphene_reference"]["protocol_guardrail"] == "do_not_call_local_scaffold_faithful_graphene_before_validation"
    assert report["allowed_now"]["decima_perturbation_cells"] is False
    assert report["pre_execution_blockers_remaining"] == []
    assert report["pre_execution_guardrails"]


def test_official_simulator_dynamic_partition_smoke():
    result = run_official_episode(
        Path.cwd(),
        scheme="dynamic_partition",
        seed=123,
        num_stream_dags=1,
    )
    assert result.num_finished_jobs == DEFAULT_CONFIG.num_init_dags + 1
    assert result.decisions > 0
    assert result.total_reward < 0
    assert result.mean_job_completion_time > 0


def test_official_pytorch_state_translation_uses_official_action_space():
    modules = configure_official_simulator(
        Path.cwd(),
        seed=321,
        num_stream_dags=1,
    )
    env = modules.environment()
    env.seed(321)
    env.reset()
    state = translate_state(env.observe())
    assert state.node_features.shape[1] == DEFAULT_CONFIG.node_input_dim
    assert state.job_features.shape[1] == DEFAULT_CONFIG.job_input_dim
    assert state.job_valid_mask.shape[1] == DEFAULT_CONFIG.exec_cap
    assert len(state.adj_mats) == DEFAULT_CONFIG.max_depth
    assert len(state.masks) == DEFAULT_CONFIG.max_depth
