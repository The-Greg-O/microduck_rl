"""Run cfg invariants (CPU, no GPU): registration, the speed curriculum's
start/end, reward-weight signs and the recipe's pose regimes, the command mix,
and that every experiment knob actually parses from the environment."""

import importlib
import math
import types

import pytest

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_run_env_cfg as run_cfg_mod
from mjlab_microduck.tasks.microduck_run_env_cfg import (
    DEFAULT_AIR_MAX,
    DEFAULT_AIR_MIN,
    DEFAULT_FORWARD_FRAC,
    DEFAULT_SPEED_CEILING,
    DEFAULT_STAGE_SCALE,
    MicroduckRunRlCfg,
    RUN_RUNNING_THRESHOLD,
    RUN_SPEED_STAGES,
    RUN_STD_RUNNING,
    RUN_STD_STANDING,
    RUN_STD_WALKING,
    RUN_WALKING_THRESHOLD,
    make_microduck_run_env_cfg,
    run_speed_stages,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import NUM_STEPS_PER_ENV


def test_registered():
    import mjlab_microduck.tasks  # noqa: F401  registers on import
    from mjlab.tasks.registry import list_tasks

    tasks = list_tasks()
    assert "Mjlab-Run-Flat-MicroDuck" in tasks
    assert "Mjlab-Run-Rough-MicroDuck" in tasks


# ── speed curriculum ─────────────────────────────────────────────────────────


def test_command_starts_at_walk_ceiling():
    cfg = make_microduck_run_env_cfg()
    ranges = cfg.commands["twist"].ranges
    assert ranges.lin_vel_x == (-0.4, 0.4)
    assert ranges.lin_vel_y == (-0.3, 0.3)
    assert ranges.ang_vel_z == (-1.0, 1.0)


def test_speed_curriculum_starts_at_04_and_ends_at_the_ceiling():
    cfg = make_microduck_run_env_cfg()
    stages = cfg.curriculum["speed_ceiling"].params["speed_stages"]
    assert cfg.curriculum["speed_ceiling"].func is microduck_mdp.speed_ceiling_curriculum
    assert cfg.curriculum["speed_ceiling"].params["command_name"] == "twist"
    assert stages[0] == {"step": 0, "ceiling": 0.4}
    assert stages[-1]["ceiling"] == DEFAULT_SPEED_CEILING == 1.1
    assert stages[-1]["step"] == 6000 * NUM_STEPS_PER_ENV
    # Monotone, and never above the ceiling.
    ceilings = [s["ceiling"] for s in stages]
    steps = [s["step"] for s in stages]
    assert ceilings == sorted(ceilings)
    assert steps == sorted(steps)
    assert max(ceilings) <= DEFAULT_SPEED_CEILING
    # Stage boundaries are env steps = iterations × 24.
    assert all(s["step"] % NUM_STEPS_PER_ENV == 0 for s in stages)


def test_speed_ceiling_knob_clips_every_stage(monkeypatch):
    monkeypatch.setenv("MICRODUCK_RUN_SPEED_CEILING", "0.8")
    cfg = make_microduck_run_env_cfg()
    stages = cfg.curriculum["speed_ceiling"].params["speed_stages"]
    assert stages[0]["ceiling"] == 0.4
    assert stages[-1]["ceiling"] == 0.8
    assert max(s["ceiling"] for s in stages) == 0.8
    # The initial command range is clipped too (0.4 <= 0.8, so unchanged here).
    assert cfg.commands["twist"].ranges.lin_vel_x == (-0.4, 0.4)


def test_speed_ceiling_below_the_first_stage_clips_the_start_too(monkeypatch):
    monkeypatch.setenv("MICRODUCK_RUN_SPEED_CEILING", "0.3")
    cfg = make_microduck_run_env_cfg()
    stages = cfg.curriculum["speed_ceiling"].params["speed_stages"]
    assert cfg.commands["twist"].ranges.lin_vel_x == (-0.4, 0.3)
    assert {s["ceiling"] for s in stages} == {0.3}


def test_stage_scale_stretches_every_schedule(monkeypatch):
    monkeypatch.setenv("MICRODUCK_RUN_STAGE_SCALE", "2.0")
    cfg = make_microduck_run_env_cfg()
    speed = cfg.curriculum["speed_ceiling"].params["speed_stages"]
    action = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    standing = cfg.curriculum["standing_envs"].params["standing_stages"]
    assert speed[-1]["step"] == 12_000 * NUM_STEPS_PER_ENV
    assert action[-1]["step"] == 6_000 * NUM_STEPS_PER_ENV
    assert standing[-1]["step"] == 3_000 * NUM_STEPS_PER_ENV
    # Values are untouched by the scale — only the boundaries move.
    assert speed[-1]["ceiling"] == 1.1
    assert action[-1]["weight"] == -0.5
    assert standing[-1]["rel_standing_envs"] == 0.10


def test_speed_ceiling_curriculum_only_moves_the_upper_bound():
    """The mdp function itself: lower bound preserved, latest stage wins."""

    ranges = types.SimpleNamespace(
        lin_vel_x=(-0.4, 0.4), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0)
    )
    term = types.SimpleNamespace(cfg=types.SimpleNamespace(ranges=ranges))
    env = types.SimpleNamespace(
        common_step_counter=0,
        command_manager=types.SimpleNamespace(get_term=lambda name: term),
    )

    stages = run_speed_stages()
    microduck_mdp.speed_ceiling_curriculum(env, None, "twist", stages)
    assert ranges.lin_vel_x == (-0.4, 0.4)

    env.common_step_counter = 2500 * NUM_STEPS_PER_ENV
    microduck_mdp.speed_ceiling_curriculum(env, None, "twist", stages)
    assert ranges.lin_vel_x == (-0.4, 0.55)

    env.common_step_counter = 99_000 * NUM_STEPS_PER_ENV
    out = microduck_mdp.speed_ceiling_curriculum(env, None, "twist", stages)
    assert ranges.lin_vel_x == (-0.4, 1.1)
    assert float(out) == pytest.approx(1.1)
    # y / yaw untouched.
    assert ranges.lin_vel_y == (-0.3, 0.3)
    assert ranges.ang_vel_z == (-1.0, 1.0)


# ── reward signs and the recipe's numbers ────────────────────────────────────

PAY_TERMS = (
    "track_linear_velocity",
    "track_angular_velocity",
    "air_time",
    "upright",
    "pose",
    "head_pose_tracking",
)
COST_TERMS = (
    "foot_clearance",
    "foot_slip",
    "foot_swing_height",
    "body_ang_vel",
    "angular_momentum",
    "self_collisions",
    "action_rate_l2",
    "dof_pos_limits",
)


def test_every_pay_positive_and_every_cost_negative():
    cfg = make_microduck_run_env_cfg()
    for name in PAY_TERMS:
        assert cfg.rewards[name].weight > 0, name
    for name in COST_TERMS:
        assert cfg.rewards[name].weight < 0, name
    # head_pose_bias is a SELF-NEGATING penalty (returns <= 0) → non-negative
    # weight; the velocity curriculum ramps it up from 0.
    assert cfg.rewards["head_pose_bias"].weight >= 0
    # body_pose stays wired at weight 0 so the 61D obs slot stays alive.
    assert cfg.rewards["body_pose_tracking"].weight == 0.0
    assert "body_command" in cfg.observations["actor"].terms


def test_recipe_weights():
    cfg = make_microduck_run_env_cfg()
    expected = {
        "track_linear_velocity": 4.0,
        "track_angular_velocity": 2.0,
        "air_time": 3.0,
        "upright": 2.0,
        "pose": 1.0,
        "head_pose_tracking": 3.5,
        "foot_clearance": -2.0,
        "foot_slip": -0.1,
        "foot_swing_height": -0.25,
        "body_ang_vel": -0.025,
        "angular_momentum": -0.01,
        "self_collisions": -1.0,
        "action_rate_l2": -0.1,
    }
    for name, weight in expected.items():
        assert cfg.rewards[name].weight == pytest.approx(weight), name
    # Speed is the largest term in the stack — the whole point of the task.
    assert cfg.rewards["track_linear_velocity"].weight == max(
        cfg.rewards[n].weight for n in cfg.rewards
    )


def test_tracking_and_upright_stds():
    cfg = make_microduck_run_env_cfg()
    assert cfg.rewards["track_linear_velocity"].params["std"] == pytest.approx(
        math.sqrt(0.1)
    )
    assert cfg.rewards["track_angular_velocity"].params["std"] == pytest.approx(
        math.sqrt(0.5)
    )
    # Lean is cheap: std² 0.12, vs the walk recipe's 0.05.
    assert cfg.rewards["upright"].params["std"] == pytest.approx(math.sqrt(0.12))
    assert cfg.rewards["head_pose_tracking"].params["std"] == 0.5


def test_pose_thresholds_and_stds():
    cfg = make_microduck_run_env_cfg()
    params = cfg.rewards["pose"].params
    assert params["walking_threshold"] == RUN_WALKING_THRESHOLD == 0.01
    assert params["running_threshold"] == RUN_RUNNING_THRESHOLD == 0.40
    assert params["std_standing"] == RUN_STD_STANDING
    assert params["std_walking"] == RUN_STD_WALKING
    assert params["std_running"] == RUN_STD_RUNNING
    # Per the recipe, in LEG_JOINT order (hip_yaw, hip_roll, hip_pitch, knee, ankle).
    assert list(RUN_STD_STANDING.values()) == [0.1, 0.05, 0.15, 0.15, 0.1]
    assert list(RUN_STD_WALKING.values()) == [0.3, 0.05, 0.4, 0.4, 0.25]
    assert list(RUN_STD_RUNNING.values()) == [0.4, 0.08, 0.6, 0.6, 0.4]
    # The running regime must be strictly looser than walking, or it is dead code.
    for key, walk in RUN_STD_WALKING.items():
        assert RUN_STD_RUNNING[key] > walk, key
    # Pose still excludes the command-driven neck/head joints.
    assert "neck" in params["asset_cfg"].joint_names[0]


def test_gait_terms_gated_on_a_live_command():
    cfg = make_microduck_run_env_cfg()
    for name in ("air_time", "foot_clearance", "foot_slip", "foot_swing_height"):
        assert cfg.rewards[name].params["command_threshold"] == 0.01, name
    assert cfg.rewards["air_time"].params["threshold_min"] == DEFAULT_AIR_MIN == 0.15
    assert cfg.rewards["air_time"].params["threshold_max"] == DEFAULT_AIR_MAX == 0.35
    assert cfg.rewards["foot_clearance"].params["target_height"] == 0.03


def test_action_rate_ramp_caps_at_half():
    cfg = make_microduck_run_env_cfg()
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert [s["weight"] for s in stages] == [-0.1, -0.2, -0.3, -0.4, -0.5]
    assert [s["step"] // NUM_STEPS_PER_ENV for s in stages] == [
        0,
        1000,
        1500,
        2250,
        3000,
    ]
    assert all(s["weight"] < 0 for s in stages)


# ── command mix ──────────────────────────────────────────────────────────────


def test_command_mix():
    cfg = make_microduck_run_env_cfg()
    command = cfg.commands["twist"]
    assert isinstance(command, microduck_mdp.VelocityCommandCommandOnlyCfg)
    assert command.rel_forward_envs == DEFAULT_FORWARD_FRAC == 0.55
    assert command.rel_turn_in_place_envs == 0.0
    assert command.rel_standing_envs == 0.02
    standing = cfg.curriculum["standing_envs"].params["standing_stages"]
    assert standing[0]["rel_standing_envs"] == 0.02
    assert standing[-1]["rel_standing_envs"] == 0.10
    assert standing[-1]["step"] == 1500 * NUM_STEPS_PER_ENV


# ── knobs ────────────────────────────────────────────────────────────────────


def test_knob_defaults_match_the_recipe():
    assert (DEFAULT_SPEED_CEILING, DEFAULT_STAGE_SCALE) == (1.1, 1.0)
    assert (DEFAULT_FORWARD_FRAC, DEFAULT_AIR_MIN, DEFAULT_AIR_MAX) == (
        0.55,
        0.15,
        0.35,
    )
    assert RUN_SPEED_STAGES[0] == (0, 0.4)
    assert RUN_SPEED_STAGES[-1] == (6000, 1.1)


def test_every_knob_parses_from_the_environment(monkeypatch):
    monkeypatch.setenv("MICRODUCK_RUN_FORWARD_FRAC", "0.75")
    monkeypatch.setenv("MICRODUCK_RUN_AIR_MIN", "0.1")
    monkeypatch.setenv("MICRODUCK_RUN_AIR_MAX", "0.4")
    cfg = make_microduck_run_env_cfg()
    assert cfg.commands["twist"].rel_forward_envs == 0.75
    assert cfg.rewards["air_time"].params["threshold_min"] == 0.1
    assert cfg.rewards["air_time"].params["threshold_max"] == 0.4


def test_module_constants_resolve_from_the_environment(monkeypatch):
    """The bare module constants are the import-time snapshot of the knobs."""
    monkeypatch.setenv("MICRODUCK_RUN_SPEED_CEILING", "0.9")
    monkeypatch.setenv("MICRODUCK_RUN_STAGE_SCALE", "1.5")
    reloaded = importlib.reload(run_cfg_mod)
    try:
        assert reloaded.SPEED_CEILING == 0.9
        assert reloaded.STAGE_SCALE == 1.5
        assert reloaded.run_speed_stages()[-1]["ceiling"] == 0.9
    finally:
        monkeypatch.undo()
        importlib.reload(run_cfg_mod)


def test_bad_knob_is_a_hard_error(monkeypatch):
    monkeypatch.setenv("MICRODUCK_RUN_SPEED_CEILING", "fast")
    with pytest.raises(ValueError):
        make_microduck_run_env_cfg()


def test_empty_knob_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("MICRODUCK_RUN_SPEED_CEILING", "")
    cfg = make_microduck_run_env_cfg()
    assert cfg.curriculum["speed_ceiling"].params["speed_stages"][-1]["ceiling"] == 1.1


# ── runner cfg ───────────────────────────────────────────────────────────────


def test_runner_cfg():
    assert MicroduckRunRlCfg.experiment_name == "run"
    assert MicroduckRunRlCfg.algorithm.symmetry_cfg is not None
    assert MicroduckRunRlCfg.actor.obs_normalization is True
    assert MicroduckRunRlCfg.critic.obs_normalization is True
    assert MicroduckRunRlCfg.max_iterations == 8_000
    assert MicroduckRunRlCfg.num_steps_per_env == NUM_STEPS_PER_ENV == 24
