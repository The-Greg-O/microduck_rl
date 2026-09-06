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
    DEFAULT_LATERAL_FRAC,
    DEFAULT_REVERSE_FRAC,
    DEFAULT_STOP_POSE_W,
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


def test_ladder_keeps_climbing_above_the_last_fixed_rung():
    from mjlab_microduck.tasks.microduck_run_env_cfg import run_speed_stages, RUN_SPEED_STAGES
    stages = run_speed_stages(ceiling=1.5, scale=1.0)
    values = [s["ceiling"] for s in stages]
    assert values[-1] == 1.5 and len(stages) == len(RUN_SPEED_STAGES) + 4
    steps = [s["step"] for s in stages]
    assert steps[-1] - steps[-2] == 500 * 24 or steps[-1] - steps[-2] == 500
    # A lower ceiling still clips in place.
    assert run_speed_stages(ceiling=0.8, scale=1.0)[-1]["ceiling"] == 0.8


def test_turn_in_place_bucket_is_a_knob(monkeypatch):
    from mjlab_microduck.tasks.microduck_run_env_cfg import make_microduck_run_env_cfg
    assert make_microduck_run_env_cfg().commands["twist"].rel_turn_in_place_envs == 0.0
    monkeypatch.setenv("MICRODUCK_RUN_TURN_FRAC", "0.15")
    assert make_microduck_run_env_cfg().commands["twist"].rel_turn_in_place_envs == 0.15


def test_yaw_tracking_weight_is_a_knob(monkeypatch):
    from mjlab_microduck.tasks.microduck_run_env_cfg import make_microduck_run_env_cfg
    assert make_microduck_run_env_cfg().rewards["track_angular_velocity"].weight == 2.0
    monkeypatch.setenv("MICRODUCK_RUN_TRACK_ANG_W", "4.0")
    assert make_microduck_run_env_cfg().rewards["track_angular_velocity"].weight == 4.0


# ── round seven: the manoeuvre buckets ───────────────────────────────────────
#
# docs/research/runner-room-falls.md, Recommendation (a) changes 3 and 4: the
# reverse (the brain's obstacle reaction) is the command under 57 of the two
# office days' 149 falls and was never trained; the lateral floor is the brain's
# real turn in place and the only command in either day with zero falls.

import torch  # noqa: E402
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand  # noqa: E402


class _BucketStub(microduck_mdp.VelocityCommandCommandOnly):
    """A resample harness: real class, real methods, no env/scene/sim."""

    device = "cpu"  # CommandTerm makes this a read-only property

    def __init__(self, cfg, n):  # noqa: D107 - deliberately skips CommandTerm.__init__
        self.cfg = cfg
        self.vel_command_b = torch.zeros(n, 3)
        self.vel_command_w = torch.zeros(n, 3)
        self.is_standing_env = torch.ones(n, dtype=torch.bool)
        self.is_forward_env = torch.ones(n, dtype=torch.bool)


def _resample(monkeypatch, n=20_000, **kwargs):
    """Run one full resample of `n` envs with the base draw stubbed out."""
    cfg = microduck_mdp.VelocityCommandCommandOnlyCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        ranges=microduck_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.9), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0)
        ),
        **kwargs,
    )
    monkeypatch.setattr(
        UniformVelocityCommand, "_resample_command", lambda self, env_ids: None
    )
    stub = _BucketStub(cfg, n)
    torch.manual_seed(0)
    stub._resample_command(torch.arange(n))
    return stub


def test_reverse_bucket_samples_the_measured_obstacle_reaction(monkeypatch):
    stub = _resample(monkeypatch, rel_reverse_envs=1.0)
    vx, vy, wz = stub.vel_command_b.unbind(dim=-1)
    assert vx.min() >= -0.5 and vx.max() <= -0.3, "vx ∈ [-0.5, -0.3]"
    assert (vy == 0.0).all(), "no lateral component in a back-out"
    assert wz.abs().max() <= 0.3, "|wz| <= 0.3 (reverse_yaw_max)"
    # Both signs of yaw, and the range is actually spanned (not a constant).
    assert (wz > 0.1).any() and (wz < -0.1).any()
    assert vx.min() < -0.45 and vx.max() > -0.35


def test_lateral_bucket_samples_the_measured_lateral_floor(monkeypatch):
    stub = _resample(monkeypatch, rel_lateral_envs=1.0)
    vx, vy, wz = stub.vel_command_b.unbind(dim=-1)
    assert vx.abs().max() <= 0.1, "vx ∈ [-0.1, 0.1]"
    assert 0.2 <= vy.abs().min() and vy.abs().max() <= 0.3, "|vy| ∈ [0.2, 0.3]"
    assert 0.4 <= wz.abs().min() and wz.abs().max() <= 1.0, "|wz| ∈ [0.4, 1.0]"
    for signed in (vy, wz):
        frac = float((signed > 0).float().mean())
        assert 0.45 < frac < 0.55, "both sides are trained, roughly evenly"


def test_a_bucket_env_is_never_standing_or_forward(monkeypatch):
    """The base template's two independent draws must not erase a bucket.

    `_update_command` zeroes a standing env's command every step, and the
    forward-only draw rewrites vx/vy/wz — either would silently undo the
    manoeuvre this env was picked to practise.
    """
    for kwargs in (
        {"rel_reverse_envs": 1.0},
        {"rel_lateral_envs": 1.0},
        {"rel_turn_in_place_envs": 1.0},
    ):
        stub = _resample(monkeypatch, n=2048, **kwargs)
        assert not stub.is_standing_env.any(), kwargs
        assert not stub.is_forward_env.any(), kwargs
        torch.testing.assert_close(stub.vel_command_w, stub.vel_command_b)


def test_buckets_are_mutually_exclusive_and_hit_their_fractions(monkeypatch):
    """One uniform draw partitions all three, so no env is in two buckets."""
    stub = _resample(
        monkeypatch,
        n=200_000,
        rel_turn_in_place_envs=0.22,
        rel_reverse_envs=0.10,
        rel_lateral_envs=0.10,
    )
    vx, vy, wz = stub.vel_command_b.unbind(dim=-1)
    is_turn = (vx == 0.0) & (vy == 0.0) & (wz.abs() >= 0.4)
    is_rev = (vx <= -0.3) & (vx >= -0.5) & (vy == 0.0) & (wz.abs() <= 0.3)
    is_lat = (vy.abs() >= 0.2) & (vy.abs() <= 0.3) & (wz.abs() >= 0.4)
    untouched = (vx == 0.0) & (vy == 0.0) & (wz == 0.0)  # never resampled

    # No env satisfies two bucket signatures at once.
    assert (is_turn.int() + is_rev.int() + is_lat.int()).max() <= 1
    n = len(vx)
    assert abs(float(is_turn.float().mean()) - 0.22) < 0.01
    assert abs(float(is_rev.float().mean()) - 0.10) < 0.01
    assert abs(float(is_lat.float().mean()) - 0.10) < 0.01
    # The remaining 58% keeps whatever the base draw gave it (here: nothing).
    assert abs(float(untouched.float().mean()) - 0.58) < 0.01
    assert n == 200_000


def test_bucket_fractions_are_validated():
    cfg = microduck_mdp.VelocityCommandCommandOnlyCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        ranges=microduck_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.9), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0)
        ),
        rel_turn_in_place_envs=0.22,
        rel_reverse_envs=0.10,
        rel_lateral_envs=0.10,
    )
    assert cfg.validate_bucket_fractions() == pytest.approx(0.42)
    # The whole resample may be spoken for, but not more than the whole.
    cfg.rel_turn_in_place_envs = 0.80
    assert cfg.validate_bucket_fractions() == pytest.approx(1.0)
    cfg.rel_lateral_envs = 0.11
    with pytest.raises(ValueError, match="sum to <= 1.0"):
        cfg.validate_bucket_fractions()
    cfg.rel_lateral_envs = -0.1
    with pytest.raises(ValueError, match="must be >= 0"):
        cfg.validate_bucket_fractions()


def test_over_subscribed_buckets_fail_at_cfg_build(monkeypatch):
    """A bad mix must die on CPU, not 40 minutes into a paid GPU job."""
    monkeypatch.setenv("MICRODUCK_RUN_TURN_FRAC", "0.5")
    monkeypatch.setenv("MICRODUCK_RUN_REVERSE_FRAC", "0.4")
    monkeypatch.setenv("MICRODUCK_RUN_LATERAL_FRAC", "0.4")
    with pytest.raises(ValueError, match="sum to <= 1.0"):
        make_microduck_run_env_cfg()


def test_reverse_and_lateral_are_knobs_defaulting_off(monkeypatch):
    command = make_microduck_run_env_cfg().commands["twist"]
    assert (DEFAULT_REVERSE_FRAC, DEFAULT_LATERAL_FRAC) == (0.0, 0.0)
    assert command.rel_reverse_envs == 0.0
    assert command.rel_lateral_envs == 0.0
    monkeypatch.setenv("MICRODUCK_RUN_REVERSE_FRAC", "0.10")
    monkeypatch.setenv("MICRODUCK_RUN_LATERAL_FRAC", "0.10")
    command = make_microduck_run_env_cfg().commands["twist"]
    assert command.rel_reverse_envs == 0.10
    assert command.rel_lateral_envs == 0.10


def test_bucket_ranges_are_the_measured_office_commands():
    cfg = make_microduck_run_env_cfg().commands["twist"]
    assert cfg.reverse_lin_vel_x == (-0.5, -0.3)
    assert cfg.reverse_ang_vel_z == (-0.3, 0.3)
    assert cfg.lateral_lin_vel_x == (-0.1, 0.1)
    assert cfg.lateral_lin_vel_y_mag == (0.2, 0.3)
    assert cfg.lateral_ang_vel_z_mag == (0.4, 1.0)


# ── round seven: the stop-from-speed pose ────────────────────────────────────


def _stop_pose_env(n, joint_pos, root_z, command):
    """Minimal env for `stop_pose_reward`: 14 servo joints, HOME as default."""
    home = torch.zeros(n, 14)
    home[:, 2] = -0.4579   # left_hip_pitch
    home[:, 11] = 0.4579   # right_hip_pitch
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_pos=joint_pos,
            default_joint_pos=home,
            root_link_pos_w=torch.stack(
                [torch.zeros(n), torch.zeros(n), root_z], dim=-1
            ),
        ),
        find_joints=lambda pattern: (list(range(14)), [f"j{i}" for i in range(14)]),
    )

    class _Scene(dict):
        terrain = types.SimpleNamespace(env_origins=torch.zeros(n, 3))

    return types.SimpleNamespace(
        scene=_Scene(robot=asset),
        command_manager=types.SimpleNamespace(get_command=lambda name: command),
    )


def _stop_pose(env):
    return microduck_mdp.stop_pose_reward(env, command_name="twist")


def test_stop_pose_pays_only_at_a_zero_command():
    n = 4
    home = torch.zeros(n, 14)
    home[:, 2], home[:, 11] = -0.4579, 0.4579
    z = torch.full((n,), microduck_mdp.RUN_STAND_Z)
    # env 0: stopped. envs 1-3: a live vx / vy / wz.
    command = torch.zeros(n, 3)
    command[1, 0], command[2, 1], command[3, 2] = 0.3, 0.2, 0.8
    r = _stop_pose(_stop_pose_env(n, home.clone(), z, command))
    assert r[0] == pytest.approx(1.0, abs=1e-5), "at HOME + STAND_Z, stopped → 1"
    assert (r[1:] == 0.0).all(), "a live command pays nothing, however good the pose"


def test_stop_pose_peaks_at_home_and_stand_height():
    n = 5
    home = torch.zeros(n, 14)
    home[:, 2], home[:, 11] = -0.4579, 0.4579
    command = torch.zeros(n, 3)
    z = torch.full((n,), microduck_mdp.RUN_STAND_Z)
    joints = home.clone()
    # 0: exactly HOME. 1: a 0.15 rad crouch on both knees. 2: a deep crouch.
    joints[1, 3] += 0.15
    joints[1, 12] -= 0.15
    joints[2, 3] += 0.6
    joints[2, 12] -= 0.6
    # 3: HOME pose but 2 cm low (a squat the stander cannot inherit).
    z[3] -= 0.02
    # 4: HOME pose, 1 cm low.
    z[4] -= 0.01
    r = _stop_pose(_stop_pose_env(n, joints, z, command))
    assert r[0] == pytest.approx(1.0, abs=1e-5)
    assert r.argmax() == 0, "HOME at STAND height is the unique peak"
    assert r[1] < r[0] and r[2] < r[1], "the deeper the crouch, the less it pays"
    assert r[3] < r[4] < r[0], "a trunk off STAND height is priced monotonically"
    assert r[3] < 0.02, "2 cm low is ~2 stds out — effectively nothing"
    assert (r >= 0.0).all()


def test_stop_pose_ignores_the_head_joints():
    """The head is command-driven (head_pose_tracking); pulling it to HOME here
    would fight that term, exactly as the leg-only `pose` reward avoids."""
    n = 2
    home = torch.zeros(n, 14)
    home[:, 2], home[:, 11] = -0.4579, 0.4579
    joints = home.clone()
    joints[1, 5:9] = 0.8  # neck_pitch, head_pitch, head_yaw, head_roll
    z = torch.full((n,), microduck_mdp.RUN_STAND_Z)
    r = _stop_pose(_stop_pose_env(n, joints, z, torch.zeros(n, 3)))
    assert r[0] == pytest.approx(float(r[1]), abs=1e-6)
    assert microduck_mdp.RUN_LEG_JOINT_INDICES == (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)


def test_stop_pose_is_a_knob_defaulting_off(monkeypatch):
    cfg = make_microduck_run_env_cfg()
    assert DEFAULT_STOP_POSE_W == 0.0
    term = cfg.rewards["stop_pose"]
    assert term.func is microduck_mdp.stop_pose_reward
    assert term.weight == 0.0, "registered but OFF — earlier runs reproduce"
    assert term.params["pose_std"] == 0.15
    assert term.params["height_std"] == 0.01
    assert term.params["target_height"] == microduck_mdp.RUN_STAND_Z == 0.115
    monkeypatch.setenv("MICRODUCK_RUN_STOP_POSE_W", "2.0")
    assert make_microduck_run_env_cfg().rewards["stop_pose"].weight == 2.0


def test_stop_pose_is_a_positive_reward_not_a_penalty():
    """Sign convention (AGENTS.md): stop_pose_reward returns 0..1, so its
    weight must be POSITIVE — a negative one would pay for NOT standing."""
    monkeypatch_free = make_microduck_run_env_cfg().rewards["stop_pose"]
    assert monkeypatch_free.weight >= 0.0
