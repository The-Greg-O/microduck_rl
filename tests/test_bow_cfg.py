"""Bow cfg invariants (CPU, no GPU): registration, the command is pinned, the
episode and its phase boundaries, every cost carries a negative weight, the
measured head/pitch SIGNS, and the per-env phase memory on a fake env."""

import math
import types

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_bow_env_cfg import (
    BOW_COLLAPSE_Z,
    BOW_PITCH_RAD,
    BOW_Z,
    DIP_END_S,
    EPISODE_LENGTH_S,
    HOLD_END_S,
    RISE_END_S,
    STAND_END_S,
    STAND_Z,
    MicroduckBowRlCfg,
    make_microduck_bow_env_cfg,
)


def test_registered():
    import mjlab_microduck.tasks  # noqa: F401  registers on import
    from mjlab.tasks.registry import list_tasks
    assert "Mjlab-Bow-Flat-MicroDuck" in list_tasks()


def test_command_is_pinned_and_episode_short():
    cfg = make_microduck_bow_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.VelocityCommandCommandOnlyCfg)
    assert cmd.ranges.lin_vel_x == (-0.01, 0.01)
    assert cmd.ranges.lin_vel_y == (-0.01, 0.01)
    assert cmd.ranges.ang_vel_z == (-0.05, 0.05)
    assert cmd.rel_standing_envs == 0.0 and cmd.heading_command is False
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 4.0
    # The head/body command slots stay alive (61D obs contract) even though the
    # bow drives the head itself.
    assert "head_pose" in cfg.commands and "body_pose" in cfg.commands
    assert cfg.rewards["head_pose_tracking"].weight == 0.0
    for gone in ("standing_envs", "head_pose_range", "action_rate_weight"):
        assert gone not in cfg.curriculum, gone


def test_phase_boundaries_partition_the_episode():
    assert (DIP_END_S, HOLD_END_S, RISE_END_S, STAND_END_S) == (1.0, 2.0, 3.0, 4.0)
    assert STAND_END_S == EPISODE_LENGTH_S
    assert 0.0 < DIP_END_S < HOLD_END_S < RISE_END_S <= STAND_END_S
    # The crouch is a 3 cm dip and the collapse floor is well below it.
    assert math.isclose(STAND_Z - BOW_Z, 0.030, abs_tol=1e-9)
    assert BOW_COLLAPSE_Z < BOW_Z - 0.02


def test_gait_terms_gone_and_bow_terms_signed():
    cfg = make_microduck_bow_env_cfg()
    for gone in ("air_time", "foot_clearance", "foot_swing_height", "pose",
                 "upright", "head_pose_bias"):
        assert gone not in cfg.rewards, gone
    for pos in ("bow_height", "bow_pitch", "bow_head", "bow_upright",
                "stand_pose", "track_linear_velocity", "track_angular_velocity"):
        assert cfg.rewards[pos].weight > 0, pos
    for cost in ("foot_lift", "drift", "foot_slip", "action_rate_l2",
                 "dof_pos_limits", "self_collisions"):
        assert cfg.rewards[cost].weight < 0, cost
    # Feet planted is the hardest constraint of the trick.
    assert cfg.rewards["foot_lift"].weight <= -3.0
    assert cfg.rewards["foot_lift"].weight < cfg.rewards["drift"].weight
    # foot_slip must not stay command-gated at a pinned ~zero command.
    assert cfg.rewards["foot_slip"].params["command_threshold"] < 0.0
    # action_rate stays light: no ramp to -1.0 over a 4 s trick.
    assert cfg.rewards["action_rate_l2"].weight >= -0.5


def test_collapse_termination_added():
    cfg = make_microduck_bow_env_cfg()
    term = cfg.terminations["collapsed"]
    assert term.func is microduck_mdp.root_height_below
    assert term.params["min_height"] == BOW_COLLAPSE_Z
    assert term.time_out is False
    assert "fell_over" in cfg.terminations   # base fall check kept


def test_symmetry_on_and_obs_normalized():
    assert MicroduckBowRlCfg.algorithm.symmetry_cfg is not None
    assert MicroduckBowRlCfg.actor.obs_normalization is True
    assert MicroduckBowRlCfg.critic.obs_normalization is True
    assert MicroduckBowRlCfg.experiment_name == "bow"
    assert MicroduckBowRlCfg.max_iterations == 2_000


def test_measured_signs_are_not_flipped():
    """FK on robot_walk.xml (see the cfg docstring): head DOWN is neck_pitch
    NEGATIVE / head_pitch POSITIVE, and nose-down trunk pitch is negative in
    the asin(R[2,0]) convention. These are the two signs a rewrite gets wrong."""
    down_neck, down_head = microduck_mdp.BOW_HEAD_DOWN
    up_neck, up_head = microduck_mdp.BOW_HEAD_UP
    assert down_neck < 0.0 < down_head
    assert up_head < 0.0 < up_neck
    assert BOW_PITCH_RAD > 0.0            # magnitude; the target is its negative
    q = torch.tensor([[math.cos(BOW_PITCH_RAD / 2), 0.0,
                       math.sin(BOW_PITCH_RAD / 2), 0.0]])   # nose-down by +y
    asset = types.SimpleNamespace(data=types.SimpleNamespace(root_link_quat_w=q))
    assert float(microduck_mdp._bow_nose_up(asset)[0]) < 0.0


# ── the phase memory on a fake env ───────────────────────────────────────────

class _FakeSensor:
    def __init__(self, n):
        self.data = types.SimpleNamespace(found=torch.ones(n, 2))


class _FakeScene:
    def __init__(self, n, sensor, asset):
        self.sensors = {"feet_ground_contact": sensor}
        self.terrain = types.SimpleNamespace(env_origins=torch.zeros(n, 3))
        self._asset = asset

    def __getitem__(self, key):      # env.scene["robot"]
        return self._asset


class _FakeEnv:
    def __init__(self, n=1):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = 0.02
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        sensor = _FakeSensor(n)
        asset = types.SimpleNamespace(
            # 14 servos, no passive_* joints: _servo_joint_ids is the identity.
            find_joints=lambda pattern: (list(range(14)), []),
            data=types.SimpleNamespace(
                root_link_pos_w=torch.tensor([[0.0, 0.0, STAND_Z]] * n),
                root_link_quat_w=torch.tensor([[1.0, 0, 0, 0]] * n),
                joint_pos=torch.zeros(n, 14),
                default_joint_pos=torch.zeros(n, 14)))
        self.scene = _FakeScene(n, sensor, asset)
        self._sensor, self._asset = sensor, asset

    def tick(self, contacts=(1, 1)):
        """Advance one env step with the given per-foot contact flags."""
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self._sensor.data.found = torch.tensor(
            [list(contacts)] * self.num_envs, dtype=torch.float32)
        microduck_mdp._bow_update(
            self, "feet_ground_contact", microduck_mdp._DEFAULT_ASSET_CFG)

    def run_to(self, t_s):
        while float(self.episode_length_buf[0]) * self.step_dt < t_s - 1e-9:
            self.tick()


def test_phase_advances_with_time():
    env = _FakeEnv()
    seen = []
    for _ in range(int(STAND_END_S / env.step_dt)):
        env.tick()
        seen.append((round(float(env._bow_t[0]), 3), int(env._bow_phase[0]),
                     round(float(env._bow_blend[0]), 4),
                     round(float(env._bow_rise[0]), 4)))
    by_t = dict((t, p) for t, p, _, _ in seen)
    assert by_t[0.02] == microduck_mdp.BOW_PHASE_DIP
    assert by_t[1.0] == microduck_mdp.BOW_PHASE_HOLD    # boundary belongs to hold
    assert by_t[1.5] == microduck_mdp.BOW_PHASE_HOLD
    assert by_t[2.0] == microduck_mdp.BOW_PHASE_RISE
    assert by_t[2.5] == microduck_mdp.BOW_PHASE_RISE
    assert by_t[3.0] == microduck_mdp.BOW_PHASE_STAND
    assert by_t[4.0] == microduck_mdp.BOW_PHASE_STAND
    blend = dict((t, b) for t, _, b, _ in seen)
    rise = dict((t, r) for t, _, _, r in seen)
    # s ramps 0→1 over the dip, holds, ramps back to 0 over the rise.
    assert blend[0.5] == 0.5 and blend[1.0] == 1.0 and blend[1.5] == 1.0
    assert math.isclose(blend[2.5], 0.5, abs_tol=1e-6) and blend[3.0] == 0.0
    assert blend[4.0] == 0.0
    # r stays 0 until the rise, then ramps to 1 and holds.
    assert rise[1.5] == 0.0 and math.isclose(rise[2.5], 0.5, abs_tol=1e-6)
    assert rise[3.0] == 1.0 and rise[4.0] == 1.0


def test_targets_follow_the_slewed_ramp():
    env = _FakeEnv()
    h = lambda: float(microduck_mdp.bow_height_reward(env)[0])
    p = lambda: float(microduck_mdp.bow_pitch_reward(env)[0])
    env.tick()                       # t = 0.02 s: target is still ~STAND_Z
    assert h() > 0.99, "standing at t=0 must score: the ramp starts at STAND_Z"
    # Diving to the crouch immediately pays a fraction of staying on the
    # ramp — being ahead of the slew is not worth it (no jackpot).
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
    assert h() < 0.2
    env.run_to(HOLD_END_S)           # ... but at the hold it is the target
    assert h() > 0.99
    # A level trunk scores in the stand phase; nose-down scores at the hold.
    half = BOW_PITCH_RAD / 2
    level = p()
    assert level < 0.2, "a level trunk must score far under a bowed one"
    env._asset.data.root_link_quat_w = torch.tensor(
        [[math.cos(half), 0.0, math.sin(half), 0.0]])
    assert p() > 0.99, "nose-down is +y rotation; a flipped sign fails here"
    env._asset.data.root_link_quat_w = torch.tensor(
        [[math.cos(half), 0.0, -math.sin(half), 0.0]])
    assert p() < level, "nose-UP must score worse than level during the hold"


def test_stand_pose_pays_only_in_the_stand_phase():
    env = _FakeEnv()
    legs = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
    pose = lambda: float(
        microduck_mdp.bow_stand_pose_reward(env, joint_indices=legs)[0])
    env.run_to(HOLD_END_S)          # perfectly at HOME, but the wrong phase
    assert pose() == 0.0
    env.run_to(RISE_END_S - 0.02)
    assert pose() == 0.0, "the rise is not the finish"
    env.run_to(STAND_END_S)
    assert pose() == 1.0
    env._asset.data.joint_pos = torch.full((1, 14), 0.5)   # folded, not standing
    assert pose() < 0.1


def test_head_target_goes_down_then_up():
    env = _FakeEnv()
    down = torch.zeros(1, 14)
    down[0, 5], down[0, 6] = microduck_mdp.BOW_HEAD_DOWN
    up = torch.zeros(1, 14)
    up[0, 5], up[0, 6] = microduck_mdp.BOW_HEAD_UP
    head = lambda: float(microduck_mdp.bow_head_reward(env)[0])
    env.tick()                      # t≈0: the target is HOME, and it is at HOME
    assert head() > 0.99
    env.run_to(HOLD_END_S)          # holding at HOME while the bow wants it down
    at_home = head()
    env._asset.data.joint_pos = down
    assert head() > 0.99 > at_home, "head must be DOWN through the hold"
    env.run_to(STAND_END_S)
    assert head() < 0.5, "the head that stayed down does not finish the bow"
    env._asset.data.joint_pos = up
    assert head() > 0.99, "head must be LIFTED by the stand phase"


def test_foot_lift_cost_fires_on_contact_loss():
    env = _FakeEnv()
    lift = lambda: float(microduck_mdp.bow_foot_lift_penalty(env)[0])
    env.tick(contacts=(0, 0))        # spawn drop, not yet landed: free
    assert lift() == 0.0
    env.tick(contacts=(1, 1))
    assert lift() == 0.0
    env.tick(contacts=(0, 1))        # one foot up
    assert lift() == 0.5
    env.tick(contacts=(0, 0))        # both feet up
    assert lift() == 1.0
    env.tick(contacts=(1, 1))
    assert lift() == 0.0


def test_drift_cost_grows_and_saturates():
    env = _FakeEnv()
    env.tick()
    assert float(microduck_mdp.bow_drift_penalty(env)[0]) == 0.0
    env._asset.data.root_link_pos_w = torch.tensor([[0.05, 0.0, STAND_Z]])
    env.tick()
    assert 0.0 < float(microduck_mdp.bow_drift_penalty(env)[0]) < 1.0
    env._asset.data.root_link_pos_w = torch.tensor([[0.5, 0.0, STAND_Z]])
    env.tick()
    assert float(microduck_mdp.bow_drift_penalty(env)[0]) == 1.0


def test_memory_rearms_on_fresh_episode():
    env = _FakeEnv()
    env.tick(contacts=(1, 1))
    env.run_to(RISE_END_S)
    assert int(env._bow_phase[0]) == microduck_mdp.BOW_PHASE_STAND
    assert bool(env._bow_landed[0]) is True
    env._asset.data.root_link_pos_w = torch.tensor([[0.3, -0.2, STAND_Z]])
    env.episode_length_buf[:] = 0            # reset
    env.tick(contacts=(0, 0))
    assert int(env._bow_phase[0]) == microduck_mdp.BOW_PHASE_DIP
    assert float(env._bow_blend[0]) < 0.05 and float(env._bow_rise[0]) == 0.0
    assert bool(env._bow_landed[0]) is False, "touchdown latch must re-arm"
    assert torch.allclose(env._bow_home[0], torch.tensor([0.3, -0.2]))
    assert float(microduck_mdp.bow_drift_penalty(env)[0]) == 0.0


def test_update_runs_once_per_step():
    env = _FakeEnv()
    env.tick()
    t0 = float(env._bow_t[0])
    env.episode_length_buf += 5              # would change t if it recomputed
    microduck_mdp._bow_update(
        env, "feet_ground_contact", microduck_mdp._DEFAULT_ASSET_CFG)
    assert float(env._bow_t[0]) == t0, "memory must be keyed on the step counter"
