"""TippyTaps cfg invariants (CPU, no GPU): registration, the command is
pinned, gait terms are gone, every cost carries a negative weight, and the
per-env tap memory behaves on a fake sensor."""

import math
import types

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_tippy_taps_env_cfg import (
    EPISODE_LENGTH_S,
    MicroduckTippyTapsRlCfg,
    make_microduck_tippy_taps_env_cfg,
)


def test_registered():
    import mjlab_microduck.tasks  # noqa: F401  registers on import
    from mjlab.tasks.registry import list_tasks
    assert "Mjlab-TippyTaps-Flat-MicroDuck" in list_tasks()


def test_command_is_pinned_and_episode_short():
    cfg = make_microduck_tippy_taps_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.VelocityCommandCommandOnlyCfg)
    assert cmd.ranges.lin_vel_x == (-0.01, 0.01)
    assert cmd.ranges.ang_vel_z == (-0.05, 0.05)
    assert cmd.rel_standing_envs == 0.0 and cmd.heading_command is False
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 4.0
    assert "standing_envs" not in cfg.curriculum


def test_gait_terms_gone_and_tap_terms_signed():
    cfg = make_microduck_tippy_taps_env_cfg()
    for gone in ("air_time", "foot_clearance", "foot_swing_height", "foot_slip", "pose"):
        assert gone not in cfg.rewards, gone
    for pos in ("tap", "switch_feet", "pose_stand_legs", "height_stand",
                "upright", "track_linear_velocity", "head_pose_tracking"):
        assert cfg.rewards[pos].weight > 0, pos
    for cost in ("idle", "hop", "drift", "action_rate_l2", "dof_pos_limits", "self_collisions"):
        assert cfg.rewards[cost].weight < 0, cost
    assert cfg.rewards["tap"].weight > cfg.rewards["switch_feet"].weight > 0


def test_symmetry_on_and_obs_normalized():
    assert MicroduckTippyTapsRlCfg.algorithm.symmetry_cfg is not None
    assert MicroduckTippyTapsRlCfg.actor.obs_normalization is True
    assert MicroduckTippyTapsRlCfg.experiment_name == "tippy_taps"


# ── the tap memory on a fake env ─────────────────────────────────────────────

class _FakeSensor:
    def __init__(self, n):
        self.data = types.SimpleNamespace(
            found=torch.ones(n, 2), current_air_time=torch.zeros(n, 2))


class _FakeScene:
    def __init__(self, n, sensor, asset):
        self.sensors = {"feet_ground_contact": sensor}
        self.terrain = types.SimpleNamespace(env_origins=torch.zeros(n, 3))
        self._asset = asset

    def __getitem__(self, key):      # env.scene["robot"]
        return self._asset


class _FakeEnv:
    def __init__(self, n=3):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = 0.02
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        sensor = _FakeSensor(n)
        asset = types.SimpleNamespace(data=types.SimpleNamespace(
            root_link_pos_w=torch.zeros(n, 3),
            root_link_quat_w=torch.tensor([[1.0, 0, 0, 0]] * n)))
        self.scene = _FakeScene(n, sensor, asset)
        self._sensor, self._asset = sensor, asset

    def tick(self, contacts):
        """contacts: list of (left, right) bools per env; advances one step."""
        self.common_step_counter += 1
        self.episode_length_buf += 1
        c = torch.tensor(contacts, dtype=torch.float32)
        self._sensor.data.found = c
        air = self._sensor.data.current_air_time
        self._sensor.data.current_air_time = torch.where(c > 0, torch.zeros_like(air), air + self.step_dt)
        # The real env evaluates every reward term each step, so the memory
        # ticks once per step; mirror that here.
        microduck_mdp._tt_update(self, "feet_ground_contact", microduck_mdp._DEFAULT_ASSET_CFG)


def test_switch_pays_only_after_alternating_feet():
    env = _FakeEnv(1)
    env.tick([(1, 1)]); env.tick([(1, 1)])
    sw = lambda: float(microduck_mdp.tt_switch_reward(env)[0])
    tap = lambda: float(microduck_mdp.tt_tap_reward(env)[0])
    for _ in range(4):          # left up 0.08 s: a tap, nothing to alternate from
        env.tick([(0, 1)])
    assert tap() == 1.0 and sw() == 0.0
    env.tick([(1, 1)])
    for _ in range(4):          # same foot again: still no switch pay
        env.tick([(0, 1)])
    assert sw() == 0.0
    env.tick([(1, 1)])
    for _ in range(4):          # other foot: alternation pays
        env.tick([(1, 0)])
    assert sw() == 1.0 and tap() == 1.0
    for _ in range(int(microduck_mdp.TT_MAX_AIR_S / env.step_dt) + 2):
        env.tick([(1, 0)])      # ... until the lift becomes a hold
    assert sw() == 0.0 and tap() == 0.0


def test_idle_cost_ramps_after_grace_and_hop_waits_for_touchdown():
    env = _FakeEnv(1)
    idle = lambda: float(microduck_mdp.tt_idle_penalty(env)[0])
    hop = lambda: float(microduck_mdp.tt_hop_penalty(env)[0])
    env.tick([(0, 0)])           # spawn drop: both feet up, not yet landed
    assert hop() == 0.0
    for _ in range(int(0.25 / env.step_dt)):
        env.tick([(1, 1)])
    assert idle() == 0.0
    for _ in range(int(1.0 / env.step_dt)):
        env.tick([(1, 1)])
    assert idle() == 1.0
    env.tick([(0, 0)])
    assert hop() == 1.0, "airborne after touchdown is a hop"
    for _ in range(3):           # a real lift resets the idle clock
        env.tick([(0, 1)])
    assert idle() == 0.0


def test_memory_rearms_on_fresh_episode():
    env = _FakeEnv(1)
    env.tick([(1, 1)])
    for _ in range(3):
        env.tick([(0, 1)])
    microduck_mdp.tt_tap_reward(env)
    assert int(env._tt_prev_up[0]) >= 0
    env.episode_length_buf[:] = 0    # reset
    env.tick([(1, 1)])
    microduck_mdp.tt_tap_reward(env)
    assert int(env._tt_prev_up[0]) == -1 and bool(env._tt_landed[0]) is True
    assert math.isclose(float(env._tt_idle[0]), env.step_dt, rel_tol=1e-4)
