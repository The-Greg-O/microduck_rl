"""Pivot cfg invariants (CPU, no GPU): registration, the command is pinned and
carries the ±1 direction flag, the pivot-fighting terms are gone, every cost
carries a negative weight, and the per-env pivot memory behaves on a fake env
(pin foot follows the sign, the pin term fires on displacement and on lost
contact, the paddle pays alternation only, progress is a monotone potential,
everything re-arms on reset).

Same shape as tests/test_happy_spin_cfg.py and tests/test_bow_cfg.py — that is
the pattern this task follows.
"""

import math
import types

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_pivot_env_cfg import (
    EPISODE_LENGTH_S,
    PIVOT_DIRECTION,
    PV_DIRECTION_CCW,
    PV_DIRECTION_CW,
    PV_TARGET_YAW,
    MicroduckPivotRlCfg,
    make_microduck_pivot_env_cfg,
)


# ── cfg invariants ───────────────────────────────────────────────────────────

def test_registered():
    import mjlab_microduck.tasks  # noqa: F401  registers on import
    from mjlab.tasks.registry import list_tasks
    tasks = list_tasks()
    assert "Mjlab-Pivot-Flat-MicroDuck" in tasks
    # the other turn-in-place tasks are different animals and stay registered
    assert "Mjlab-HappySpin-Flat-MicroDuck" in tasks
    assert "Mjlab-Spin-Flat-MicroDuck" in tasks


def test_command_is_pinned_and_episode_short():
    cfg = make_microduck_pivot_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.PivotCommandCfg)
    assert cmd.ranges.lin_vel_x == (-0.01, 0.01)
    assert cmd.ranges.lin_vel_y == (-0.01, 0.01)
    assert cmd.rel_standing_envs == 0.0 and cmd.heading_command is False
    # never resampled inside an episode: the direction flag the policy reads on
    # step 0 is the one it reads on step 200
    assert cmd.resampling_time_range[0] >= EPISODE_LENGTH_S
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 4.0
    assert "standing_envs" not in cfg.curriculum


def test_direction_is_sampled_by_default_and_pinnable():
    assert PIVOT_DIRECTION is None
    assert make_microduck_pivot_env_cfg().commands["twist"].direction is None
    for d in (PV_DIRECTION_CCW, PV_DIRECTION_CW):
        assert make_microduck_pivot_env_cfg(direction=d).commands["twist"].direction == d
        assert make_microduck_pivot_env_cfg(direction=d).rewards["pin"].params[
            "direction"
        ] == d


def test_command_writes_the_direction_flag():
    """`PivotCommand._resample_command` writes ±1 into the yaw slot (obs[50])
    and leaves the linear slots tiny-but-alive."""
    ranges = types.SimpleNamespace(lin_vel_x=(-0.01, 0.01), lin_vel_y=(-0.01, 0.01))
    n = 512

    def stub(direction):
        return types.SimpleNamespace(
            cfg=types.SimpleNamespace(ranges=ranges, direction=direction),
            device="cpu",
            vel_command_b=torch.zeros(n, 3),
            vel_command_w=torch.zeros(n, 3),
            is_standing_env=torch.ones(n, dtype=torch.bool),
        )

    ids = torch.arange(n)
    free = stub(None)
    microduck_mdp.PivotCommand._resample_command(free, ids)
    yaw = free.vel_command_b[:, 2]
    assert set(yaw.unique().tolist()) == {-1.0, 1.0}, "the yaw slot is a ±1 flag"
    frac = float((yaw > 0).float().mean())
    assert 0.35 < frac < 0.65, "both directions are sampled, roughly evenly"
    assert free.vel_command_b[:, :2].abs().max() <= 0.01
    assert not free.is_standing_env.any(), "a pivoting env is never a standing env"
    torch.testing.assert_close(free.vel_command_w, free.vel_command_b)

    for d in (PV_DIRECTION_CCW, PV_DIRECTION_CW):
        pinned = stub(d)
        microduck_mdp.PivotCommand._resample_command(pinned, ids)
        assert (pinned.vel_command_b[:, 2] == d).all()


def test_pivot_fighting_terms_are_gone():
    cfg = make_microduck_pivot_env_cfg()
    # gait shaping (which, unlike in the other pinned-command skills, would NOT
    # have been inert: it gates on command magnitude and the yaw slot is ±1),
    # plus the three that fight a pivot head-on.
    for gone in ("air_time", "foot_clearance", "foot_swing_height", "foot_slip",
                 "pose", "angular_momentum", "track_angular_velocity",
                 "track_linear_velocity"):
        assert gone not in cfg.rewards, gone
    # body_ang_vel is x/y only — it mates roll/pitch thrash without touching
    # yaw, so it STAYS. And foot slip comes back for the PIN foot alone.
    assert "body_ang_vel" in cfg.rewards
    assert "pin_slip" in cfg.rewards


def test_reward_signs():
    cfg = make_microduck_pivot_env_cfg()
    for pos in ("pivot_progress", "pivot_complete", "pin", "paddle", "settle",
                "height_stand", "upright", "head_pose_tracking"):
        assert cfg.rewards[pos].weight > 0, pos
    for cost in ("pin_displacement", "pin_slip", "pivot_radius", "tilt",
                 "body_ang_vel", "action_rate_l2", "dof_pos_limits",
                 "self_collisions"):
        assert cfg.rewards[cost].weight < 0, cost
    # action rate stays LIGHT during discovery and ramps gently, late
    assert cfg.rewards["action_rate_l2"].weight == -0.05
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert [s["weight"] for s in stages] == [-0.05, -0.10, -0.20]
    assert stages[1]["step"] >= 1000 * 24


def test_a_spin_scores_clearly_less_than_a_pivot():
    """The weights, not the prose, are what make this a pivot: per step of the
    turn, keeping the pin planted is worth the pin pay AND the two pin costs."""
    cfg = make_microduck_pivot_env_cfg()
    pivot = cfg.rewards["pin"].weight + cfg.rewards["paddle"].weight
    spin = cfg.rewards["pin_displacement"].weight + cfg.rewards["pin_slip"].weight
    assert pivot - spin >= 5.0, "a two-footed spin must lose by a wide margin"
    # the turn itself still outweighs the settle it buys
    assert cfg.rewards["pivot_progress"].weight >= cfg.rewards["pin"].weight


def test_terminates_on_fall():
    cfg = make_microduck_pivot_env_cfg()
    assert cfg.terminations["fell_over"].time_out is False


def test_symmetry_off_and_runner_cfg():
    # the pin foot and the paddle foot have different roles: no mirror prior
    assert MicroduckPivotRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckPivotRlCfg.actor.obs_normalization is True
    assert MicroduckPivotRlCfg.critic.obs_normalization is True
    assert MicroduckPivotRlCfg.experiment_name == "pivot"
    assert MicroduckPivotRlCfg.max_iterations == 2_500


def test_direction_constants():
    assert PV_DIRECTION_CCW == 1.0 and PV_DIRECTION_CW == -1.0
    assert math.isclose(PV_TARGET_YAW, 2.0 * math.pi)


# ── the pivot memory on a fake env ───────────────────────────────────────────

_N_SERVO = 14
_LEFT, _RIGHT = 0, 1


class _FakeAsset:
    def __init__(self, n):
        # site_pos_w is (B, 2, 3): left_foot, right_foot — the order the
        # contact sensor uses too.
        feet = torch.zeros(n, 2, 3)
        feet[:, _LEFT, 1] = 0.042
        feet[:, _RIGHT, 1] = -0.042
        self.data = types.SimpleNamespace(
            root_link_pos_w=torch.zeros(n, 3),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n),
            root_link_ang_vel_b=torch.zeros(n, 3),
            site_pos_w=feet,
            joint_pos=torch.zeros(n, _N_SERVO),
            default_joint_pos=torch.zeros(n, _N_SERVO),
        )

    def find_joints(self, pattern):
        return list(range(_N_SERVO)), [f"j{i}" for i in range(_N_SERVO)]


class _FakeSensor:
    def __init__(self, n):
        self.data = types.SimpleNamespace(
            found=torch.ones(n, 2),
            current_air_time=torch.zeros(n, 2),
        )


class _FakeScene:
    def __init__(self, n, asset):
        self.sensors = {"feet_ground_contact": _FakeSensor(n)}
        self.terrain = types.SimpleNamespace(env_origins=torch.zeros(n, 3))
        self._asset = asset

    def __getitem__(self, key):      # env.scene["robot"]
        return self._asset


class _FakeCommandManager:
    def __init__(self, n, yaw):
        self._cmd = torch.zeros(n, 3)
        self._cmd[:, 2] = yaw

    def get_command(self, name):
        assert name == "twist"
        return self._cmd


class _FakeEnv:
    def __init__(self, n=1, yaw_flag=PV_DIRECTION_CCW):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = 0.02
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self._asset = _FakeAsset(n)
        self.scene = _FakeScene(n, self._asset)
        self.command_manager = _FakeCommandManager(n, yaw_flag)

    @property
    def sensor(self):
        return self.scene.sensors["feet_ground_contact"]

    def set_contact(self, foot, down):
        self.sensor.data.found[:, foot] = 1.0 if down else 0.0
        if down:
            self.sensor.data.current_air_time[:, foot] = 0.0

    def tick(self, omega_z=0.0, pin_dxy=(0.0, 0.0), trunk_dxy=(0.0, 0.0)):
        """Advance one env step: trunk yaw rate, pin-foot drift, trunk drift."""
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self._asset.data.root_link_ang_vel_b[:, 2] = omega_z
        pin = int(self._pin_index())
        self._asset.data.site_pos_w[:, pin, 0] += pin_dxy[0]
        self._asset.data.site_pos_w[:, pin, 1] += pin_dxy[1]
        self._asset.data.root_link_pos_w[:, 0] += trunk_dxy[0]
        self._asset.data.root_link_pos_w[:, 1] += trunk_dxy[1]
        # air time accrues for whichever foot is off the floor
        air = self.sensor.data.current_air_time
        up = self.sensor.data.found <= 0
        air += up.float() * self.step_dt
        air *= up.float()
        # The real env evaluates every reward term each step, so the memory
        # ticks once per step; mirror that here.
        microduck_mdp._pv_update(self)

    def _pin_index(self):
        return getattr(self, "_pv_pin", torch.zeros(1, dtype=torch.long))[0]


def _turn_for(env, omega, seconds, **kw):
    for _ in range(int(round(seconds / env.step_dt))):
        env.tick(omega, **kw)


def test_pin_foot_follows_the_direction_flag():
    """The pin is the INSIDE foot of the turn: left for counter-clockwise."""
    ccw = _FakeEnv(yaw_flag=PV_DIRECTION_CCW)
    ccw.tick()
    assert float(ccw._pv_dir[0]) == 1.0
    assert int(ccw._pv_pin[0]) == _LEFT

    cw = _FakeEnv(yaw_flag=PV_DIRECTION_CW)
    cw.tick()
    assert float(cw._pv_dir[0]) == -1.0
    assert int(cw._pv_pin[0]) == _RIGHT

    # an all-zero twist (an unconfigured skill entry) reads as counter-clockwise
    zero = _FakeEnv(yaw_flag=0.0)
    zero.tick()
    assert float(zero._pv_dir[0]) == 1.0


def test_direction_is_latched_for_the_episode():
    """Observation noise on the flag must not swap the pin foot mid-turn."""
    env = _FakeEnv(yaw_flag=PV_DIRECTION_CCW)
    _turn_for(env, 2.0, 0.5)
    env.command_manager._cmd[:, 2] = PV_DIRECTION_CW      # the flag flickers
    env.tick(2.0)
    assert float(env._pv_dir[0]) == 1.0, "latched at episode start"
    assert int(env._pv_pin[0]) == _LEFT


def test_progress_is_a_monotone_potential():
    env = _FakeEnv()
    omega = math.pi                       # half a turn per second
    _turn_for(env, omega, 1.0)
    assert math.isclose(float(env._pv_yaw[0]), math.pi, rel_tol=1e-6)
    assert bool(env._pv_done[0]) is False
    assert float(microduck_mdp.pv_progress_reward(env)[0]) > 0.0
    # rocking back pays nothing and cannot re-earn the same degrees
    peak = float(env._pv_yaw[0])
    _turn_for(env, -omega, 0.5)
    assert float(env._pv_yaw[0]) == peak
    replay = 0.0
    for _ in range(int(0.5 / env.step_dt)):
        env.tick(omega)
        replay += float(microduck_mdp.pv_progress_reward(env)[0])
    assert replay == 0.0, "re-crossing a banked arc pays nothing"
    # and the full turn flips into the settle phase, once
    _turn_for(env, omega, 1.0)
    assert bool(env._pv_done[0]) is True


def test_progress_rate_is_normalized_and_capped():
    env = _FakeEnv()
    env.tick(math.pi)                     # half the cap rate
    assert math.isclose(float(microduck_mdp.pv_progress_reward(env)[0]), 0.5, rel_tol=1e-6)
    env.tick(2 * math.pi)                 # exactly the cap
    assert math.isclose(float(microduck_mdp.pv_progress_reward(env)[0]), 1.0, rel_tol=1e-6)
    env.tick(20.0)                        # way past it: still 1.0, no extra pay
    assert math.isclose(float(microduck_mdp.pv_progress_reward(env)[0]), 1.0, rel_tol=1e-6)


def test_progress_total_is_potential_based():
    """A completed turn is worth the same total however fast it is done — the
    speed incentive lives in the settle phase, not here."""
    totals = []
    for omega in (math.pi, 2 * math.pi):
        env = _FakeEnv()
        total = 0.0
        for _ in range(200):              # a full 4 s episode
            env.tick(omega)
            total += float(microduck_mdp.pv_progress_reward(env)[0])
        totals.append(total)
    assert math.isclose(totals[0], totals[1], rel_tol=1e-6)


def test_complete_bonus_is_one_shot():
    env = _FakeEnv()
    fired = 0
    for _ in range(200):
        env.tick(2 * math.pi)
        fired += int(round(float(microduck_mdp.pv_complete_bonus(env)[0])))
    assert fired == 1, "a per-step 'you are done' reward would be a jackpot"


def test_pin_term_fires_on_displacement():
    env = _FakeEnv()
    env.tick(2.0)
    assert float(microduck_mdp.pv_pin_reward(env)[0]) > 0.99
    assert float(microduck_mdp.pv_pin_displacement_penalty(env)[0]) == 0.0
    # slide the pin 1 mm/step for 20 steps = 2 cm: well past PV_PIN_STD_M
    _turn_for(env, 2.0, 0.4, pin_dxy=(0.001, 0.0))
    assert math.isclose(float(env._pv_pin_d[0]), 0.02, rel_tol=1e-5)
    assert float(microduck_mdp.pv_pin_reward(env)[0]) < 0.2
    cost = float(microduck_mdp.pv_pin_displacement_penalty(env)[0])
    assert 0.0 < cost <= 1.0
    # a full foot length away and the cost is saturated
    _turn_for(env, 2.0, 0.4, pin_dxy=(0.002, 0.0))
    assert float(microduck_mdp.pv_pin_reward(env)[0]) < 1e-6
    assert float(microduck_mdp.pv_pin_displacement_penalty(env)[0]) == 1.0


def test_pin_term_fires_on_lost_contact():
    env = _FakeEnv()
    env.tick(2.0)
    assert float(microduck_mdp.pv_pin_reward(env)[0]) > 0.99
    env.set_contact(_LEFT, False)         # the pin leaves the floor
    env.tick(2.0)
    assert float(microduck_mdp.pv_pin_reward(env)[0]) == 0.0
    env.set_contact(_LEFT, True)
    env.tick(2.0)
    assert float(microduck_mdp.pv_pin_reward(env)[0]) > 0.99


def test_pin_slip_is_charged_only_while_planted():
    env = _FakeEnv()
    env.tick(2.0)
    assert float(microduck_mdp.pv_pin_slip_penalty(env)[0]) == 0.0
    env.tick(2.0, pin_dxy=(0.002, 0.0))   # 0.1 m/s of in-contact sliding
    assert float(microduck_mdp.pv_pin_slip_penalty(env)[0]) == 1.0
    env.set_contact(_LEFT, False)         # the same motion in the air is a swing
    env.tick(2.0, pin_dxy=(0.002, 0.0))
    assert float(microduck_mdp.pv_pin_slip_penalty(env)[0]) == 0.0


def test_paddle_pays_alternation_only():
    env = _FakeEnv()
    env.tick(2.0)                                     # both feet down: no stroke
    assert float(microduck_mdp.pv_paddle_reward(env)[0]) == 0.0

    env.set_contact(_RIGHT, False)                    # the FREE foot lifts
    paid = 0.0
    for _ in range(int(0.3 / env.step_dt)):           # inside the stroke window
        env.tick(2.0)
        paid += float(microduck_mdp.pv_paddle_reward(env)[0])
    assert paid > 0.0, "a stroke in the air-time window pays"

    # hold it up past the window and the pay stops: that is a flamingo, not a
    # paddle, and it cannot be farmed by simply never landing.
    for _ in range(int(1.0 / env.step_dt)):
        env.tick(2.0)
    assert float(microduck_mdp.pv_paddle_reward(env)[0]) == 0.0

    # plant it, lift again: a fresh window, bought with a fresh contact
    env.set_contact(_RIGHT, True)
    env.tick(2.0)
    env.set_contact(_RIGHT, False)
    again = 0.0
    for _ in range(int(0.3 / env.step_dt)):
        env.tick(2.0)
        again += float(microduck_mdp.pv_paddle_reward(env)[0])
    assert again > 0.0, "alternation re-arms the pay"


def test_paddle_pays_neither_the_pin_nor_a_hop():
    env = _FakeEnv()
    env.tick(2.0)
    env.set_contact(_LEFT, False)         # the PIN lifts instead: no pay
    for _ in range(int(0.2 / env.step_dt)):
        env.tick(2.0)
    assert float(microduck_mdp.pv_paddle_reward(env)[0]) == 0.0
    env.set_contact(_RIGHT, False)        # both feet up: a hop, still no pay
    for _ in range(int(0.1 / env.step_dt)):
        env.tick(2.0)
    assert float(microduck_mdp.pv_paddle_reward(env)[0]) == 0.0


def test_paddle_is_gated_on_the_spin_phase():
    env = _FakeEnv()
    _turn_for(env, 2 * math.pi, 1.1)      # the turn is done
    assert bool(env._pv_done[0]) is True
    env.set_contact(_RIGHT, False)
    for _ in range(int(0.2 / env.step_dt)):
        env.tick(0.0)
    assert float(microduck_mdp.pv_paddle_reward(env)[0]) == 0.0, (
        "paddling on after the turn must not compete with settling"
    )


def test_settle_pays_only_when_done_still_posed_and_on_both_feet():
    env = _FakeEnv()
    _turn_for(env, 2 * math.pi, 1.0)      # complete the turn
    assert bool(env._pv_done[0]) is True
    env.tick(2 * math.pi)                 # still whirling: near zero
    assert float(microduck_mdp.pv_settle_reward(env)[0]) < 0.05
    env.tick(0.0)                         # stopped, legs at HOME, feet down
    assert float(microduck_mdp.pv_settle_reward(env)[0]) > 0.95
    env.set_contact(_RIGHT, False)        # one foot up: not a settle
    env.tick(0.0)
    assert float(microduck_mdp.pv_settle_reward(env)[0]) == 0.0
    env.set_contact(_RIGHT, True)
    env._asset.data.joint_pos[:] = 1.0    # legs nowhere near HOME
    env.tick(0.0)
    assert float(microduck_mdp.pv_settle_reward(env)[0]) < 0.05


def test_settle_and_progress_are_gated_on_upright():
    env = _FakeEnv()
    env.tick(2 * math.pi)
    assert float(microduck_mdp.pv_progress_reward(env)[0]) > 0.0
    assert float(microduck_mdp.pv_pin_reward(env)[0]) > 0.0
    # tip past the 40° gate: quat for a 90° roll
    s = math.sin(math.pi / 4)
    env._asset.data.root_link_quat_w[:] = torch.tensor([math.cos(math.pi / 4), s, 0.0, 0.0])
    env.tick(2 * math.pi)
    assert float(microduck_mdp.pv_progress_reward(env)[0]) == 0.0
    assert float(microduck_mdp.pv_pin_reward(env)[0]) == 0.0


def test_radius_cost_allows_the_orbit_but_not_a_walk_away():
    """The trunk MUST travel — it orbits the pin — so only leaving the circle
    is priced."""
    env = _FakeEnv()
    env.tick(0.0)
    assert float(microduck_mdp.pv_radius_penalty(env)[0]) == 0.0
    # sit on the orbit radius itself (4.2 cm from the pin): free
    env._asset.data.root_link_pos_w[:, 0] = 0.0
    env._asset.data.root_link_pos_w[:, 1] = microduck_mdp.PV_RADIUS_M + 0.042
    env.tick(0.0)
    assert float(microduck_mdp.pv_radius_penalty(env)[0]) == 0.0
    # walk half a metre away: saturated, bounded
    for _ in range(50):
        env.tick(0.0, trunk_dxy=(0.01, 0.0))
    assert float(microduck_mdp.pv_radius_penalty(env)[0]) == 1.0


def test_tilt_cost_is_bounded_and_nonnegative():
    env = _FakeEnv()
    env.tick(0.0)
    assert float(microduck_mdp.pv_tilt_penalty(env)[0]) == 0.0
    s = math.sin(math.pi / 4)
    env._asset.data.root_link_quat_w[:] = torch.tensor([math.cos(math.pi / 4), s, 0.0, 0.0])
    env.tick(0.0)
    assert 0.0 < float(microduck_mdp.pv_tilt_penalty(env)[0]) <= 1.0


def test_memory_rearms_on_fresh_episode():
    env = _FakeEnv()
    _turn_for(env, 2 * math.pi, 1.2, pin_dxy=(0.001, 0.0))
    assert bool(env._pv_done[0]) is True and float(env._pv_yaw[0]) > 0.0
    assert float(env._pv_pin_d[0]) > 0.0
    home_before = env._pv_pin_home[0].clone()
    env.episode_length_buf[:] = 0          # reset
    env.tick(0.0)
    assert float(env._pv_yaw[0]) == 0.0
    assert float(env._pv_raw[0]) == 0.0
    assert bool(env._pv_done[0]) is False
    assert bool(env._pv_just_done[0]) is False
    assert float(env._pv_pin_d[0]) == 0.0, "the pin re-anchors where it now is"
    assert not torch.allclose(env._pv_pin_home[0], home_before)
    assert float(microduck_mdp.pv_pin_displacement_penalty(env)[0]) == 0.0
    assert float(microduck_mdp.pv_pin_slip_penalty(env)[0]) == 0.0


def test_direction_is_re_read_on_a_fresh_episode():
    """A per-env sign resampled at reset must reach the memory: the flag is
    latched per EPISODE, not per policy."""
    env = _FakeEnv(yaw_flag=PV_DIRECTION_CCW)
    _turn_for(env, 2.0, 0.2)
    assert int(env._pv_pin[0]) == _LEFT
    env.command_manager._cmd[:, 2] = PV_DIRECTION_CW
    env.episode_length_buf[:] = 0
    env.tick(0.0)
    assert float(env._pv_dir[0]) == -1.0
    assert int(env._pv_pin[0]) == _RIGHT
