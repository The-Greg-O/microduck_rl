"""HappySpin cfg invariants (CPU, no GPU): registration, the command is pinned,
the spin-fighting terms are gone, every cost carries a negative weight, and the
per-env yaw memory behaves on a fake env (accumulates, flips phase at 360°,
caps, re-arms on reset).

Same shape as tests/test_tippy_taps_cfg.py — that is the pattern this task
follows. NOTE: `Mjlab-Spin-Flat-MicroDuck` is the OTHER, pre-existing spin (the
cyclic roller gesture); this suite must leave it alone, and asserts so.
"""

import math
import types

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_happy_spin_env_cfg import (
    EPISODE_LENGTH_S,
    HS_COMPLETE_TOL,
    HS_DIRECTION,
    HS_HEADING_STD,
    HS_OVERSHOOT_SAT,
    HS_TARGET_YAW,
    W_HEADING,
    W_OVERSHOOT,
    MicroduckHappySpinRlCfg,
    make_microduck_happy_spin_env_cfg,
)


def test_registered_alongside_the_roller_spin():
    import mjlab_microduck.tasks  # noqa: F401  registers on import
    from mjlab.tasks.registry import list_tasks
    tasks = list_tasks()
    assert "Mjlab-HappySpin-Flat-MicroDuck" in tasks
    # the pre-existing roller spin is a different task and stays registered
    assert "Mjlab-Spin-Flat-MicroDuck" in tasks


def test_command_is_pinned_and_episode_short():
    cfg = make_microduck_happy_spin_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.VelocityCommandCommandOnlyCfg)
    assert cmd.ranges.lin_vel_x == (-0.01, 0.01)
    assert cmd.ranges.lin_vel_y == (-0.01, 0.01)
    assert cmd.ranges.ang_vel_z == (-0.05, 0.05)
    assert cmd.rel_standing_envs == 0.0 and cmd.heading_command is False
    # never resampled inside an episode
    assert cmd.resampling_time_range[0] >= EPISODE_LENGTH_S
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 4.0
    assert "standing_envs" not in cfg.curriculum


def test_spin_fighting_terms_are_gone():
    cfg = make_microduck_happy_spin_env_cfg()
    # gait terms, plus the two that fight a spin head-on: angular_momentum
    # (3D norm) and track_angular_velocity (command pinned to ~0 yaw rate).
    for gone in ("air_time", "foot_clearance", "foot_swing_height", "foot_slip",
                 "pose", "angular_momentum", "track_angular_velocity"):
        assert gone not in cfg.rewards, gone
    # body_ang_vel is x/y only — it mates roll/pitch thrash without touching
    # yaw, so it STAYS.
    assert "body_ang_vel" in cfg.rewards


def test_reward_signs():
    cfg = make_microduck_happy_spin_env_cfg()
    for pos in ("spin_progress", "spin_complete", "settle", "heading",
                "height_stand", "upright", "track_linear_velocity",
                "head_pose_tracking"):
        assert cfg.rewards[pos].weight > 0, pos
    for cost in ("drift", "tilt", "overshoot", "body_ang_vel", "action_rate_l2",
                 "dof_pos_limits", "self_collisions"):
        assert cfg.rewards[cost].weight < 0, cost
    # the turn itself outweighs the settle it buys
    assert cfg.rewards["spin_progress"].weight > cfg.rewards["settle"].weight
    # action rate stays LIGHT during discovery and ramps gently
    assert cfg.rewards["action_rate_l2"].weight == -0.05
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert [s["weight"] for s in stages] == [-0.05, -0.10, -0.20]


def test_terminates_on_fall():
    cfg = make_microduck_happy_spin_env_cfg()
    assert cfg.terminations["fell_over"].time_out is False


def test_symmetry_off_and_runner_cfg():
    # a spin has a direction: the left/right mirror is the OTHER spin
    assert MicroduckHappySpinRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckHappySpinRlCfg.actor.obs_normalization is True
    assert MicroduckHappySpinRlCfg.critic.obs_normalization is True
    assert MicroduckHappySpinRlCfg.experiment_name == "happy_spin"
    assert MicroduckHappySpinRlCfg.max_iterations == 2_000


def test_direction_is_counter_clockwise():
    assert HS_DIRECTION == 1.0
    assert math.isclose(HS_TARGET_YAW, 2.0 * math.pi)


# ── the yaw memory on a fake env ─────────────────────────────────────────────

_N_SERVO = 14


class _FakeAsset:
    def __init__(self, n):
        self.data = types.SimpleNamespace(
            root_link_pos_w=torch.zeros(n, 3),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n),
            root_link_ang_vel_b=torch.zeros(n, 3),
            joint_pos=torch.zeros(n, _N_SERVO),
            default_joint_pos=torch.zeros(n, _N_SERVO),
        )

    def find_joints(self, pattern):
        return list(range(_N_SERVO)), [f"j{i}" for i in range(_N_SERVO)]


class _FakeScene:
    def __init__(self, n, asset):
        self.sensors = {}
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
        self._asset = _FakeAsset(n)
        self.scene = _FakeScene(n, self._asset)

    def tick(self, omega_z, dx=0.0):
        """Advance one env step with a trunk yaw rate (rad/s) and x drift."""
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self._asset.data.root_link_ang_vel_b[:, 2] = omega_z
        self._asset.data.root_link_pos_w[:, 0] += dx
        # The real env evaluates every reward term each step, so the memory
        # ticks once per step; mirror that here.
        microduck_mdp._hs_update(self)


def _spin_for(env, omega, seconds):
    for _ in range(int(round(seconds / env.step_dt))):
        env.tick(omega)


def test_yaw_accumulates_and_flips_phase_at_360():
    env = _FakeEnv()
    omega = math.pi          # half a turn per second → 2 s for the full turn
    _spin_for(env, omega, 1.0)
    assert math.isclose(float(env._hs_yaw[0]), math.pi, rel_tol=1e-6)
    assert bool(env._hs_done[0]) is False
    assert float(microduck_mdp.hs_settle_reward(env)[0]) == 0.0
    assert float(microduck_mdp.hs_progress_reward(env)[0]) > 0.0
    _spin_for(env, omega, 1.0)
    assert math.isclose(float(env._hs_yaw[0]), 2 * math.pi, rel_tol=1e-6)
    assert bool(env._hs_done[0]) is True, "360 degrees flips into the settle phase"


def test_progress_rate_is_normalized_and_capped():
    env = _FakeEnv()
    env.tick(math.pi)                      # half the cap rate
    assert math.isclose(float(microduck_mdp.hs_progress_reward(env)[0]), 0.5, rel_tol=1e-6)
    env.tick(2 * math.pi)                  # exactly the cap
    assert math.isclose(float(microduck_mdp.hs_progress_reward(env)[0]), 1.0, rel_tol=1e-6)
    env.tick(20.0)                         # way past it: still 1.0, no extra pay
    assert math.isclose(float(microduck_mdp.hs_progress_reward(env)[0]), 1.0, rel_tol=1e-6)


def test_backspin_banks_nothing_and_must_be_undone():
    """The potential measures NET counter-clockwise yaw: turning the wrong way
    pays nothing, and those degrees have to be given back before progress
    resumes (a 360 that isn't net 360 isn't the trick)."""
    env = _FakeEnv()
    _spin_for(env, -4.0, 0.5)              # clockwise: the wrong way, -2 rad
    assert float(env._hs_yaw[0]) == 0.0
    assert float(microduck_mdp.hs_progress_reward(env)[0]) == 0.0
    assert math.isclose(float(env._hs_raw[0]), -2.0, rel_tol=1e-6)
    _spin_for(env, 4.0, 0.4)               # +1.6 rad: still under water
    assert float(env._hs_yaw[0]) == 0.0
    assert float(microduck_mdp.hs_progress_reward(env)[0]) == 0.0
    _spin_for(env, 4.0, 0.2)               # -2.0 +1.6 +0.8 = net +0.4 rad
    assert math.isclose(float(env._hs_yaw[0]), 0.4, rel_tol=1e-5)
    assert float(microduck_mdp.hs_progress_reward(env)[0]) > 0.0


def test_wobble_cannot_re_earn_the_same_degrees():
    """The potential is a running maximum, so rocking back and forth over the
    same arc pays once, not once per pass."""
    env = _FakeEnv()
    _spin_for(env, 4.0, 0.25)              # +1.0 rad
    peak = float(env._hs_yaw[0])
    _spin_for(env, -4.0, 0.25)             # back to 0
    assert float(env._hs_yaw[0]) == peak
    replay = 0.0
    for _ in range(int(0.25 / env.step_dt)):   # forward over the same arc again
        env.tick(4.0)
        replay += float(microduck_mdp.hs_progress_reward(env)[0])
    assert replay == 0.0, "re-crossing an arc already banked pays nothing"
    assert math.isclose(float(env._hs_yaw[0]), peak, rel_tol=1e-6)


def test_over_spinning_pays_nothing_after_completion():
    env = _FakeEnv()
    _spin_for(env, 2 * math.pi, 1.2)       # more than a full turn
    assert bool(env._hs_done[0]) is True
    assert float(env._hs_yaw[0]) <= 2 * math.pi + 1e-6
    env.tick(2 * math.pi)
    assert float(microduck_mdp.hs_progress_reward(env)[0]) == 0.0


def test_progress_total_is_potential_based():
    """The completed turn is worth the same total however fast it is spun —
    the speed incentive lives in the settle phase, not here."""
    totals = []
    for omega in (math.pi, 2 * math.pi):
        env = _FakeEnv()
        total = 0.0
        for _ in range(200):               # a full 4 s episode
            env.tick(omega)
            total += float(microduck_mdp.hs_progress_reward(env)[0])
        totals.append(total)
    assert math.isclose(totals[0], totals[1], rel_tol=1e-6)
    assert math.isclose(totals[0], 1.0 / 0.02, rel_tol=1e-6)


def test_complete_bonus_is_one_shot():
    env = _FakeEnv()
    fired = 0
    for _ in range(200):
        env.tick(2 * math.pi)
        fired += int(round(float(microduck_mdp.hs_complete_bonus(env)[0])))
    assert fired == 1, "a per-step 'you are done' reward would be a jackpot"


def test_settle_pays_only_when_done_still_and_posed():
    env = _FakeEnv()
    _spin_for(env, 2 * math.pi, 1.0)       # complete the turn
    assert bool(env._hs_done[0]) is True
    env.tick(2 * math.pi)                  # still whirling: near zero
    assert float(microduck_mdp.hs_settle_reward(env)[0]) < 0.05
    env.tick(0.0)                          # stopped, legs at HOME: near one
    assert float(microduck_mdp.hs_settle_reward(env)[0]) > 0.95
    env._asset.data.joint_pos[:] = 1.0     # legs nowhere near HOME
    env.tick(0.0)
    assert float(microduck_mdp.hs_settle_reward(env)[0]) < 0.05


def test_settle_and_progress_are_gated_on_upright():
    env = _FakeEnv()
    env.tick(2 * math.pi)
    assert float(microduck_mdp.hs_progress_reward(env)[0]) > 0.0
    # tip past the 40° gate: quat for a 90° roll
    s = math.sin(math.pi / 4)
    env._asset.data.root_link_quat_w[:] = torch.tensor([math.cos(math.pi / 4), s, 0.0, 0.0])
    env.tick(2 * math.pi)
    assert float(microduck_mdp.hs_progress_reward(env)[0]) == 0.0


def test_drift_and_tilt_costs_are_bounded_and_nonnegative():
    env = _FakeEnv()
    env.tick(0.0)
    assert float(microduck_mdp.hs_drift_penalty(env)[0]) == 0.0
    assert float(microduck_mdp.hs_tilt_penalty(env)[0]) == 0.0
    for _ in range(50):                    # 50 × 1 cm = 50 cm away
        env.tick(0.0, dx=0.01)
    assert float(microduck_mdp.hs_drift_penalty(env)[0]) == 1.0   # saturated, bounded
    s = math.sin(math.pi / 4)
    env._asset.data.root_link_quat_w[:] = torch.tensor([math.cos(math.pi / 4), s, 0.0, 0.0])
    env.tick(0.0)
    tilt = float(microduck_mdp.hs_tilt_penalty(env)[0])
    assert 0.0 < tilt <= 1.0


def test_memory_rearms_on_fresh_episode():
    env = _FakeEnv()
    for _ in range(50):
        env.tick(2 * math.pi, dx=0.002)
    assert bool(env._hs_done[0]) is True and float(env._hs_yaw[0]) > 0.0
    home_before = env._hs_home[0].clone()
    env.episode_length_buf[:] = 0          # reset
    env.tick(0.0)
    assert float(env._hs_yaw[0]) == 0.0
    assert bool(env._hs_done[0]) is False
    assert bool(env._hs_just_done[0]) is False
    assert not torch.allclose(env._hs_home[0], home_before), "home re-anchors"
    assert float(microduck_mdp.hs_drift_penalty(env)[0]) == 0.0


# ── where it STOPS: the heading term and the over-rotation cost ───────────────
# happy-spin-v1 (ckpt 1999, BAM, noise + DR on, 3 seeds) spun a clean 360° at
# 0.74–0.88 s with no falls and < 8 cm drift, then coasted on to 415 / 457 /
# 415° total — settling 55–100° past its starting heading. The stack was blind
# to that: the progress potential clamps at 2π (over-rotation pays nothing, but
# costs nothing), the completion bonus fired on CROSSING, and settle pays for
# stillness facing ANY direction. These tests lock in the two terms that fix it.


def _turn(env, radians, steps):
    """Turn exactly `radians` over `steps` env steps (constant yaw rate)."""
    omega = radians / (steps * env.step_dt)
    for _ in range(steps):
        env.tick(omega)


def _complete_the_turn(env, steps=50):
    _turn(env, HS_TARGET_YAW, steps)
    assert bool(env._hs_done[0]) is True
    return env


def test_heading_peaks_at_exactly_one_turn():
    env = _complete_the_turn(_FakeEnv())
    env.tick(0.0)                          # stopped, dead on the mark
    assert float(microduck_mdp.hs_heading_reward(env)[0]) > 0.999


def test_heading_is_near_zero_sixty_degrees_past():
    env = _complete_the_turn(_FakeEnv())
    _turn(env, math.radians(60.0), 10)     # 420° total: v1's failure mode
    env.tick(0.0)
    assert float(microduck_mdp.hs_heading_reward(env)[0]) < 1e-6


def test_heading_falls_off_over_ten_degrees():
    """std ≈ 10°: the argmax is stopping ON the mark, not near it."""
    env = _complete_the_turn(_FakeEnv())
    _turn(env, math.radians(10.0), 10)
    env.tick(0.0)
    at_ten = float(microduck_mdp.hs_heading_reward(env)[0])
    assert math.isclose(at_ten, math.exp(-1.0), rel_tol=1e-3)
    assert math.isclose(HS_HEADING_STD, math.radians(10.0))


def test_heading_pays_nothing_before_the_turn_completes():
    """Settle-phase only — which is what makes the WRAPPED error unambiguous:
    the same wrapped heading at zero turns must not pay."""
    env = _FakeEnv()
    _turn(env, math.radians(2.0), 5)       # barely moved: wrapped err ≈ -2°
    assert bool(env._hs_done[0]) is False
    assert float(microduck_mdp.hs_heading_reward(env)[0]) == 0.0


def test_heading_is_gated_on_upright():
    env = _complete_the_turn(_FakeEnv())
    s = math.sin(math.pi / 4)              # 90° roll, past the 40° gate
    env._asset.data.root_link_quat_w[:] = torch.tensor(
        [math.cos(math.pi / 4), s, 0.0, 0.0]
    )
    env.tick(0.0)
    assert float(microduck_mdp.hs_heading_reward(env)[0]) == 0.0


def test_overshoot_is_zero_below_one_turn_and_grows_past_it():
    env = _FakeEnv()
    _turn(env, math.pi, 25)                # half a turn
    assert float(microduck_mdp.hs_overshoot_penalty(env)[0]) == 0.0
    _turn(env, math.pi, 25)                # exactly one turn: still free
    assert float(microduck_mdp.hs_overshoot_penalty(env)[0]) < 1e-4
    _turn(env, math.radians(45.0), 10)     # half of the 90° saturation
    assert math.isclose(
        float(microduck_mdp.hs_overshoot_penalty(env)[0]), 0.5, rel_tol=1e-4
    )
    _turn(env, math.radians(45.0), 10)     # 90° over: saturated
    assert math.isclose(
        float(microduck_mdp.hs_overshoot_penalty(env)[0]), 1.0, rel_tol=1e-4
    )


def test_overshoot_is_bounded():
    """Bounded per the playbook: it can never outweigh completing the turn, so
    'don't spin at all' never becomes the argmax."""
    env = _FakeEnv()
    _turn(env, 3.0 * HS_TARGET_YAW, 60)    # three turns
    assert float(microduck_mdp.hs_overshoot_penalty(env)[0]) == 1.0
    assert math.isclose(HS_OVERSHOOT_SAT, math.radians(90.0))


def test_overshoot_stops_charging_once_the_heading_is_recovered():
    """Instantaneous, not a running maximum: turning back onto the mark is the
    correction we want, and it lands where the heading Gaussian peaks."""
    env = _complete_the_turn(_FakeEnv())
    _turn(env, math.radians(45.0), 10)
    assert float(microduck_mdp.hs_overshoot_penalty(env)[0]) > 0.0
    _turn(env, -math.radians(45.0), 10)    # back onto the starting heading
    env.tick(0.0)
    assert float(microduck_mdp.hs_overshoot_penalty(env)[0]) < 1e-4
    assert float(microduck_mdp.hs_heading_reward(env)[0]) > 0.999


def test_complete_bonus_fires_just_short_of_the_turn_and_progress_continues():
    """The bonus fires WITHIN a few degrees rather than on crossing, so the
    settle stack is live while the last degrees are turned — and
    `spin_progress` must keep paying through them, or the early latch would
    just buy a turn that ends 5° short."""
    env = _FakeEnv()
    omega = 0.5                            # slow: 0.01 rad per step
    env.tick(omega)
    while not bool(env._hs_done[0]):
        env.tick(omega)
    at_latch = float(env._hs_raw[0])
    assert at_latch < HS_TARGET_YAW, "the latch flips BEFORE the full turn"
    assert HS_TARGET_YAW - at_latch <= HS_COMPLETE_TOL + omega * env.step_dt
    assert math.isclose(HS_COMPLETE_TOL, math.radians(5.0))
    env.tick(omega)                        # still turning the last degrees
    assert float(microduck_mdp.hs_progress_reward(env)[0]) > 0.0


def test_complete_bonus_still_fires_exactly_once_with_the_early_trigger():
    env = _FakeEnv()
    fired = 0
    for _ in range(200):
        env.tick(2 * math.pi)
        fired += int(round(float(microduck_mdp.hs_complete_bonus(env)[0])))
    assert fired == 1


def test_stopping_on_the_mark_beats_crossing_fast_and_drifting():
    """The whole point, in weighted per-step arithmetic over a 4 s episode:
    v1's 'cross at 0.8 s, coast to 415°' must now lose to 'cross at 0.8 s and
    stop'. Only the two NEW terms are summed here — they are the entire delta
    between v1's stack and this one."""

    def episode(overshoot_deg, drift_steps):
        env = _FakeEnv()
        total = 0.0
        _turn(env, HS_TARGET_YAW, 40)      # the measured 0.8 s turn (unchanged)
        omega = (
            math.radians(overshoot_deg) / (drift_steps * env.step_dt)
            if drift_steps
            else 0.0
        )
        for step in range(160):            # the settle phase, 160 steps
            env.tick(omega if step < drift_steps else 0.0)
            total += W_HEADING * float(microduck_mdp.hs_heading_reward(env)[0])
            total += W_OVERSHOOT * float(
                microduck_mdp.hs_overshoot_penalty(env)[0]
            )
        return total

    stop_on_the_mark = episode(0.0, 0)
    v1_415 = episode(55.0, 20)             # 55° past: v1's BEST seed
    v1_457 = episode(97.0, 20)             # 97° past: v1's worst seed

    assert stop_on_the_mark > 600           # ~4.0 × 1.0 × 160
    assert v1_415 < -150 and v1_457 < -250  # over-rotation is no longer free
    assert stop_on_the_mark - v1_415 > 400
    assert stop_on_the_mark - v1_457 > 900
