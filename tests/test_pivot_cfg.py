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
    PIN_TIGHTEN_ITER,
    PIVOT_DIRECTION,
    PV_DIRECTION_CCW,
    PV_DIRECTION_CW,
    PV_TARGET_YAW,
    W_PIN_DISPLACEMENT,
    W_PIN_DISPLACEMENT_START,
    MicroduckPivotRlCfg,
    make_microduck_pivot_env_cfg,
)

# The per-step pay cap. Ticking at exactly this rate makes every rate-scaled
# term (progress, and now the pin) score its full 1.0, which is what the
# geometry assertions below want to isolate.
_CAP = microduck_mdp.PV_RATE_CAP


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
                 "stall", "counter_yaw",
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
    spin = W_PIN_DISPLACEMENT + cfg.rewards["pin_slip"].weight   # tightened value
    assert pivot - spin >= 5.0, "a two-footed spin must lose by a wide margin"
    # the turn itself still outweighs the settle it buys
    assert cfg.rewards["pivot_progress"].weight >= cfg.rewards["pin"].weight
    # ...and the paddle is worth lifting the free foot for from step one: it
    # must at least cover the displacement cost a first clumsy attempt incurs.
    assert cfg.rewards["paddle"].weight >= abs(W_PIN_DISPLACEMENT_START) * 2


def test_standing_still_loses():
    """The v1 argmax, priced with the current weights. Standing collects the
    stall cost and nothing else; pivoting collects progress, pin and paddle."""
    cfg = make_microduck_pivot_env_cfg()
    w = {k: cfg.rewards[k].weight for k in
         ("pivot_progress", "pin", "paddle", "stall")}
    rate = (2.0 * math.pi) / microduck_mdp.PV_RATE_CAP   # a 1 turn/s pivot
    still = w["stall"] * 1.0                             # pin/progress pay 0
    pivoting = (w["pivot_progress"] + w["pin"]) * rate + w["paddle"] * 0.5
    assert still < 0.0, "standing still must LOSE, not merely fail to win"
    assert pivoting - still >= 8.0, (still, pivoting)


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
    assert math.isclose(_CAP, 1.5 * 2 * math.pi), "1.5 turns/s"
    env.tick(_CAP / 2)                    # half the cap rate
    assert math.isclose(float(microduck_mdp.pv_progress_reward(env)[0]), 0.5, rel_tol=1e-6)
    env.tick(_CAP)                        # exactly the cap
    assert math.isclose(float(microduck_mdp.pv_progress_reward(env)[0]), 1.0, rel_tol=1e-6)
    env.tick(50.0)                        # way past it: still 1.0, no extra pay
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
    env.tick(_CAP)                        # turning at the pay cap: pin scores 1
    assert float(microduck_mdp.pv_pin_reward(env)[0]) > 0.99
    assert float(microduck_mdp.pv_pin_displacement_penalty(env)[0]) == 0.0
    # slide the pin 1 mm/step for 20 steps = 2 cm: well past PV_PIN_STD_M.
    # Half the cap rate, so the turn does not finish and tip the costs into
    # their settle-phase gate before the geometry has been checked.
    _turn_for(env, _CAP / 2, 0.4, pin_dxy=(0.001, 0.0))
    assert math.isclose(float(env._pv_pin_d[0]), 0.02, rel_tol=1e-5)
    assert bool(env._pv_done[0]) is False
    assert float(microduck_mdp.pv_pin_reward(env)[0]) < 0.2
    cost = float(microduck_mdp.pv_pin_displacement_penalty(env)[0])
    assert 0.0 < cost <= 1.0
    # a full foot length away and the cost is saturated
    _turn_for(env, _CAP / 2, 0.4, pin_dxy=(0.002, 0.0))
    assert float(microduck_mdp.pv_pin_reward(env)[0]) < 1e-6
    assert float(microduck_mdp.pv_pin_displacement_penalty(env)[0]) == 1.0


def test_pin_costs_stop_at_the_settle():
    """"Planted" is a constraint on the TURN. Charging the tightened -2.0 for
    every step of the ~150-step settle would swamp the settle pay and make
    finishing the trick worse than never finishing it."""
    env = _FakeEnv()
    _turn_for(env, _CAP, 0.5, pin_dxy=(0.002, 0.0))    # drag the pin 5 cm
    assert float(env._pv_pin_d[0]) > microduck_mdp.PV_PIN_SAT_M
    assert float(microduck_mdp.pv_pin_displacement_penalty(env)[0]) == 1.0
    _turn_for(env, _CAP, 0.6)                          # ...and finish the turn
    assert bool(env._pv_done[0]) is True
    assert float(env._pv_pin_d[0]) > microduck_mdp.PV_PIN_SAT_M, "still moved"
    assert float(microduck_mdp.pv_pin_displacement_penalty(env)[0]) == 0.0


def test_pin_pays_nothing_while_standing_still():
    """THE v1 defect. Contact 1 x stay 1 used to pay the pin its full weight to
    a robot that never moved, which made standing the argmax of the stack. The
    pay is now scaled by the step's progress: no turn, no pin."""
    env = _FakeEnv()
    for _ in range(50):                   # a full second of perfect stillness
        env.tick(0.0)
    assert float(env._pv_pin_d[0]) == 0.0, "planted, in contact, upright"
    assert float(env.sensor.data.found[0, _LEFT]) == 1.0
    assert float(microduck_mdp.pv_pin_reward(env)[0]) == 0.0
    # and it scales with the turn rather than switching on
    env.tick(_CAP / 2)
    assert math.isclose(float(microduck_mdp.pv_pin_reward(env)[0]), 0.5, rel_tol=1e-6)
    env.tick(_CAP)
    assert math.isclose(float(microduck_mdp.pv_pin_reward(env)[0]), 1.0, rel_tol=1e-6)
    # over-turning past the full turn banks nothing, so the pin stops paying too
    _turn_for(env, _CAP, 1.0)
    assert bool(env._pv_done[0]) is True
    env.tick(_CAP)
    assert float(microduck_mdp.pv_pin_reward(env)[0]) == 0.0


def test_pin_term_fires_on_lost_contact():
    env = _FakeEnv()
    env.tick(_CAP)
    assert float(microduck_mdp.pv_pin_reward(env)[0]) > 0.99
    env.set_contact(_LEFT, False)         # the pin leaves the floor
    env.tick(_CAP)
    assert float(microduck_mdp.pv_pin_reward(env)[0]) == 0.0
    env.set_contact(_LEFT, True)
    env.tick(_CAP)
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


# ── the direction sign, end to end ───────────────────────────────────────────

def test_progress_sign_follows_the_command_flag():
    """The chain that decides which way the robot is paid to turn:
    `PivotCommand` writes ±1 into the twist yaw slot (obs[50]) → `_pv_update`
    latches it as `_pv_dir` → `_pv_omega` = dir × root_link_ang_vel_b[:, 2]
    (body-frame z, positive = counter-clockwise from above) → the potential
    integrates `_pv_omega`.

    Commanded +1 with a POSITIVE body yaw rate must earn; commanded -1 with a
    NEGATIVE body yaw rate must earn the same. Turning the other way earns
    nothing in either case.
    """
    for flag, right_way in ((PV_DIRECTION_CCW, +1.0), (PV_DIRECTION_CW, -1.0)):
        env = _FakeEnv(yaw_flag=flag)
        earned = 0.0
        for _ in range(25):                       # half a second
            env.tick(right_way * _CAP)
            earned += float(microduck_mdp.pv_progress_reward(env)[0])
        assert float(env._pv_dir[0]) == flag
        assert float(env._pv_omega[0]) > 0.0, "signed rate: + is the way asked"
        assert float(env._pv_yaw[0]) > 0.0, f"flag {flag} banked no progress"
        assert math.isclose(earned, 25.0, rel_tol=1e-6), "full pay every step"

        wrong = _FakeEnv(yaw_flag=flag)
        paid = 0.0
        for _ in range(25):
            wrong.tick(-right_way * _CAP)
            paid += float(microduck_mdp.pv_progress_reward(wrong)[0])
        assert float(wrong._pv_omega[0]) < 0.0
        assert float(wrong._pv_yaw[0]) == 0.0
        assert paid == 0.0, f"flag {flag} was paid for turning the wrong way"


def test_progress_sign_holds_for_a_pinned_direction():
    """Same chain with `direction=±1` forced (the one-way task), which bypasses
    the command read entirely — the two paths must agree."""
    for forced, right_way in ((PV_DIRECTION_CCW, +1.0), (PV_DIRECTION_CW, -1.0)):
        env = _FakeEnv(yaw_flag=-forced)          # flag says the OPPOSITE
        for _ in range(10):
            env.common_step_counter += 1
            env.episode_length_buf += 1
            env._asset.data.root_link_ang_vel_b[:, 2] = right_way * _CAP
            microduck_mdp._pv_update(env, direction=forced)
        assert float(env._pv_dir[0]) == forced, "the forced sign wins"
        assert float(env._pv_yaw[0]) > 0.0


def test_counter_rotation_is_charged():
    """THE SIGN FIX. Before this term the flag was one-sided: the commanded
    direction paid and the opposite direction was free, so the only
    direction-dependent pressure on a policy that had not found the turn was
    the pin/paddle loading — which yaws the trunk AGAINST the command (measured
    -21°/-17° on +1, +28°/+33° on -1 at checkpoint 2499)."""
    for flag, right_way in ((PV_DIRECTION_CCW, +1.0), (PV_DIRECTION_CW, -1.0)):
        env = _FakeEnv(yaw_flag=flag)
        env.tick(right_way * microduck_mdp.PV_COUNTER_CAP)
        assert float(microduck_mdp.pv_counter_yaw_penalty(env)[0]) == 0.0

        env.tick(-right_way * microduck_mdp.PV_COUNTER_CAP)
        assert float(microduck_mdp.pv_counter_yaw_penalty(env)[0]) == 1.0
        env.tick(-right_way * microduck_mdp.PV_COUNTER_CAP / 2)
        assert math.isclose(
            float(microduck_mdp.pv_counter_yaw_penalty(env)[0]), 0.5, rel_tol=1e-6
        )
        env.tick(-right_way * 50.0)                    # bounded, no jackpot
        assert float(microduck_mdp.pv_counter_yaw_penalty(env)[0]) == 1.0

    # ...and it is OFF once the turn is done: stopping is the settle's job, and
    # decelerating out of a turn overshoots the other way.
    env = _FakeEnv()
    _turn_for(env, _CAP, 1.0)
    assert bool(env._pv_done[0]) is True
    env.tick(-_CAP)
    assert float(microduck_mdp.pv_counter_yaw_penalty(env)[0]) == 0.0


def test_wrong_way_yaw_cannot_bury_the_potential():
    """The raw integral is floored at zero. Un-floored, a wrong-way wobble dug a
    hole the policy had to climb back out of before a single degree of progress
    paid again — so the cheapest response to an accidental counter-rotation was
    to stop trying, which is what the stalled policy did."""
    env = _FakeEnv()
    _turn_for(env, -_CAP, 1.0)                    # a full second the wrong way
    assert float(env._pv_raw[0]) == 0.0, "floored, not -3pi"
    paid = 0.0
    for _ in range(5):                            # turning the right way now
        env.tick(_CAP)
        paid += float(microduck_mdp.pv_progress_reward(env)[0])
    assert math.isclose(paid, 5.0, rel_tol=1e-6), "progress pays immediately"


def test_stall_cost_fires():
    """Bounded penalty when the turn has not advanced for more than 0.5 s
    during the spin phase."""
    env = _FakeEnv()
    env.tick(0.0)
    assert float(microduck_mdp.pv_stall_penalty(env)[0]) == 0.0
    for _ in range(24):                           # 0.48 s of nothing: grace
        env.tick(0.0)
    assert float(env._pv_stall_s[0]) < microduck_mdp.PV_STALL_GRACE_S
    assert float(microduck_mdp.pv_stall_penalty(env)[0]) == 0.0

    for _ in range(13):                           # past the grace, ramping
        env.tick(0.0)
    mid = float(microduck_mdp.pv_stall_penalty(env)[0])
    assert 0.0 < mid < 1.0, mid
    for _ in range(50):                           # saturated and BOUNDED
        env.tick(0.0)
    assert float(microduck_mdp.pv_stall_penalty(env)[0]) == 1.0

    env.tick(_CAP)                                # one step of turn clears it
    assert float(env._pv_stall_s[0]) == 0.0
    assert float(microduck_mdp.pv_stall_penalty(env)[0]) == 0.0

    # and standing still is the GOAL once the turn is done: no stall cost there
    _turn_for(env, _CAP, 1.0)
    assert bool(env._pv_done[0]) is True
    for _ in range(100):
        env.tick(0.0)
    assert float(microduck_mdp.pv_stall_penalty(env)[0]) == 0.0

    # ...and it re-arms with the episode
    env.episode_length_buf[:] = 0
    env.tick(0.0)
    assert float(env._pv_stall_s[0]) == 0.0


# ── discovery curricula ──────────────────────────────────────────────────────

def test_pin_curriculum_stages_parse():
    """Loose pin while the turn is being discovered, tight once it exists —
    and the pay and the cost tighten on the SAME iterations."""
    cfg = make_microduck_pivot_env_cfg()
    std_stages = cfg.curriculum["pin_std"].params["std_stages"]
    w_stages = cfg.curriculum["pin_displacement_weight"].params["weight_stages"]

    assert cfg.curriculum["pin_std"].func is microduck_mdp.pv_pin_std_curriculum
    assert cfg.curriculum["pin_displacement_weight"].func is microduck_mdp.reward_weight
    assert cfg.curriculum["pin_std"].params["reward_name"] == "pin"
    assert cfg.curriculum["pin_displacement_weight"].params["reward_name"] == (
        "pin_displacement"
    )

    # start loose, end at the measured values, monotone in between
    assert std_stages[0]["std"] == 0.04 and std_stages[-1]["std"] == 0.015
    assert w_stages[0]["weight"] == W_PIN_DISPLACEMENT_START == -0.5
    assert w_stages[-1]["weight"] == W_PIN_DISPLACEMENT == -2.0
    assert [s["std"] for s in std_stages] == sorted(
        (s["std"] for s in std_stages), reverse=True
    )
    assert [s["weight"] for s in w_stages] == sorted(
        (s["weight"] for s in w_stages), reverse=True
    ), "the displacement cost only ever gets heavier"
    # steps are env steps (iteration x 24), strictly increasing, aligned
    for stages, key in ((std_stages, "std"), (w_stages, "weight")):
        steps = [s["step"] for s in stages]
        assert steps == sorted(steps) and len(set(steps)) == len(steps)
        assert steps[0] == 0
        assert steps[-1] == PIN_TIGHTEN_ITER * 24 == 1500 * 24
        assert all(isinstance(s[key], float) for s in stages)
    assert [s["step"] for s in std_stages] == [s["step"] for s in w_stages]

    # the term the curriculum drives starts where stage 0 says it does
    assert cfg.rewards["pin"].params["pin_std"] == std_stages[0]["std"]
    assert cfg.rewards["pin_displacement"].weight == w_stages[0]["weight"]


def test_pin_curriculum_applies_through_the_manager():
    """AGENTS.md: mutate term cfgs via the managers — writes to env.cfg are
    silent no-ops."""
    stages = [{"step": 0, "std": 0.04}, {"step": 100, "std": 0.015}]
    term = types.SimpleNamespace(params={"pin_std": 0.0})
    env = types.SimpleNamespace(
        common_step_counter=0,
        reward_manager=types.SimpleNamespace(get_term_cfg=lambda name: term),
    )
    microduck_mdp.pv_pin_std_curriculum(env, None, "pin", stages)
    assert term.params["pin_std"] == 0.04
    env.common_step_counter = 101
    microduck_mdp.pv_pin_std_curriculum(env, None, "pin", stages)
    assert term.params["pin_std"] == 0.015


# ── the arithmetic, run rather than asserted in prose ────────────────────────

_PIVOT_TERMS = (
    "pivot_progress", "pivot_complete", "pin", "paddle", "settle",
    "pin_displacement", "pin_slip", "stall", "counter_yaw",
)


def _score(env, weights):
    """Weighted sum of the pivot-specific terms for the current step.

    height_stand / upright / head_pose_tracking are common to every strategy
    and cancel, so they are left out (and the fake env has no bodies to
    resolve them against).
    """
    fns = {
        "pivot_progress": microduck_mdp.pv_progress_reward,
        "pivot_complete": microduck_mdp.pv_complete_bonus,
        "pin": microduck_mdp.pv_pin_reward,
        "paddle": microduck_mdp.pv_paddle_reward,
        "settle": microduck_mdp.pv_settle_reward,
        "pin_displacement": microduck_mdp.pv_pin_displacement_penalty,
        "pin_slip": microduck_mdp.pv_pin_slip_penalty,
        "stall": microduck_mdp.pv_stall_penalty,
        "counter_yaw": microduck_mdp.pv_counter_yaw_penalty,
    }
    return sum(weights[n] * float(fns[n](env)[0]) for n in _PIVOT_TERMS)


def _episode(strategy, weights, steps=200):
    """Run one 4 s episode of `strategy` and return the total pivot reward."""
    env = _FakeEnv()
    total = 0.0
    turn_rate = 2 * math.pi                       # 1 turn/s
    for k in range(steps):
        done = bool(env._pv_done[0]) if hasattr(env, "_pv_done") else False
        if strategy == "still":
            env.tick(0.0)
        elif strategy == "pivot":
            if done:
                env.set_contact(_RIGHT, True)
                env.tick(0.0)
            else:
                # the free foot alternates: 0.2 s up, 0.1 s down, pin planted
                env.set_contact(_RIGHT, (k % 15) >= 10)
                env.tick(turn_rate)
        elif strategy == "spin":
            # both feet on the floor, both sliding: the pin travels
            env.tick(0.0 if done else turn_rate,
                     pin_dxy=(0.0, 0.0) if done else (0.002, 0.0))
        total += _score(env, weights)
    return total


def test_the_arithmetic_pivoting_wins():
    """still vs pivoting vs a two-footed spin, priced with the live weights on
    the fake env — the per-step table in the cfg docstring, executed."""
    cfg = make_microduck_pivot_env_cfg()
    weights = {n: cfg.rewards[n].weight for n in _PIVOT_TERMS}
    weights["pin_displacement"] = W_PIN_DISPLACEMENT   # the tightened value

    still = _episode("still", weights)
    pivot = _episode("pivot", weights)
    spin = _episode("spin", weights)

    assert still < 0.0, f"standing still must LOSE outright, scored {still}"
    assert pivot > spin > still, (pivot, spin, still)
    assert pivot - still > 1000.0, "turning must beat standing by a landslide"
    assert pivot - spin > 200.0, "a pivot must beat a two-footed spin clearly"

    # the wrong-way wiggle checkpoint 2499 actually settled into: it collects
    # nothing, and now pays for both the stalling and the direction.
    env = _FakeEnv()
    wrong = 0.0
    for _ in range(200):
        env.tick(-0.4)                            # ~-23 deg/s, as measured
        wrong += _score(env, weights)
    assert wrong < still, (wrong, still)


def test_the_loose_curriculum_still_prefers_a_pivot():
    """At iteration 0 the pin costs are at their loose settings — a spin is
    cheapest there, so check the ranking holds at BOTH ends of the ramp."""
    cfg = make_microduck_pivot_env_cfg()
    weights = {n: cfg.rewards[n].weight for n in _PIVOT_TERMS}   # loose (-0.5)
    assert weights["pin_displacement"] == W_PIN_DISPLACEMENT_START
    assert _episode("pivot", weights) > _episode("spin", weights) > _episode(
        "still", weights
    )
