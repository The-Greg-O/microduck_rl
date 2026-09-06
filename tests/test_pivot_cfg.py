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
    W_ALIVE,
    W_COMPLETE,
    W_PIN_BROKEN,
    W_PIN_BROKEN_START,
    W_BODY_ANG_VEL,
    W_PIN_DISPLACEMENT,
    W_PIN_DISPLACEMENT_START,
    W_PROGRESS,
    W_SETTLE,
    W_TERMINATED,
    W_TILT,
    MicroduckPivotRlCfg,
    make_microduck_pivot_env_cfg,
)

# The per-step pay cap (v5: 0.5 turns/s — a 2 s turn — down from v4's 0.75).
# Ticking at exactly this rate makes every rate-scaled term (progress, and the
# pin) score its full 1.0, which is what the geometry assertions below isolate.
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
    for pos in ("pivot_progress", "pivot_complete", "pin", "paddle",
                "paddle_reach", "settle",
                "height_stand", "upright", "head_pose_tracking"):
        assert cfg.rewards[pos].weight > 0, pos
    for cost in ("pin_displacement", "pin_slip", "pin_broken",
                 "two_feet_still", "pivot_radius", "tilt", "terminated",
                 "stall", "counter_yaw",
                 "body_ang_vel", "action_rate_l2", "dof_pos_limits",
                 "self_collisions"):
        assert cfg.rewards[cost].weight < 0, cost
    # action rate stays LIGHT during discovery and ramps gently, late
    assert cfg.rewards["action_rate_l2"].weight == -0.05
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert [s["weight"] for s in stages] == [-0.05, -0.10, -0.20]
    assert stages[1]["step"] >= 1600 * 24   # v4: stretched with the run


def test_v3_terms_exist_with_the_stated_thresholds():
    """The three v3 additions, wired with the measured numbers: the pin is a
    hard requirement, the paddle has to reach, the shuffle is charged."""
    cfg = make_microduck_pivot_env_cfg()

    assert cfg.rewards["pin_broken"].func is microduck_mdp.pv_pin_broken_penalty
    assert cfg.rewards["pin_broken"].weight == W_PIN_BROKEN_START == -1.0
    assert W_PIN_BROKEN == -3.0
    assert microduck_mdp.PV_PIN_BREAK_AIR_S == 0.04     # 40 ms off the floor
    assert microduck_mdp.PV_PIN_BREAK_M == 0.03         # ...or 3 cm from anchor
    assert microduck_mdp.PV_PIN_REPLANT_M == 0.02       # repaired within 2 cm
    assert microduck_mdp.PV_PIN_REPLANT_M < microduck_mdp.PV_PIN_BREAK_M, (
        "hysteresis: the repair band must be tighter than the break band"
    )

    reach = cfg.rewards["paddle_reach"]
    assert reach.func is microduck_mdp.pv_paddle_reach_reward
    assert reach.weight == 3.0
    assert reach.params["reach_min"] == microduck_mdp.PV_REACH_MIN_M == 0.04
    assert reach.params["reach_max"] == microduck_mdp.PV_REACH_MAX_M == 0.08
    # the stroke window floor rose so a shuffle's twitch no longer clears it
    assert microduck_mdp.PV_MIN_AIR_S == 0.08
    assert cfg.rewards["paddle"].params["min_air"] == 0.08
    assert cfg.rewards["paddle"].params["max_air"] == 0.45

    two = cfg.rewards["two_feet_still"]
    assert two.func is microduck_mdp.pv_two_feet_still_penalty
    assert two.weight == -2.0
    assert two.params["yaw_gate"] == microduck_mdp.PV_TWOFOOT_YAW_GATE == 0.5
    assert two.params["yaw_cap"] > two.params["yaw_gate"]


def test_v4_terms_price_staying_up():
    """The v4 changes that SURVIVE into v5: the fall is charged, the progress
    potential is gated tightly on upright, the paddle window is wide, and the
    two lean/rock costs are felt. (v5 re-sizes the fall penalty and the pay cap;
    those are `test_v5_makes_staying_up_worth_more_than_any_turn`.)"""
    cfg = make_microduck_pivot_env_cfg()

    term = cfg.rewards["terminated"]
    assert term.func is microduck_mdp.pv_fall_penalty
    assert term.params["term_names"] == ("fell_over",)
    assert "nan_state" not in term.params["term_names"], (
        "a sim blow-up is not something the policy chose"
    )
    # the termination it prices has to be a real, non-timeout termination
    assert cfg.terminations["fell_over"].time_out is False

    # the progress gate: 15 deg, and STRICTLY tighter than the fallen gate it
    # replaces for this purpose (40 deg is where the fall is already lost).
    assert microduck_mdp.PV_PROGRESS_TILT_DEG == 15.0
    assert (microduck_mdp.PV_PROGRESS_TILT_DEG
            < microduck_mdp.PV_UPRIGHT_GATE_DEG == 40.0)

    # the stroke window: wide enough for a slow turn's long strokes
    assert microduck_mdp.PV_MIN_AIR_S == 0.08
    assert microduck_mdp.PV_MAX_AIR_S == 0.45

    assert cfg.rewards["tilt"].weight == W_TILT == -2.0
    assert cfg.rewards["body_ang_vel"].weight == W_BODY_ANG_VEL == -0.2

    assert MicroduckPivotRlCfg.max_iterations == 4_000


def test_v5_makes_staying_up_worth_more_than_any_turn():
    """THE v5 CHANGE, as an invariant rather than as a table.

    v4's log went the wrong way: iteration 1000 had mean episode length 191 of
    200 with 1.6 falls/iter, iteration 3999 had length 77 with 47 falls/iter and
    `pivot_progress` RISING the whole way (0.59 -> 0.82 -> 0.90). The policy was
    not failing to learn; it was learning that a turn that falls at ~0.9 s
    out-earns a careful one. v5 removes that trade by construction, and the
    construction is checkable: the ALIVE income of a surviving episode must
    exceed the ENTIRE progress budget of a perfect turn.
    """
    cfg = make_microduck_pivot_env_cfg()
    steps = int(round(EPISODE_LENGTH_S / 0.02))          # 200

    # (a) the alive income, upright-gated at the same 15 deg as the potential
    alive = cfg.rewards["alive"]
    assert alive.func is microduck_mdp.pv_alive_reward
    assert alive.weight == W_ALIVE == 3.0
    assert microduck_mdp.PV_ALIVE_TILT_DEG == 15.0
    assert alive.params["gate_tilt_above_deg"] == microduck_mdp.PV_ALIVE_TILT_DEG

    # THE ARITHMETIC THAT DEFINES v5. A full turn at the pay cap takes
    # target/cap seconds and pays W_PROGRESS for each of those steps; surviving
    # pays W_ALIVE for all 200. Surviving has to be worth more.
    turn_steps = (PV_TARGET_YAW / microduck_mdp.PV_RATE_CAP) / 0.02
    progress_budget = W_PROGRESS * turn_steps
    alive_income = W_ALIVE * steps
    assert alive_income > progress_budget, (alive_income, progress_budget)
    assert alive_income == 600.0 and progress_budget == 400.0

    # (b) the fall, charged against the term it competes with
    assert cfg.rewards["terminated"].weight == W_TERMINATED == -100.0
    # ...and still not a jackpot: one step in 200, smaller than the income the
    # same fall forfeits.
    forfeited = (W_ALIVE + cfg.rewards["upright"].weight
                 + cfg.rewards["height_stand"].weight) * (steps - 45)
    assert forfeited > 2 * abs(W_TERMINATED), forfeited

    # (c) the pay cap, halved again: full pay on a 2 s turn, not a 1.33 s one
    assert math.isclose(microduck_mdp.PV_RATE_CAP, 0.5 * 2 * math.pi)
    assert math.isclose(PV_TARGET_YAW / microduck_mdp.PV_RATE_CAP, 2.0)
    assert cfg.rewards["pivot_progress"].weight == W_PROGRESS == 4.0
    # half the episode is left for the settle
    assert turn_steps == steps / 2

    # (d) the pin curriculum stops SHORT and stops EARLY, then holds
    assert microduck_mdp.PV_PIN_STD_FLOOR_M == 0.025
    assert microduck_mdp.PV_PIN_STD_FLOOR_M > microduck_mdp.PV_PIN_STD_M, (
        "v5 deliberately never reaches the measured 1.5 cm"
    )
    assert W_PIN_DISPLACEMENT == -1.5
    assert PIN_TIGHTEN_ITER == 1600
    assert PIN_TIGHTEN_ITER < 0.5 * MicroduckPivotRlCfg.max_iterations, (
        "the back half of the run consolidates rather than tightens"
    )
    std_stages = cfg.curriculum["pin_std"].params["std_stages"]
    assert std_stages[-1]["std"] == microduck_mdp.PV_PIN_STD_FLOOR_M
    assert std_stages[-1]["step"] == PIN_TIGHTEN_ITER * 24

    # (e) the jackpot moved to FINISHING UPRIGHT
    assert cfg.rewards["pivot_complete"].weight == W_COMPLETE == 15.0
    assert cfg.rewards["settle"].weight == W_SETTLE == 6.0
    assert microduck_mdp.PV_FINISH_TILT_DEG == 15.0
    for name in ("pivot_complete", "settle"):
        assert cfg.rewards[name].params["gate_tilt_above_deg"] == (
            microduck_mdp.PV_FINISH_TILT_DEG
        ), name


def test_the_fall_penalty_is_one_shot_and_charges_only_the_fall():
    env = _FakeEnv()
    env.tick(0.0)
    assert float(microduck_mdp.pv_fall_penalty(env)[0]) == 0.0
    env.termination_manager.fire("nan_state")
    assert float(microduck_mdp.pv_fall_penalty(env)[0]) == 0.0, (
        "a NaN guard is a sim blow-up, not a behaviour"
    )
    env.termination_manager.fire("out_of_terrain_bounds")
    assert float(microduck_mdp.pv_fall_penalty(env)[0]) == 0.0, (
        "the trunk of a pivoting robot is SUPPOSED to travel"
    )
    env.termination_manager.fire("fell_over")
    assert float(microduck_mdp.pv_fall_penalty(env)[0]) == 1.0
    # a term the env does not have must not raise (the base cfg owns the names)
    assert float(microduck_mdp.pv_fall_penalty(env, term_names=("nope",))[0]) == 0.0


def test_the_progress_potential_freezes_past_fifteen_degrees():
    """The v4 gate on the potential itself, not on a term's output: degrees
    turned while toppling are not progress, and nothing banked is burned."""
    env = _FakeEnv()
    _turn_for(env, _CAP, 0.4)                     # 0.3 turns, upright
    banked = float(env._pv_yaw[0])
    assert banked > 0.0

    env.set_tilt(20.0)                            # past 15, well short of 40
    for _ in range(50):                           # 1 s: past the stall grace
        env.tick(_CAP)
        assert float(microduck_mdp.pv_progress_reward(env)[0]) == 0.0
        assert float(microduck_mdp.pv_pin_reward(env)[0]) == 0.0
    assert float(env._pv_yaw[0]) == banked, "banked degrees are never burned"
    # ...and the stall timer runs while the potential is frozen, so leaning is
    # not a free parking spot either.
    assert float(microduck_mdp.pv_stall_penalty(env)[0]) > 0.0

    env.set_tilt(10.0)                            # back inside the gate
    env.tick(_CAP)
    assert float(env._pv_yaw[0]) > banked
    assert float(microduck_mdp.pv_progress_reward(env)[0]) > 0.0


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


def test_standing_still_loses_to_pivoting_but_beats_falling():
    """The v1 argmax, priced with the current weights — and the v5 correction to
    what "loses" means.

    v4 wanted standing still to be NEGATIVE, and it was: the stall bill and
    nothing else. That is exactly the shape that made falling attractive, since
    ending the episode also ends the bill. v5 pays a large constant `alive`
    income instead, so standing still is comfortably in profit — it just earns
    far less than a pivot. The task-specific stack (progress, pin, paddle,
    stall) still ranks pivoting far above standing; the alive income is added to
    both and cancels, which is precisely why it is safe to make it this large.
    """
    cfg = make_microduck_pivot_env_cfg()
    w = {k: cfg.rewards[k].weight for k in
         ("pivot_progress", "pin", "paddle", "stall", "alive")}
    rate = 1.0                                           # a pivot at the pay cap
    still = w["stall"] * 1.0                             # pin/progress pay 0
    pivoting = (w["pivot_progress"] + w["pin"]) * rate + w["paddle"] * 0.5
    assert still < 0.0, "the stall bill still makes doing nothing the worse deal"
    assert pivoting - still >= 8.0, (still, pivoting)

    # ...and the alive income, which both collect, is what stops the stall bill
    # from making a fall look like an escape: it is bigger than the bill.
    assert w["alive"] + still > 0.0, (
        "standing still must be in PROFIT, or ending the episode is an exit"
    )


def test_terminates_on_fall():
    cfg = make_microduck_pivot_env_cfg()
    assert cfg.terminations["fell_over"].time_out is False


def test_symmetry_off_and_runner_cfg():
    # the pin foot and the paddle foot have different roles: no mirror prior
    assert MicroduckPivotRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckPivotRlCfg.actor.obs_normalization is True
    assert MicroduckPivotRlCfg.critic.obs_normalization is True
    assert MicroduckPivotRlCfg.experiment_name == "pivot"
    assert MicroduckPivotRlCfg.max_iterations == 4_000


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


class _FakeTerminations:
    """Just enough of mjlab's TerminationManager for the v4 fall penalty."""

    def __init__(self, n):
        self.active_terms = ["fell_over", "out_of_terrain_bounds", "nan_state"]
        self._dones = {k: torch.zeros(n, dtype=torch.bool) for k in self.active_terms}

    def get_term(self, name):
        return self._dones[name]

    def fire(self, name):
        self._dones[name][:] = True


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
        self.termination_manager = _FakeTerminations(n)

    @property
    def sensor(self):
        return self.scene.sensors["feet_ground_contact"]

    def set_contact(self, foot, down):
        self.sensor.data.found[:, foot] = 1.0 if down else 0.0
        if down:
            self.sensor.data.current_air_time[:, foot] = 0.0

    def set_tilt(self, deg):
        """Roll the trunk `deg` degrees about x — what a topple looks like to
        every tilt-aware gate in the stack."""
        half = math.radians(deg) / 2.0
        self._asset.data.root_link_quat_w[:] = torch.tensor(
            [math.cos(half), math.sin(half), 0.0, 0.0]
        )

    def tick(self, omega_z=0.0, pin_dxy=(0.0, 0.0), trunk_dxy=(0.0, 0.0),
             free_dxy=(0.0, 0.0)):
        """Advance one env step: trunk yaw rate, pin-foot drift, trunk drift,
        and (v3) free-foot travel, which is what the paddle's reach scores."""
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self._asset.data.root_link_ang_vel_b[:, 2] = omega_z
        pin = int(self._pin_index())
        self._asset.data.site_pos_w[:, pin, 0] += pin_dxy[0]
        self._asset.data.site_pos_w[:, pin, 1] += pin_dxy[1]
        self._asset.data.site_pos_w[:, 1 - pin, 0] += free_dxy[0]
        self._asset.data.site_pos_w[:, 1 - pin, 1] += free_dxy[1]
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
    assert math.isclose(_CAP, 0.5 * 2 * math.pi), "v5: 0.5 turns/s"
    env.tick(_CAP / 2)                    # half the cap rate
    assert math.isclose(float(microduck_mdp.pv_progress_reward(env)[0]), 0.5, rel_tol=1e-6)
    env.tick(_CAP)                        # exactly the cap
    assert math.isclose(float(microduck_mdp.pv_progress_reward(env)[0]), 1.0, rel_tol=1e-6)
    env.tick(50.0)                        # way past it: still 1.0, no extra pay
    assert math.isclose(float(microduck_mdp.pv_progress_reward(env)[0]), 1.0, rel_tol=1e-6)


def test_progress_total_is_potential_based():
    """A completed turn is worth the same total however fast it is done, AT OR
    BELOW the pay cap — the speed incentive lives in the settle phase, not here.

    Above the cap the total FALLS, which is the v4 cap doing its job: at
    1 turn/s (v3's whip speed) a third of the turn's progress pay is simply
    forfeited, because the per-step clamp bites and the potential is spent."""
    totals = {}
    for omega in (math.pi / 2, math.pi, _CAP, 2 * math.pi):
        env = _FakeEnv()
        total = 0.0
        for _ in range(200):              # a full 4 s episode
            env.tick(omega)
            total += float(microduck_mdp.pv_progress_reward(env)[0])
        totals[omega] = total
    at_or_under = [totals[o] for o in (math.pi / 2, math.pi, _CAP)]
    assert math.isclose(min(at_or_under), max(at_or_under), rel_tol=1e-6)
    assert totals[2 * math.pi] < totals[_CAP] * 0.8, (
        "turning faster than the cap forfeits pay — v3's 1.5 turns/s cap was "
        "read as a target", totals
    )


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
    finishing the trick worse than never finishing it.

    The drag here stays under PV_PIN_BREAK_M: past 3 cm the v3 latch freezes the
    potential and the turn can never finish at all, which is the subject of
    `test_a_broken_pin_earns_no_progress`."""
    env = _FakeEnv()
    _turn_for(env, _CAP, 0.24, pin_dxy=(0.002, 0.0))   # drag the pin 2.4 cm
    assert 0.0 < float(env._pv_pin_d[0]) < microduck_mdp.PV_PIN_BREAK_M
    assert bool(env._pv_broken[0]) is False
    assert 0.0 < float(microduck_mdp.pv_pin_displacement_penalty(env)[0]) < 1.0
    _turn_for(env, _CAP, 2.0)                          # ...and finish the turn
    assert bool(env._pv_done[0]) is True
    assert float(env._pv_pin_d[0]) > 0.0, "still moved"
    assert float(microduck_mdp.pv_pin_displacement_penalty(env)[0]) == 0.0
    assert float(microduck_mdp.pv_pin_broken_penalty(env)[0]) == 0.0


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
    _turn_for(env, _CAP, 2.1)
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


# ── v3: the pin is a hard requirement ────────────────────────────────────────

def test_pin_breaks_on_lost_contact_past_the_tolerance():
    """40 ms of tolerance: a one-step contact flicker is free, two steps is a
    broken pin."""
    env = _FakeEnv()
    env.tick(_CAP)
    assert bool(env._pv_broken[0]) is False
    assert float(microduck_mdp.pv_pin_broken_penalty(env)[0]) == 0.0

    env.set_contact(_LEFT, False)
    env.tick(_CAP)                        # 0.02 s in the air: inside tolerance
    assert math.isclose(float(env._pv_pin_air_s[0]), 0.02, rel_tol=1e-5)
    assert bool(env._pv_broken[0]) is False
    env.tick(_CAP)                        # 0.04 s... still not PAST 0.04
    assert bool(env._pv_broken[0]) is False
    env.tick(_CAP)                        # 0.06 s: broken
    assert bool(env._pv_broken[0]) is True
    assert float(microduck_mdp.pv_pin_broken_penalty(env)[0]) == 1.0

    # putting it back down within 2 cm of the anchor repairs it, and only that
    env.set_contact(_LEFT, True)
    env.tick(_CAP)
    assert bool(env._pv_broken[0]) is False
    assert float(microduck_mdp.pv_pin_broken_penalty(env)[0]) == 0.0


def test_pin_breaks_on_displacement_and_needs_a_replant_to_repair():
    """Break at 3 cm, repair only at 2 cm — hysteresis, and measured against the
    ORIGINAL anchor, so "recover" means stepping back rather than re-declaring
    wherever the foot ended up to be the new pin."""
    env = _FakeEnv()
    _turn_for(env, _CAP, 0.28, pin_dxy=(0.001, 0.0))       # 1.4 cm out
    assert bool(env._pv_broken[0]) is False
    _turn_for(env, _CAP, 0.34, pin_dxy=(0.001, 0.0))       # ~3.1 cm out
    assert float(env._pv_pin_d[0]) > microduck_mdp.PV_PIN_BREAK_M
    assert bool(env._pv_broken[0]) is True
    assert float(microduck_mdp.pv_pin_broken_penalty(env)[0]) == 1.0

    # walking back to 2.5 cm is NOT enough: still outside the repair band
    _turn_for(env, _CAP, 0.12, pin_dxy=(-0.001, 0.0))      # ~1.9... check
    assert bool(env._pv_broken[0]) == (
        float(env._pv_pin_d[0]) > microduck_mdp.PV_PIN_REPLANT_M
    )
    # ...all the way back to the anchor and it is repaired
    _turn_for(env, _CAP, 0.4, pin_dxy=(-0.001, 0.0))
    env._asset.data.site_pos_w[:, _LEFT, :2] = env._pv_pin_home
    env.tick(_CAP)
    assert float(env._pv_pin_d[0]) < microduck_mdp.PV_PIN_REPLANT_M
    assert bool(env._pv_broken[0]) is False


def test_a_broken_pin_earns_no_progress():
    """THE v3 GATE, and the reason a shuffle-spin stops being viable: while the
    pin is broken the potential does not advance, so the turn simply does not
    count. Nothing banked is destroyed — the potential is a running maximum —
    it only stops accruing, and the stall timer starts running."""
    env = _FakeEnv()
    _turn_for(env, _CAP, 0.2)                    # 10 clean steps of turn
    banked = float(env._pv_yaw[0])
    assert banked > 0.0

    env.set_contact(_LEFT, False)                # break the pin
    _turn_for(env, _CAP, 0.5)
    assert bool(env._pv_broken[0]) is True
    assert float(env._pv_yaw[0]) == banked, "banked degrees survive, untouched"
    assert float(microduck_mdp.pv_progress_reward(env)[0]) == 0.0
    assert float(microduck_mdp.pv_pin_reward(env)[0]) == 0.0
    assert float(env._pv_stall_s[0]) > 0.0, "a frozen potential IS a stall"

    env.set_contact(_LEFT, True)                 # re-plant on the anchor
    env.tick(_CAP)
    assert bool(env._pv_broken[0]) is False
    assert math.isclose(
        float(microduck_mdp.pv_progress_reward(env)[0]), 1.0, rel_tol=1e-6
    ), "progress resumes immediately, no hole to climb out of"


def test_pin_broken_cost_is_bounded_and_spin_phase_only():
    env = _FakeEnv()
    _turn_for(env, _CAP, 2.1)                    # finish the turn cleanly
    assert bool(env._pv_done[0]) is True
    env.set_contact(_LEFT, False)                # lift the pin during the settle
    _turn_for(env, 0.0, 0.5)
    assert bool(env._pv_broken[0]) is True
    assert float(microduck_mdp.pv_pin_broken_penalty(env)[0]) == 0.0, (
        "the settle is where both feet come back under the robot"
    )


def test_pin_broken_is_free_before_the_spawn_has_landed():
    """The spawn transient is not a broken pin."""
    env = _FakeEnv()
    env.set_contact(_LEFT, False)
    env.set_contact(_RIGHT, False)
    _turn_for(env, _CAP, 0.5)                    # airborne the whole time
    assert bool(env._pv_landed[0]) is False
    assert bool(env._pv_broken[0]) is False
    assert float(microduck_mdp.pv_pin_broken_penalty(env)[0]) == 0.0


# ── v3: the paddle has to REACH ──────────────────────────────────────────────

def _stroke(env, seconds, free_dxy, omega=2.0):
    """Lift the free foot, carry it `free_dxy` per step, plant it."""
    env.set_contact(_RIGHT, False)
    for _ in range(int(round(seconds / env.step_dt))):
        env.tick(omega, free_dxy=free_dxy)
    env.set_contact(_RIGHT, True)
    env.tick(omega)
    return float(microduck_mdp.pv_paddle_reach_reward(env)[0])


def test_reach_pays_a_stroke_that_travels_and_not_a_tap():
    env = _FakeEnv()
    env.tick(2.0)
    # 10 steps x 6 mm = 6 cm: squarely inside the 4-8 cm plateau
    assert _stroke(env, 0.2, (0.0, 0.006)) > 0.99
    # a tap in place: the foot lifted, but it went nowhere
    tap = _stroke(env, 0.2, (0.0, 0.0))
    assert 0.0 < tap < 0.25, tap
    # a lunge way past the reach: bounded, and worth almost nothing
    lunge = _stroke(env, 0.2, (0.0, 0.015))       # 15 cm
    assert 0.0 <= lunge < 0.05, lunge
    # the band edges themselves pay full
    assert _stroke(env, 0.2, (0.0, 0.004)) > 0.99   # 4 cm
    assert _stroke(env, 0.2, (0.0, 0.008)) > 0.99   # 8 cm


def test_reach_is_one_shot_per_plant():
    """A per-step reach pay is farmable by parking the free foot out wide."""
    env = _FakeEnv()
    env.tick(2.0)
    assert _stroke(env, 0.2, (0.0, 0.006)) > 0.99
    paid = 0.0
    for _ in range(20):                           # stand on it for 0.4 s
        env.tick(2.0)
        paid += float(microduck_mdp.pv_paddle_reach_reward(env)[0])
    assert paid == 0.0, "the plant pays once; the next one must be re-earned"
    # and holding the foot out in the AIR pays nothing either
    env.set_contact(_RIGHT, False)
    held = 0.0
    for _ in range(50):
        env.tick(2.0)
        held += float(microduck_mdp.pv_paddle_reach_reward(env)[0])
    assert held == 0.0


def test_reach_ignores_contact_chatter():
    """A foot that never really leaves the floor has not taken a stroke: the
    plant edge needs PV_MIN_AIR_S of air behind it."""
    env = _FakeEnv()
    env.tick(2.0)
    paid = 0.0
    for _ in range(20):
        env.set_contact(_RIGHT, False)
        env.tick(2.0, free_dxy=(0.0, 0.03))       # 3 cm per flicker, chattering
        env.set_contact(_RIGHT, True)
        env.tick(2.0)
        paid += float(microduck_mdp.pv_paddle_reach_reward(env)[0])
    assert paid == 0.0, "0.02 s of air is not a stroke"


def test_reach_needs_the_pin_down_and_the_spin_phase():
    env = _FakeEnv()
    env.tick(2.0)
    env.set_contact(_LEFT, False)                 # a HOP: the pin is airborne
    assert _stroke(env, 0.2, (0.0, 0.006)) == 0.0
    env.set_contact(_LEFT, True)

    done = _FakeEnv()
    _turn_for(done, 2 * math.pi, 1.1)
    assert bool(done._pv_done[0]) is True
    assert _stroke(done, 0.2, (0.0, 0.006), omega=0.0) == 0.0


# ── v3: turning on two planted feet is the shuffle ───────────────────────────

def test_two_feet_still_charges_the_shuffle_and_not_a_stance():
    env = _FakeEnv()
    env.tick(0.0)                                 # both feet down, not turning
    assert float(microduck_mdp.pv_two_feet_still_penalty(env)[0]) == 0.0
    env.tick(0.4)                                 # under the 0.5 rad/s gate
    assert float(microduck_mdp.pv_two_feet_still_penalty(env)[0]) == 0.0

    gate = microduck_mdp.PV_TWOFOOT_YAW_GATE
    cap = microduck_mdp.PV_TWOFOOT_YAW_CAP
    env.tick(gate + 0.5 * (cap - gate))           # halfway up the ramp
    assert math.isclose(
        float(microduck_mdp.pv_two_feet_still_penalty(env)[0]), 0.5, rel_tol=1e-6
    )
    env.tick(1.5)                                 # the v2 shuffle's actual rate
    assert math.isclose(
        float(microduck_mdp.pv_two_feet_still_penalty(env)[0]),
        (1.5 - gate) / (cap - gate), rel_tol=1e-6,
    )
    env.tick(50.0)                                # bounded: no jackpot
    assert float(microduck_mdp.pv_two_feet_still_penalty(env)[0]) == 1.0
    # unsigned: turning the WRONG way on two feet is the same failure
    env.tick(-50.0)
    assert float(microduck_mdp.pv_two_feet_still_penalty(env)[0]) == 1.0


def test_two_feet_still_needs_both_feet_and_the_spin_phase():
    env = _FakeEnv()
    env.tick(50.0)
    assert float(microduck_mdp.pv_two_feet_still_penalty(env)[0]) == 1.0
    env.set_contact(_RIGHT, False)                # one foot up: this is a pivot
    env.tick(50.0)
    assert float(microduck_mdp.pv_two_feet_still_penalty(env)[0]) == 0.0

    env.set_contact(_RIGHT, True)
    _turn_for(env, _CAP, 2.1)
    assert bool(env._pv_done[0]) is True
    env.tick(50.0)                                # decelerating out of the turn
    assert float(microduck_mdp.pv_two_feet_still_penalty(env)[0]) == 0.0, (
        "stopping with both feet down is the settle's goal, not a shuffle"
    )


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
    # 0.2 mm/step for 1.2 s = 1.2 cm of pin creep: under PV_PIN_BREAK_M, so the
    # turn still completes (a broken pin freezes the potential — see
    # `test_a_broken_pin_earns_no_progress`).
    _turn_for(env, 2 * math.pi, 1.2, pin_dxy=(0.0002, 0.0))
    assert bool(env._pv_done[0]) is True and float(env._pv_yaw[0]) > 0.0
    assert float(env._pv_pin_d[0]) > 0.0
    home_before = env._pv_pin_home[0].clone()
    env.episode_length_buf[:] = 0          # reset
    env.tick(0.0)
    assert float(env._pv_yaw[0]) == 0.0
    assert float(env._pv_raw[0]) == 0.0
    assert bool(env._pv_done[0]) is False
    assert bool(env._pv_just_done[0]) is False
    assert bool(env._pv_broken[0]) is False
    assert float(env._pv_pin_air_s[0]) == 0.0
    assert float(env._pv_reach[0]) == 0.0
    assert bool(env._pv_planted[0]) is False
    assert float(env._pv_pin_d[0]) == 0.0, "the pin re-anchors where it now is"
    assert not torch.allclose(env._pv_pin_home[0], home_before)
    assert float(microduck_mdp.pv_pin_displacement_penalty(env)[0]) == 0.0
    assert float(microduck_mdp.pv_pin_slip_penalty(env)[0]) == 0.0
    assert float(microduck_mdp.pv_pin_broken_penalty(env)[0]) == 0.0


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
    _turn_for(env, _CAP, 2.1)
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
    _turn_for(env, _CAP, 2.1)
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

    # start loose, end at the v5 FLOOR (not the measured 1.5 cm / -2.0),
    # monotone in between
    assert std_stages[0]["std"] == 0.04 and std_stages[-1]["std"] == 0.025
    assert std_stages[-1]["std"] == microduck_mdp.PV_PIN_STD_FLOOR_M
    assert w_stages[0]["weight"] == W_PIN_DISPLACEMENT_START == -0.5
    assert w_stages[-1]["weight"] == W_PIN_DISPLACEMENT == -1.5
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
        assert steps[-1] == PIN_TIGHTEN_ITER * 24 == 1600 * 24
        assert all(isinstance(s[key], float) for s in stages)
    assert [s["step"] for s in std_stages] == [s["step"] for s in w_stages]

    # the term the curriculum drives starts where stage 0 says it does
    assert cfg.rewards["pin"].params["pin_std"] == std_stages[0]["std"]
    assert cfg.rewards["pin_displacement"].weight == w_stages[0]["weight"]

    # v3: the third pin pressure rides the SAME iterations. Tightening one pin
    # cost while the others are slack is just a tax on attempting.
    b_stages = cfg.curriculum["pin_broken_weight"].params["weight_stages"]
    assert cfg.curriculum["pin_broken_weight"].func is microduck_mdp.reward_weight
    assert cfg.curriculum["pin_broken_weight"].params["reward_name"] == "pin_broken"
    assert b_stages[0]["weight"] == W_PIN_BROKEN_START == -1.0
    assert b_stages[-1]["weight"] == W_PIN_BROKEN == -3.0
    assert [s["weight"] for s in b_stages] == sorted(
        (s["weight"] for s in b_stages), reverse=True
    )
    assert [s["step"] for s in b_stages] == [s["step"] for s in std_stages]
    assert cfg.rewards["pin_broken"].weight == b_stages[0]["weight"]
    # ...but the PROGRESS GATE that gives the term its teeth is never relaxed:
    # it lives in `_pv_update`, not in a weight, and there is no stage list for
    # it anywhere.
    assert "progress_gate" not in cfg.curriculum


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
    "pivot_progress", "pivot_complete", "pin", "paddle", "paddle_reach",
    "settle", "alive", "pin_displacement", "pin_slip", "pin_broken",
    "two_feet_still", "stall", "counter_yaw", "terminated",
)

# The terms every strategy collects identically — WHILE IT IS ALIVE AND UPRIGHT.
#
# v3's table left these out as "common to every row, they cancel", and that is
# exactly why v3's arithmetic never priced a fall. They cancel only between
# strategies that survive the whole episode. A strategy that TERMINATES forfeits
# them for every remaining step, and that forfeit is the larger half of what a
# fall actually costs: at +3.0/step, dying at 0.6 s throws away ~+510 of income
# that standing still collects in full. It is also, arithmetically, what v3's
# log was reporting — mean episode return 4.29 over 43 steps is 4.29/(43 x 0.02)
# = ~5.0 of stack per step, i.e. essentially this common income and nothing
# else, which is what a policy that pivots and falls at 0.87 s takes home.
#
# `upright` + `height_stand` only. head_pose_tracking is left out (it is a
# neck-posture term, not a survival term), which makes the comparison
# CONSERVATIVE: counting it would make every terminating strategy look worse.
def _weights(cfg):
    """The live weights for every term the arithmetic scores."""
    return {n: cfg.rewards[n].weight for n in _PIVOT_TERMS + ("tilt",)}


def _common_per_step(cfg):
    return cfg.rewards["upright"].weight + cfg.rewards["height_stand"].weight


def _score(env, weights, common=0.0):
    """Weighted sum of the pivot-specific terms for the current step, plus the
    common upright/height income the step earns (0 once the trunk is past the
    fallen gate — a toppling robot does not collect it either)."""
    fns = {
        "pivot_progress": microduck_mdp.pv_progress_reward,
        "pivot_complete": microduck_mdp.pv_complete_bonus,
        "paddle": microduck_mdp.pv_paddle_reward,
        "paddle_reach": microduck_mdp.pv_paddle_reach_reward,
        "settle": microduck_mdp.pv_settle_reward,
        # v5's decisive term: a constant +1 per alive, upright (15 deg) step.
        "alive": microduck_mdp.pv_alive_reward,
        # ...scored at the value the curriculum actually STOPS at (2.5 cm), not
        # at the function's geometric default, so the table prices the reward
        # the policy trains under for the back 60% of the run.
        "pin": lambda e: microduck_mdp.pv_pin_reward(
            e, pin_std=microduck_mdp.PV_PIN_STD_FLOOR_M
        ),
        "pin_displacement": microduck_mdp.pv_pin_displacement_penalty,
        "pin_slip": microduck_mdp.pv_pin_slip_penalty,
        "pin_broken": microduck_mdp.pv_pin_broken_penalty,
        "two_feet_still": microduck_mdp.pv_two_feet_still_penalty,
        "stall": microduck_mdp.pv_stall_penalty,
        "counter_yaw": microduck_mdp.pv_counter_yaw_penalty,
        "terminated": microduck_mdp.pv_fall_penalty,
    }
    total = sum(weights[n] * float(fns[n](env)[0]) for n in _PIVOT_TERMS)
    alive = float(microduck_mdp._pv_upright(
        env, microduck_mdp._DEFAULT_ASSET_CFG, microduck_mdp.PV_UPRIGHT_GATE_DEG
    )[0])
    return total + common * alive


# The tilt costs are the only common-stack terms that differ per strategy in a
# way that matters here, and the fake env can compute them exactly, so they are
# scored for real rather than assumed away.
def _tilt_cost(env, weights):
    return weights["tilt"] * float(microduck_mdp.pv_tilt_penalty(env)[0])


def _episode(strategy, weights, steps=200, common=0.0):
    """Run one 4 s episode of `strategy` and return the total reward.

    `pivot`    — 0.5 turns/s (the v5 rate cap: full pay, a 2 s turn), pin
                 planted and motionless, free foot on a 0.24 s up / 0.12 s down
                 cadence carrying its contact point 6 cm per stroke, trunk
                 upright throughout — and then ~2 s of settle.
    `shuffle`  — WHAT v2 LEARNED: both feet on the floor the whole time, both
                 sliding (the pin travels at 0.1 m/s), trunk yawing at
                 1.5 rad/s. Contact fractions 1.0 / 1.0 is the limit of the
                 measured 0.91 / 0.85, and 0.1 m/s of pin travel reproduces the
                 6-17 cm of trunk drift over a 4 s episode.
    `fastfall` — WHAT v4 LEARNED, given every benefit of the doubt: a 2 turns/s
                 whip with the pin held perfectly planted and the paddle cadence
                 of a textbook pivot, the trunk tipping linearly to 90° over
                 0.9 s and `fell_over` firing there. 0.9 s is v4's OWN measured
                 number — at iteration 3999 the log reports mean episode length
                 77 steps of 200 with 47 falls per iteration — not a strawman.
                 The episode ENDS at the fall and collects nothing after it.
    `still`    — the v1 argmax: nothing moves.
    """
    env = _FakeEnv()
    total = 0.0
    turn_rate = microduck_mdp.PV_RATE_CAP         # v5: 0.5 turns/s, full pay
    shuffle_rate = 1.5                            # rad/s, as measured on v2
    whip_rate = 2 * 2 * math.pi                   # 2 turns/s, as measured on v3
    fall_step = int(round(0.9 / 0.02))            # v4's own mean: 77 steps
    for k in range(steps):
        done = bool(env._pv_done[0]) if hasattr(env, "_pv_done") else False
        if strategy == "still":
            env.tick(0.0)
        elif strategy == "pivot":
            if done:
                env.set_contact(_RIGHT, True)
                env.tick(0.0)
            else:
                # the free foot alternates: 0.24 s up, 0.12 s down, pin planted,
                # and it TRAVELS while it is up — 12 steps x 5 mm = 6 cm per
                # stroke, in the middle of the 4-8 cm reach plateau.
                up = (k % 18) < 12
                env.set_contact(_RIGHT, not up)
                env.tick(turn_rate, free_dxy=(0.0, 0.005) if up else (0.0, 0.0))
        elif strategy == "shuffle":
            env.tick(0.0 if done else shuffle_rate,
                     pin_dxy=(0.0, 0.0) if done else (0.002, 0.0))
        elif strategy == "fastfall":
            up = (k % 18) < 12
            env.set_contact(_RIGHT, not up)
            env.set_tilt(90.0 * min(1.0, (k + 1) / fall_step))
            env.tick(whip_rate, free_dxy=(0.0, 0.005) if up else (0.0, 0.0))
            if k + 1 >= fall_step:
                env.termination_manager.fire("fell_over")
        total += _score(env, weights, common) + _tilt_cost(env, weights)
        if strategy == "fastfall" and k + 1 >= fall_step:
            break                                 # a terminated episode is over
    return total


def test_the_arithmetic_pivoting_wins():
    """FOUR strategies over a whole 4 s episode, priced with the live weights on
    the fake env — the table in the cfg docstring, executed.

    v1 ranked STANDING first. v2 ranked the SHUFFLE second, and duly learned it.
    v3 ranked FALLING second and duly learned that. v4 said it had fixed the
    fall — and its own log says otherwise: mean episode length 191 -> 133 -> 74
    over iterations 1000 -> 2000 -> 3000, `fell_over` 1.6 -> 18 -> 58 per
    iteration, `pivot_progress` RISING the whole way (0.59 -> 0.82 -> 0.90). The
    v4 table was not wrong about the mechanism (a fall is paid for in forfeited
    income); it was wrong about the size. The income a fall forfeited was
    upright 2.0 + height 1.0 = 3.0/step, against a progress term paying up to
    6.0/step, and the -20 marker averaged -0.09/step. Falling at ~0.9 s was a
    good trade and the policy took it.

    v5 makes the forfeit the biggest number on the board by construction: the
    `alive` row is +600 for surviving and +21 for the faller. THE DECISIVE LINE
    IS `alive`, and the required margin below is 500 — an order of magnitude
    more than the -100 termination marker, so the ranking cannot be flipped back
    by tuning the penalty.
    """
    cfg = make_microduck_pivot_env_cfg()
    weights = _weights(cfg)
    weights["pin_displacement"] = W_PIN_DISPLACEMENT       # the tightened values
    weights["pin_broken"] = W_PIN_BROKEN
    common = _common_per_step(cfg)

    still = _episode("still", weights, common=common)
    pivot = _episode("pivot", weights, common=common)
    shuffle = _episode("shuffle", weights, common=common)
    fastfall = _episode("fastfall", weights, common=common)

    assert pivot > still > fastfall > shuffle, (pivot, still, fastfall, shuffle)
    assert pivot - still > 1000.0, "turning must beat standing by a landslide"
    assert still - shuffle > 200.0, (
        "the shuffle-spin must score BELOW standing still, not just below a "
        "pivot — v2 ranked it second and duly learned it"
    )
    # THE v5 ASSERTION, and the number the whole version turns on. v4 asserted
    # this same ordering with a 200-point margin and shipped a policy that fell
    # 47 times per iteration; 500 is the margin that says the ranking survives
    # the noise of an actual run rather than merely holding on the page.
    assert still - fastfall > 500.0, (
        "a fast pivot that falls at 0.9 s must lose to STANDING STILL by a "
        "wide margin — v4 priced the fall but not nearly enough, and the log "
        "shows the policy reading that correctly", still, fastfall
    )
    assert pivot - fastfall > 1500.0, (pivot, fastfall)

    # the wrong-way wiggle v1 settled into: it still collects nothing at all.
    env = _FakeEnv()
    wrong = 0.0
    for _ in range(200):
        env.tick(-0.4)                            # ~-23 deg/s, as measured
        wrong += _score(env, weights)
    assert wrong < _episode("still", weights), (wrong,)


def test_falling_loses_on_forfeited_income_not_only_on_the_penalty():
    """Where the fall's price actually comes from — the number that decides it
    is the one v3's arithmetic cancelled away.

    A 0.9 s fall forfeits 155 steps of the survival stack — v5's `alive` term
    plus the common upright/height income. The one-shot termination penalty is
    deliberately NOT sized to carry this on its own: a -900 spike on one
    terminal step is the kind of jackpot AGENTS.md warns about, in reverse.
    -100 is the marker; the forfeit is the money.

    v4 made this exact argument at -20 against a 3.0/step stack and LOST it, and
    that is the whole reason v5 exists: at iteration 3999 the v4 log has
    `terminated` averaging -0.09/step against `pivot_progress` at +0.90/step.
    The mechanism was right; the numbers were an order of magnitude short.
    """
    cfg = make_microduck_pivot_env_cfg()
    weights = _weights(cfg)
    common = _common_per_step(cfg)
    assert cfg.rewards["terminated"].weight == W_TERMINATED == -100.0

    survival_per_step = W_ALIVE + common
    assert survival_per_step == 6.0, "alive 3.0 + upright 2.0 + height 1.0"
    forfeited = survival_per_step * (200 - int(round(0.9 / 0.02)))
    assert forfeited > 5 * abs(W_TERMINATED), (
        "the forfeited income must dominate the one-shot penalty", forfeited
    )
    # ...and it must dominate what the fall BOUGHT, which is what v4's did not:
    # a whip cannot bank more progress than the survival income it gives up.
    assert forfeited > W_PROGRESS * (PV_TARGET_YAW / microduck_mdp.PV_RATE_CAP
                                     / 0.02), forfeited

    # and the penalty still has to be doing work: without it the gap narrows.
    unpriced = dict(weights, terminated=0.0)
    still = _episode("still", weights, common=common)
    with_pen = _episode("fastfall", weights, common=common)
    without = _episode("fastfall", unpriced, common=common)
    assert without - with_pen == abs(W_TERMINATED), (with_pen, without)
    assert with_pen < without < still, (with_pen, without, still)


def test_the_progress_gate_is_what_stops_the_falling_whip():
    """The single most load-bearing v4 change, isolated: a 2 turns/s whip that
    topples earns progress only while the trunk is within 15 deg of vertical.

    Under v3's 40 deg gate the same trajectory banked progress most of the way
    to the floor — the run's `Episode_Reward/pivot_progress` was 0.755, i.e.
    ~0.88 of the per-step cap over a 43-step episode, while every pin cost sat
    near zero. Falling was not a failure of the reward's constraints, it was
    what the reward paid for.
    """
    env = _FakeEnv()
    earned, paid_steps = 0.0, 0
    for k in range(30):                           # tipping to 90 deg over 0.6 s
        env.set_tilt(90.0 * (k + 1) / 30.0)
        env.tick(2 * 2 * math.pi)                 # 2 turns/s
        step = float(microduck_mdp.pv_progress_reward(env)[0])
        earned += step
        paid_steps += int(step > 0.0)
    assert paid_steps <= 6, (
        "past 15 deg of lean the potential must stop advancing", paid_steps
    )
    # a 0.6 s topple therefore banks well under half a turn and never completes
    assert float(env._pv_yaw[0]) < PV_TARGET_YAW / 2.0
    assert bool(env._pv_done[0]) is False
    # ...and nothing banked was BURNED: the same env, brought back upright,
    # picks up exactly where it left off.
    banked = float(env._pv_yaw[0])
    env.set_tilt(0.0)
    env.tick(microduck_mdp.PV_RATE_CAP)
    assert float(env._pv_yaw[0]) > banked


def test_the_shuffle_loses_per_step_during_the_spin_phase():
    """The episode totals are dominated by the settle a pivot reaches and a
    shuffle never does, so check the SPIN-PHASE per-step rate directly: at the
    moment the policy is choosing between the two, the shuffle must already be
    losing to doing nothing.

    v5 changes what "losing to doing nothing" means. In v4 the stall bill made
    standing still NEGATIVE, so the assertion was `shuffle < still < 0`. That is
    no longer the target: the `alive` income is deliberately large enough that a
    robot which does nothing but stay upright is comfortably in profit, because
    the whole v5 thesis is that survival must out-earn any turn that ends the
    episode. What still has to hold is the ORDER — the shuffle loses to doing
    nothing, and doing nothing loses to pivoting by a wide margin — and that the
    shuffle is genuinely under water on its own account, i.e. it is not merely
    winning less than standing but paying for the privilege.
    """
    cfg = make_microduck_pivot_env_cfg()
    weights = _weights(cfg)
    weights["pin_displacement"] = W_PIN_DISPLACEMENT
    weights["pin_broken"] = W_PIN_BROKEN

    rates = {}
    for strategy in ("still", "pivot", "shuffle"):
        env = _FakeEnv()
        total, counted = 0.0, 0
        turn_rate, shuffle_rate = microduck_mdp.PV_RATE_CAP, 1.5
        for k in range(65):                       # 1.3 s, spin phase throughout
            if strategy == "still":
                env.tick(0.0)
            elif strategy == "pivot":
                up = (k % 18) < 12
                env.set_contact(_RIGHT, not up)
                env.tick(turn_rate, free_dxy=(0.0, 0.005) if up else (0.0, 0.0))
            else:
                env.tick(shuffle_rate, pin_dxy=(0.002, 0.0))
            if bool(env._pv_done[0]):
                break                             # only price the SPIN phase
            total += _score(env, weights)
            counted += 1
        rates[strategy] = total / counted

    assert rates["pivot"] > rates["still"] > rates["shuffle"], rates
    assert rates["pivot"] > 6.0, rates
    assert rates["still"] > 0.0, "v5: staying upright is income, by design"
    assert rates["shuffle"] < 0.0, (
        "the shuffle must be under water on its own account, not merely "
        "second — the alive income is paid to it too", rates
    )
    assert rates["still"] - rates["shuffle"] > 3.0, rates
    assert rates["pivot"] - rates["still"] > 5.0, rates


def test_the_loose_curriculum_still_prefers_a_pivot():
    """At iteration 0 the pin costs are at their loose settings (-0.5 / -1.0) —
    a shuffle is cheapest there, so check the ranking holds at BOTH ends of the
    ramp. The progress GATE is not curriculum'd, which is what carries the
    ranking through the loose stage: a policy that could earn progress off the
    pin while the costs were cheap would learn to earn progress off the pin."""
    cfg = make_microduck_pivot_env_cfg()
    weights = _weights(cfg)                                      # loose
    common = _common_per_step(cfg)
    assert weights["pin_displacement"] == W_PIN_DISPLACEMENT_START
    assert weights["pin_broken"] == W_PIN_BROKEN_START
    pivot = _episode("pivot", weights, common=common)
    still = _episode("still", weights, common=common)
    shuffle = _episode("shuffle", weights, common=common)
    fastfall = _episode("fastfall", weights, common=common)
    assert pivot > still > shuffle, (pivot, still, shuffle)
    assert pivot > still > fastfall, (pivot, still, fastfall)
    # The two rows that must NOT move with the curriculum at all: `alive` and
    # the fall penalty are flat from step one, so the 500-point margin that
    # keeps the policy off the v4 trajectory is there at iteration 0 too.
    assert still - fastfall > 500.0, (still, fastfall)
    # (fastfall vs shuffle DOES swap while the pin costs are cheap — that is
    # what "cheap discovery" buys, and it is the shuffle, not the fall, that the
    # loose stage tolerates. The progress gates are never relaxed, so the
    # shuffle still cannot bank a turn.)
    assert shuffle > fastfall, (
        "at iteration 0 the shuffle is merely second-worst; by the tightened "
        "weights it is last", shuffle, fastfall
    )
