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
    FLICKER_S,
    HEAD_STD,
    HEIGHT_STD,
    HOLD_END_S,
    MATCH_GRACE_S,
    MATCH_TOL,
    NOT_DIPPED_END_S,
    NOT_DIPPED_START_S,
    NOT_RISEN_START_S,
    OFF_RAMP_BAND,
    OFF_RAMP_SLOPE,
    PITCH_STD,
    RISE_END_S,
    RISEN_TOL,
    STAND_END_S,
    STAND_POSE_HEIGHT_STD,
    STAND_POSE_STD,
    STAND_Z,
    _LEG_JOINTS,
    MicroduckBowRlCfg,
    make_microduck_bow_env_cfg,
)


_CFG = make_microduck_bow_env_cfg()   # the real weights, for the arithmetic


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
                "stand_pose", "phase_match", "risen", "track_linear_velocity",
                "track_angular_velocity"):
        assert cfg.rewards[pos].weight > 0, pos
    for cost in ("foot_lift", "not_dipped", "not_risen", "drift", "foot_slip",
                 "action_rate_l2", "dof_pos_limits", "self_collisions"):
        assert cfg.rewards[cost].weight < 0, cost
    # Feet planted stays the hardest PER-STEP rule inside the corridor …
    assert cfg.rewards["foot_lift"].weight <= -3.0
    assert cfg.rewards["foot_lift"].weight < cfg.rewards["drift"].weight
    # … but v3 deliberately makes being off the ramp cost MORE than it: a
    # frozen pose has to lose, and v2 proved that a cheap "failure to move"
    # cost simply gets ignored (its not_risen logged -0.0003).
    assert cfg.rewards["not_dipped"].weight == cfg.rewards["not_risen"].weight
    assert cfg.rewards["not_risen"].weight <= -5.0
    assert cfg.rewards["not_risen"].weight < cfg.rewards["foot_lift"].weight
    # foot_slip must not stay command-gated at a pinned ~zero command.
    assert cfg.rewards["foot_slip"].params["command_threshold"] < 0.0
    # Smoothness is 10x lighter than v2: it taxed a robot that stood still
    # 0.71/step, which is more than the whole dip is worth.
    assert -0.05 <= cfg.rewards["action_rate_l2"].weight < 0.0
    # Survival income is halved so it cannot fund parking.
    assert cfg.rewards["bow_upright"].weight == 1.0
    assert cfg.rewards["track_linear_velocity"].weight == 0.25
    # Every std is smaller than the motion it measures.
    assert cfg.rewards["bow_height"].params["std"] == HEIGHT_STD == 0.006
    assert HEIGHT_STD < (STAND_Z - BOW_Z) / 2
    assert cfg.rewards["bow_pitch"].params["std"] == PITCH_STD < BOW_PITCH_RAD / 2
    assert cfg.rewards["bow_head"].params["std"] == HEAD_STD < 0.25


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
    assert h() > 0.98, "standing at t=0 must score: the ramp starts at STAND_Z"
    # Diving to the crouch immediately pays a fraction of staying on the
    # ramp — being ahead of the slew is not worth it (no jackpot).
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
    assert h() < 0.01
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


def test_foot_lift_cost_fires_on_sustained_contact_loss():
    env = _FakeEnv()
    lift = lambda: float(microduck_mdp.bow_foot_lift_penalty(env)[0])
    env.tick(contacts=(0, 0))        # spawn drop, not yet landed: free
    assert lift() == 0.0
    env.tick(contacts=(1, 1))
    assert lift() == 0.0
    env.tick(contacts=(0, 1))        # left up for 0.02 s — still a flicker
    assert lift() == 0.0
    env.tick(contacts=(0, 0))        # left has now been up 0.04 s; right 0.02 s
    assert lift() == 0.5
    env.tick(contacts=(0, 0))        # both past the flicker window
    assert lift() == 1.0
    env.tick(contacts=(1, 1))
    assert lift() == 0.0


def test_flicker_exemption_is_a_window_not_a_discount():
    """A one-frame heel unweight during a rise is free; a hop is not.

    This is (d) of the v2 fix: at weight -3.0 the old term charged the contact
    flicker that every push-up out of a crouch produces, which is part of why
    v1 decided the rise was not worth attempting."""
    assert FLICKER_S == 0.04
    # A single-step flicker, repeated, never costs anything.
    env = _FakeEnv()
    env.tick(contacts=(1, 1))
    total = 0.0
    for _ in range(10):
        env.tick(contacts=(0, 1))    # one step in the air ...
        total += float(microduck_mdp.bow_foot_lift_penalty(env)[0])
        env.tick(contacts=(1, 1))    # ... then back down
        total += float(microduck_mdp.bow_foot_lift_penalty(env)[0])
    assert total == 0.0, "a 20 ms contact flicker must not read as a foot lift"
    # A real hop is still charged on all but its first frames.
    env = _FakeEnv()
    env.tick(contacts=(1, 1))
    charged = 0
    for _ in range(10):
        env.tick(contacts=(0, 0))
        charged += float(microduck_mdp.bow_foot_lift_penalty(env)[0]) == 1.0
    assert charged == 9, "only the first flicker step of a hop is exempt"


def test_risen_bonus_is_one_shot_and_needs_an_actual_bow():
    env = _FakeEnv()
    fired = lambda: float(microduck_mdp.bow_risen_bonus(env)[0])
    # Standing still the whole episode never earns it: no crouch was reached.
    total = 0.0
    for _ in range(int(STAND_END_S / env.step_dt)):
        env.tick()
        total += fired()
    assert total == 0.0, "the one-shot must not pay a robot that never bowed"

    # Bow, hold, then rise: exactly one payment, and not before the hold ends.
    env = _FakeEnv()
    env.tick()
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
    env.run_to(HOLD_END_S)                      # down in the crouch
    assert bool(env._bow_dipped[0]) is True
    assert fired() == 0.0
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, STAND_Z]])
    env.tick()                                  # first step back up
    assert fired() == 1.0
    paid = 0.0
    for _ in range(50):                         # ... and never again
        env.tick()
        paid += fired()
    assert paid == 0.0, "a per-step 'you are up' reward would be a jackpot"
    assert bool(env._bow_risen[0]) is True


def test_risen_bonus_needs_both_feet_planted():
    env = _FakeEnv()
    env.tick()
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
    env.run_to(HOLD_END_S)
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, STAND_Z]])
    env.tick(contacts=(0, 1))                   # up, but hopping off one foot
    assert float(microduck_mdp.bow_risen_bonus(env)[0]) == 0.0
    env.tick(contacts=(1, 1))
    assert float(microduck_mdp.bow_risen_bonus(env)[0]) == 1.0


def test_risen_tolerance_is_one_centimetre():
    env = _FakeEnv()
    env.tick()
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
    env.run_to(HOLD_END_S)
    env._asset.data.root_link_pos_w = torch.tensor(
        [[0.0, 0.0, STAND_Z - RISEN_TOL - 0.002]])
    env.tick()
    assert float(microduck_mdp.bow_risen_bonus(env)[0]) == 0.0, "1.2 cm short"
    env._asset.data.root_link_pos_w = torch.tensor(
        [[0.0, 0.0, STAND_Z - RISEN_TOL + 0.002]])
    env.tick()
    assert float(microduck_mdp.bow_risen_bonus(env)[0]) == 1.0


# ── the ramp, and the three strategies measured against it ──────────────────

_CROUCH_LEGS = {2: 0.25, 3: 0.5, 4: 0.25, 11: 0.25, 12: 0.5, 13: 0.25}


def _legs(depth_frac):
    """Leg-joint offsets from HOME for a crouch `depth_frac` of the way down."""
    return {k: v * depth_frac for k, v in _CROUCH_LEGS.items()}


def _ramp(t):
    """(z, s, r) of the slewed ramp at time t — the trajectory being asked for."""
    s = min(t / DIP_END_S, 1.0) * (1.0 - min(max(t - HOLD_END_S, 0.0), 1.0))
    r = min(max(t - HOLD_END_S, 0.0), 1.0)
    return STAND_Z + s * (BOW_Z - STAND_Z), s, r


def _textbook(t):
    z, s, r = _ramp(t)
    head = (microduck_mdp.BOW_HEAD_DOWN[0] * s + microduck_mdp.BOW_HEAD_UP[0] * r,
            microduck_mdp.BOW_HEAD_DOWN[1] * s + microduck_mdp.BOW_HEAD_UP[1] * r)
    return z, BOW_PITCH_RAD * s, head, _legs((STAND_Z - z) / (STAND_Z - BOW_Z))


def _park_high(t):
    """What bow-v2 actually did: settle at 0.101 m by 0.5 s, then freeze —
    4° nose-down, head never moved, legs half-bent."""
    u = min(t / 0.5, 1.0)
    z = STAND_Z + u * (0.101 - STAND_Z)
    return (z, math.radians(4.0) * u, (0.0, 0.0),
            _legs((STAND_Z - z) / (STAND_Z - BOW_Z)))


def _park_low(t):
    """What bow-v1 did, in its strongest form: a textbook dip and hold, then
    frozen in the crouch through the rise and the stand."""
    if t <= HOLD_END_S:
        return _textbook(t)
    return (BOW_Z, BOW_PITCH_RAD, microduck_mdp.BOW_HEAD_DOWN, _legs(1.0))


def _drive(env, traj):
    """Place the robot where `traj` says it should be at the NEXT step, then
    step. Placing before the tick keeps one `_bow_update` per step, which is
    what the one-shot terms (phase credits, `risen`) are counted on."""
    t = float(env.episode_length_buf[0] + 1) * env.step_dt
    z, nose_down, head, legs = traj(t)
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, z]])
    env._asset.data.root_link_quat_w = torch.tensor(
        [[math.cos(nose_down / 2), 0.0, math.sin(nose_down / 2), 0.0]])
    q = torch.zeros(1, 14)
    q[0, 5], q[0, 6] = head
    for j, v in legs.items():
        q[0, j] = v
    env._asset.data.joint_pos = q
    env.tick()


def _frozen(z, nose_down=0.0, head=(0.0, 0.0), depth_frac=None):
    """A strategy that never moves — the shape of every failure so far."""
    legs = _legs((STAND_Z - z) / (STAND_Z - BOW_Z) if depth_frac is None
                 else depth_frac)
    return lambda t: (z, nose_down, head, legs)


def _cfg_params(name):
    return {k: v for k, v in _CFG.rewards[name].params.items()
            if k != "sensor_name"}


def _off_ramp(env, high):
    """One of the two off-ramp costs, with the cfg's own parameters."""
    fn = (microduck_mdp.bow_not_dipped_penalty if high
          else microduck_mdp.bow_not_risen_penalty)
    return float(fn(env, **_cfg_params("not_dipped" if high else "not_risen"))[0])


def test_the_compromise_height_scores_nothing_anywhere():
    """(a) of the v3 fix, and the whole reason v2 froze at 0.101 m.

    Under v2's 2 cm std that height scored 0.53 against the crouch target and
    0.61 against the stand target — most of a 4.0-weighted term, collected by a
    robot that never moved. Every phase must now price it at ~zero, and both
    off-ramp costs must charge it."""
    env = _FakeEnv()
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, 0.101]])
    env.run_to(HOLD_END_S - 0.02)          # the last step of the hold
    at_hold = float(microduck_mdp.bow_height_reward(env, std=HEIGHT_STD)[0])
    assert _off_ramp(env, high=True) == 1.0, "parked high must lose the hold"
    env.run_to(STAND_END_S)
    at_stand = float(microduck_mdp.bow_height_reward(env, std=HEIGHT_STD)[0])
    assert _off_ramp(env, high=False) > 0.5, "parked low must lose the stand"
    assert at_hold < 0.01 and at_stand < 0.03, (at_hold, at_stand)


def test_off_ramp_costs_never_charge_a_trunk_on_the_ramp():
    """Both costs measure against the SLEWED ramp, not a fixed height — which
    is what let v2's `not_risen` log -0.0003 while the policy sat at 0.101, and
    what would otherwise punish a policy correctly half-way through its rise."""
    env = _FakeEnv()
    for _ in range(int(STAND_END_S / env.step_dt)):
        _drive(env, _textbook)
        assert _off_ramp(env, high=True) == 0.0, float(env._bow_t[0])
        assert _off_ramp(env, high=False) == 0.0, float(env._bow_t[0])


def test_off_ramp_costs_are_slopes_in_the_right_windows():
    def at(z):
        env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, z]])
        env.common_step_counter += 1          # same t, new height
        microduck_mdp._bow_update(
            env, "feet_ground_contact", microduck_mdp._DEFAULT_ASSET_CFG)

    # not_dipped: silent before 0.8 s even for a trunk that never moved …
    env = _FakeEnv()
    env.run_to(NOT_DIPPED_START_S - 0.1)
    assert _off_ramp(env, high=True) == 0.0, "must not tax the first 0.8 s"
    env.run_to(HOLD_END_S - 0.02)
    assert _off_ramp(env, high=True) == 1.0, "1.5 cm above the ramp = full cost"
    # … a slope, not a cliff: every millimetre down pays strictly less …
    costs = []
    for z in (STAND_Z, 0.101, 0.098, BOW_Z + OFF_RAMP_BAND, BOW_Z):
        at(z)
        costs.append(_off_ramp(env, high=True))
    assert costs == sorted(costs, reverse=True), costs
    assert costs[-1] == 0.0 and costs[-2] < 1e-4, "1 cm of the ramp is free"
    assert 0.0 < costs[2] < 1.0, "then a 5 mm slope to the full cost"
    # … and it switches off again for the rise.
    env.run_to(NOT_DIPPED_END_S)
    at(STAND_Z)
    assert _off_ramp(env, high=True) == 0.0, "the rise is not the dip's window"

    # not_risen: silent before 2.5 s, then charged while the trunk is BELOW.
    env = _FakeEnv()
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
    env.run_to(NOT_RISEN_START_S - 0.1)
    assert _off_ramp(env, high=False) == 0.0, "must not tax the bow itself"
    env.run_to(STAND_END_S)
    assert _off_ramp(env, high=False) == 1.0
    costs = []
    for z in (BOW_Z, 0.098, 0.101, STAND_Z - OFF_RAMP_BAND, STAND_Z):
        at(z)
        costs.append(_off_ramp(env, high=False))
    assert costs == sorted(costs, reverse=True), costs
    assert costs[0] == 1.0 and costs[-1] == 0.0 and costs[-2] < 1e-4
    assert math.isclose(costs[2], 0.8, abs_tol=1e-5), "0.101 pays 0.8, not 0"
    assert math.isclose(OFF_RAMP_SLOPE, 0.005)


def test_phase_match_needs_the_whole_phase_on_the_ramp():
    """(c): four credits per episode, and none of them buyable with a pose."""
    env = _FakeEnv()
    credits = 0.0
    for _ in range(int(STAND_END_S / env.step_dt)):
        _drive(env, _textbook)
        credits += float(microduck_mdp.bow_phase_match_reward(env)[0])
    assert credits == 4.0, "tracking the ramp earns one credit per phase"

    def frozen_credits(z):
        env = _FakeEnv()
        return sum(
            (_drive(env, _frozen(z)),
             float(microduck_mdp.bow_phase_match_reward(env)[0]))[1]
            for _ in range(int(STAND_END_S / env.step_dt)))

    # No static height earns anything: not the compromise the v2 policy froze
    # at, not the spawn height, and not standing still at STAND_Z — the last
    # one is on the ramp for the whole stand phase, which is why the credit is
    # gated on having actually reached the crouch.
    for z in (STAND_Z, 0.101, 0.127):
        assert frozen_credits(z) == 0.0, f"a trunk frozen at {z} earns nothing"
    # A trunk frozen in the crouch earns the ONE phase it really does hold —
    # the hold — and none of the other three. That is v1's whole score.
    assert frozen_credits(BOW_Z) == 1.0


def test_phase_match_forgives_the_spawn_drop_only():
    """The spawn lands 0.5–1.5 cm above the ramp (`reset_base` z = 0.12–0.13):
    that drop is free, tracking errors after MATCH_GRACE_S are not."""
    assert MATCH_GRACE_S == 0.3 and MATCH_TOL == 0.01
    env = _FakeEnv()
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, 0.127]])
    env.run_to(MATCH_GRACE_S - 0.02)
    assert bool(env._bow_phase_ok[0]) is True, "the spawn drop must be free"
    _drive(env, _textbook)
    assert bool(env._bow_phase_ok[0]) is True
    _drive(env, lambda t: (_ramp(t)[0] + MATCH_TOL + 0.002, 0.0, (0.0, 0.0), {}))
    assert bool(env._bow_phase_ok[0]) is False, "1.2 cm off the ramp clears it"
    # One bad step costs the whole phase …
    while float(env.episode_length_buf[0]) * env.step_dt < DIP_END_S - 1e-9:
        _drive(env, _textbook)
    assert float(microduck_mdp.bow_phase_match_reward(env)[0]) == 0.0
    # … and the next phase re-arms clean.
    _drive(env, _textbook)
    assert int(env._bow_phase[0]) == microduck_mdp.BOW_PHASE_HOLD
    assert bool(env._bow_phase_ok[0]) is True, "each phase starts clean"


def _episode(traj):
    """Every bow reward term summed over the 200 steps of one episode, weighted
    as the cfg weights them (before mjlab's x dt). Returns (terms, phases)."""
    env = _FakeEnv()
    w = lambda n: _CFG.rewards[n].weight
    terms, phases = {}, {}
    for _ in range(int(STAND_END_S / env.step_dt)):
        _drive(env, traj)
        step = {
            "bow_height": w("bow_height") * float(
                microduck_mdp.bow_height_reward(env, **_cfg_params("bow_height"))[0]),
            "bow_pitch": w("bow_pitch") * float(
                microduck_mdp.bow_pitch_reward(env, **_cfg_params("bow_pitch"))[0]),
            "bow_head": w("bow_head") * float(
                microduck_mdp.bow_head_reward(env, **_cfg_params("bow_head"))[0]),
            "bow_upright": w("bow_upright") * float(
                microduck_mdp.bow_upright_reward(env, **_cfg_params("bow_upright"))[0]),
            "stand_pose": w("stand_pose") * float(
                microduck_mdp.bow_stand_pose_reward(env, **_cfg_params("stand_pose"))[0]),
            "phase_match": w("phase_match") * float(
                microduck_mdp.bow_phase_match_reward(env)[0]),
            "risen": w("risen") * float(microduck_mdp.bow_risen_bonus(env)[0]),
            "not_dipped": w("not_dipped") * float(
                microduck_mdp.bow_not_dipped_penalty(env, **_cfg_params("not_dipped"))[0]),
            "not_risen": w("not_risen") * float(
                microduck_mdp.bow_not_risen_penalty(env, **_cfg_params("not_risen"))[0]),
            # Standing still and level pays every strategy the same: both track
            # terms at ~1.0. Counted for all three so the totals are honest.
            "track": 2 * _CFG.rewards["track_linear_velocity"].weight,
        }
        for k, v in step.items():
            terms[k] = terms.get(k, 0.0) + v
        ph = int(env._bow_phase[0])
        phases[ph] = phases.get(ph, 0.0) + sum(step.values())
    return terms, phases


def test_the_three_strategies():
    """The v3 arithmetic on the real reward functions (module docstring).

    A textbook bow must beat BOTH parked poses by a wide margin, and each park
    must be NEGATIVE in the phases its parking violates — which is exactly what
    v2's stack failed at: it paid the frozen 0.101 m pose 7.15 per step."""
    book, book_ph = _episode(_textbook)
    high, high_ph = _episode(_park_high)
    low, low_ph = _episode(_park_low)
    book_t, high_t, low_t = (sum(x.values()) for x in (book, high, low))
    steps = int(STAND_END_S / 0.02)

    # 1. The textbook bow wins by a wide margin against both.
    assert book_t > 2000, book_t
    assert book_t > 8 * high_t, (book_t, high_t)
    assert book_t > 2.5 * low_t, (book_t, low_t)
    # It is the only one that collects the completion terms at all …
    assert book["phase_match"] == 4 * _CFG.rewards["phase_match"].weight
    assert book["risen"] == _CFG.rewards["risen"].weight
    assert high["phase_match"] == 0.0 and high["risen"] == 0.0
    assert low["risen"] == 0.0
    # … and a trunk on the ramp is never charged by either off-ramp cost.
    assert book["not_dipped"] == 0.0 and book["not_risen"] == 0.0

    # 2. Parking high (the v2 optimum) is charged from both sides and BLEEDS in
    #    the hold and in the stand: the pose that used to be the argmax now
    #    loses money for every step it is held.
    assert high["not_dipped"] < -250 and high["not_risen"] < -200
    assert high_ph[microduck_mdp.BOW_PHASE_HOLD] < 0.0
    assert high_ph[microduck_mdp.BOW_PHASE_STAND] < 0.0
    assert high_t / steps < 1.5, "v2 paid this same pose 7.15 per step"

    # 3. Parking low (the v1 optimum) keeps what its dip and hold honestly earn
    #    — those two phases are worth exactly what the textbook's are — and
    #    loses the whole second half.
    assert math.isclose(low_ph[microduck_mdp.BOW_PHASE_DIP],
                        book_ph[microduck_mdp.BOW_PHASE_DIP], rel_tol=1e-6)
    assert low_ph[microduck_mdp.BOW_PHASE_STAND] < 0.0
    assert (low_ph[microduck_mdp.BOW_PHASE_RISE]
            + low_ph[microduck_mdp.BOW_PHASE_STAND]) < 0.0

    # 4. Neither park comes close to the standing finish's pay.
    for parked in (high_ph, low_ph):
        assert parked[microduck_mdp.BOW_PHASE_STAND] < 0.2 * book_ph[
            microduck_mdp.BOW_PHASE_STAND]


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
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
    env.run_to(HOLD_END_S)
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, STAND_Z]])
    env.run_to(RISE_END_S)
    assert int(env._bow_phase[0]) == microduck_mdp.BOW_PHASE_STAND
    assert bool(env._bow_landed[0]) is True
    assert bool(env._bow_dipped[0]) is True and bool(env._bow_risen[0]) is True
    env._asset.data.root_link_pos_w = torch.tensor([[0.3, -0.2, STAND_Z]])
    env.episode_length_buf[:] = 0            # reset
    env.tick(contacts=(0, 0))
    assert int(env._bow_phase[0]) == microduck_mdp.BOW_PHASE_DIP
    assert float(env._bow_blend[0]) < 0.05 and float(env._bow_rise[0]) == 0.0
    assert bool(env._bow_landed[0]) is False, "touchdown latch must re-arm"
    assert bool(env._bow_dipped[0]) is False, "the dip latch must re-arm"
    assert bool(env._bow_risen[0]) is False, "the rise latch must re-arm"
    assert float(microduck_mdp.bow_risen_bonus(env)[0]) == 0.0
    assert float(env._bow_air_t[0].max()) <= env.step_dt, "air clock must re-arm"
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
