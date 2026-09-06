"""Bow cfg invariants (CPU, no GPU): registration, the command is pinned, the
episode and its phase boundaries, every cost carries a negative weight, the
measured head/pitch SIGNS, and the per-env phase memory on a fake env."""

import math
import types

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_bow_env_cfg import (
    BOW_COLLAPSE_Z,
    BOW_DIP_M,
    BOW_PITCH_RAD,
    BOW_Z,
    DIP_END_S,
    EPISODE_LENGTH_S,
    FLICKER_S,
    HEAD_STD,
    HEIGHT_STD,
    HOLD_END_S,
    PITCH_STD,
    RISE_CREDITS,
    RISE_END_S,
    RISE_PAY_END_S,
    RISE_PAY_START_S,
    RISE_RATE_CAP,
    RISEN_TOL,
    STAND_END_S,
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
                "stand_pose", "rise_progress", "risen",
                "track_linear_velocity", "track_angular_velocity"):
        assert cfg.rewards[pos].weight > 0, pos
    for cost in ("foot_lift", "terminated", "drift", "foot_slip",
                 "action_rate_l2", "dof_pos_limits", "self_collisions"):
        assert cfg.rewards[cost].weight < 0, cost
    # v3's corridor credit and its two off-ramp costs are GONE: costs that make
    # a parked pose expensive also make TERMINATING cheap, and the v3 run took
    # that exit (collapsed 31.8/iteration against 4.6 time-outs).
    for gone in ("phase_match", "not_dipped", "not_risen"):
        assert gone not in cfg.rewards, gone
    # Feet planted is still the hardest per-step rule …
    assert cfg.rewards["foot_lift"].weight <= -3.0
    assert cfg.rewards["foot_lift"].weight < cfg.rewards["drift"].weight
    # … but ending the episode in a heap is far worse than any per-step cost.
    assert cfg.rewards["terminated"].weight <= -20.0
    assert cfg.rewards["terminated"].weight < 5 * cfg.rewards["foot_lift"].weight
    assert cfg.rewards["terminated"].params["term_names"] == (
        "collapsed", "fell_over")
    # foot_slip must not stay command-gated at a pinned ~zero command.
    assert cfg.rewards["foot_slip"].params["command_threshold"] < 0.0
    # Smoothness is fixed and light: the trick IS motion.
    assert -0.05 <= cfg.rewards["action_rate_l2"].weight < 0.0
    # v1's survival income is restored — v3 halved it and the policy responded
    # by ending episodes, not by working harder.
    assert cfg.rewards["bow_upright"].weight == 2.0
    assert cfg.rewards["track_linear_velocity"].weight == 0.5
    # v1's tolerances are restored too: wide enough to have a gradient for the
    # policy that exists, not the one we wish existed.
    assert cfg.rewards["bow_height"].params["std"] == HEIGHT_STD == 0.02
    assert cfg.rewards["bow_pitch"].params["std"] == PITCH_STD == 0.12
    assert cfg.rewards["bow_head"].params["std"] == HEAD_STD == 0.30
    assert cfg.rewards["stand_pose"].params["std"] == STAND_POSE_STD
    # The stand-phase pose is ADDITIVE (v1's form): no height factor to
    # multiply it to zero a centimetre short of standing.
    assert "stand_z" not in cfg.rewards["stand_pose"].params
    assert "height_std" not in cfg.rewards["stand_pose"].params


def test_rise_progress_is_wired_and_normalised():
    """A full rise is worth a fixed total, whatever speed it happens at."""
    cfg = make_microduck_bow_env_cfg()
    term = cfg.rewards["rise_progress"]
    assert term.func is microduck_mdp.bow_rise_progress_reward
    assert term.params["rate_cap"] == RISE_RATE_CAP == 0.06
    # The cap must not be BELOW the rate the ramp itself asks for, or tracking
    # the ramp would throw credits away.
    assert RISE_RATE_CAP >= BOW_DIP_M / (RISE_END_S - HOLD_END_S)
    assert RISE_CREDITS == 25.0
    assert math.isclose(RISE_CREDITS, BOW_DIP_M / (RISE_RATE_CAP * 0.02))
    # RISE_CREDITS is quoted per 20 ms control step; if the control rate ever
    # moves, the "150 points for a full rise" arithmetic moves with it.
    assert math.isclose(cfg.decimation * cfg.sim.mujoco.timestep, 0.02)
    # The window opens with the rise and closes a little after it, so a policy
    # that is still finishing at 3.0 s is paid for finishing.
    assert RISE_PAY_START_S == HOLD_END_S == 2.0
    assert RISE_END_S < RISE_PAY_END_S < STAND_END_S


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


class _FakeTerminations:
    """Just enough of mjlab's TerminationManager for the termination penalty."""

    def __init__(self, n):
        self.active_terms = ["fell_over", "out_of_terrain_bounds", "nan_state",
                             "collapsed"]
        self._dones = {k: torch.zeros(n, dtype=torch.bool)
                       for k in self.active_terms}

    def get_term(self, name):
        return self._dones[name]

    def fire(self, name):
        self._dones[name][:] = True


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
        self.termination_manager = _FakeTerminations(n)
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
    assert h() < 0.15, "being 3 cm ahead of the slew pays a small fraction"
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


def test_rise_progress_is_potential_based():
    """Holding pays nothing; the same climb cannot be sold twice.

    This is the term v1-v3 did not have, and the reason all three parked: every
    other term in the stack prices WHERE THE TRUNK IS, so a policy one
    centimetre into an unfinished rise collected nothing at all for that
    centimetre."""
    prog = lambda e: float(microduck_mdp.bow_rise_progress_reward(e)[0])

    # 1. A robot that never dips can never earn it: the potential arms at
    #    whatever height it brings into the window, and standing arms it full.
    env = _FakeEnv()
    total = sum((env.tick(), prog(env))[1]
                for _ in range(int(STAND_END_S / env.step_dt)))
    assert total == 0.0, "standing tall for four seconds must pay nothing"

    # 2. Sitting in the crouch for the whole rise pays nothing either — Delta is
    #    zero for a potential that does not move.
    env = _FakeEnv()
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
    total = sum((env.tick(), prog(env))[1]
                for _ in range(int(STAND_END_S / env.step_dt)))
    assert total == 0.0, "parking in the crouch must pay nothing"

    # 3. A bow that rises collects the credits, and only inside the window.
    env = _FakeEnv()
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
    env.run_to(HOLD_END_S)
    assert prog(env) == 0.0, "nothing is owed for the crouch itself"
    earned, z = 0.0, BOW_Z
    while z < STAND_Z - 1e-9:                    # 1 mm per step = 0.05 m/s
        z = min(z + 0.001, STAND_Z)
        env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, z]])
        env.tick()
        earned += prog(env)
    assert math.isclose(earned, RISE_CREDITS, rel_tol=1e-5), earned

    # 4. Climbing past the standing height is free, and sinking back down and
    #    re-climbing is not paid a second time.
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, STAND_Z + 0.02]])
    env.tick()
    assert prog(env) == 0.0, "overshoot past standing pays nothing"
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
    env.tick()
    again = 0.0
    for k in range(1, 31):
        env._asset.data.root_link_pos_w = torch.tensor(
            [[0.0, 0.0, BOW_Z + 0.001 * k]])
        env.tick()
        again += prog(env)
    assert again == 0.0, "a running maximum cannot sell the same climb twice"


def test_rise_progress_total_is_invariant_to_speed():
    """Rate-capped Delta of a potential: the whole rise is worth the same total
    at any speed up to the cap, and going faster than the cap throws credits
    away. That is AGENTS.md's 'no jackpots' rule applied to a rise."""
    def climb(per_step):
        env = _FakeEnv()
        env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
        env.run_to(HOLD_END_S)
        earned, z = 0.0, BOW_Z
        while (float(env.episode_length_buf[0]) * env.step_dt
               < RISE_PAY_END_S - 1e-9):
            z = min(z + per_step, STAND_Z)
            env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, z]])
            env.tick()
            earned += float(microduck_mdp.bow_rise_progress_reward(env)[0])
        return earned

    cap_per_step = RISE_RATE_CAP * 0.02              # 1.2 mm/step
    slow = climb(0.0005)     # 0.025 m/s — uses the whole 1.2 s window
    mid = climb(0.001)       # 0.050 m/s — done in 0.6 s
    at_cap = climb(cap_per_step)
    for got in (slow, mid, at_cap):
        assert math.isclose(got, RISE_CREDITS, rel_tol=1e-5), got
    # Twice the cap: half the credits are clipped away (12 capped steps plus a
    # 1.2 mm remainder, so 13.0 rather than exactly 12.5).
    assert climb(2 * cap_per_step) == 13.0


def test_rise_progress_forfeits_what_a_hop_flies_through():
    """The potential updates while airborne but the PAYMENT is gated on both
    feet planted, so a hop through standing height banks nothing and can never
    re-earn it. Without that, ~4 steps of flight (about 12 points of foot_lift)
    would buy the full 150-point rise."""
    env = _FakeEnv()
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
    env.run_to(HOLD_END_S)
    earned = 0.0
    for k in range(1, 31):                       # ballistic through the rise …
        env._asset.data.root_link_pos_w = torch.tensor(
            [[0.0, 0.0, BOW_Z + 0.001 * k]])
        env.tick(contacts=(0, 0))
        earned += float(microduck_mdp.bow_rise_progress_reward(env)[0])
    assert earned == 0.0, "no progress pay while airborne"
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
    env.tick(contacts=(1, 1))                    # … and land back in the crouch
    for k in range(1, 31):
        env._asset.data.root_link_pos_w = torch.tensor(
            [[0.0, 0.0, BOW_Z + 0.001 * k]])
        env.tick(contacts=(1, 1))
        earned += float(microduck_mdp.bow_rise_progress_reward(env)[0])
    assert earned == 0.0, "height flown through is forfeited, not banked"


def test_rise_progress_window_closes():
    env = _FakeEnv()
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
    env.run_to(RISE_PAY_END_S)
    late = 0.0
    for k in range(1, 31):
        env._asset.data.root_link_pos_w = torch.tensor(
            [[0.0, 0.0, BOW_Z + 0.001 * k]])
        env.tick()
        late += float(microduck_mdp.bow_rise_progress_reward(env)[0])
    assert late == 0.0, "rising after 3.2 s is not what the trick asks for"


def test_termination_penalty_charges_only_the_behavioural_terminations():
    env = _FakeEnv()
    pen = lambda: float(microduck_mdp.bow_termination_penalty(env)[0])
    env.tick()
    assert pen() == 0.0
    env.termination_manager.fire("nan_state")
    assert pen() == 0.0, "a sim blow-up is not something the policy chose"
    env.termination_manager.fire("collapsed")
    assert pen() == 1.0
    env2 = _FakeEnv()
    env2.termination_manager.fire("fell_over")
    assert float(microduck_mdp.bow_termination_penalty(env2)[0]) == 1.0
    # A term the env does not have must not raise (the base cfg owns the names).
    assert float(microduck_mdp.bow_termination_penalty(
        env2, term_names=("nope",))[0]) == 0.0


def _collapse(t):
    """A textbook bow that gives up and folds at 1.5 s. Given every point it
    could possibly earn on the way down, so the comparison is honest."""
    if t < 1.5 - 1e-9:
        return _textbook(t)
    return BOW_COLLAPSE_Z - 0.001, BOW_PITCH_RAD, microduck_mdp.BOW_HEAD_DOWN, _legs(1.0)


def _episode(traj, terminate_s=None):
    """Every bow reward term summed over one episode, weighted as the cfg
    weights them (before mjlab's x dt). An episode that TERMINATES stops
    collecting at `terminate_s` — which is the whole point of the comparison:
    v3's stack made that silence cheaper than the pose it was pricing.

    Returns (terms, phases)."""
    env = _FakeEnv()
    w = lambda n: _CFG.rewards[n].weight
    terms, phases = {}, {}
    for _ in range(int(STAND_END_S / env.step_dt)):
        _drive(env, traj)
        t = float(env._bow_t[0])
        dead = terminate_s is not None and t >= terminate_s - 1e-9
        if dead:
            env.termination_manager.fire("collapsed")
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
            "rise_progress": w("rise_progress") * float(
                microduck_mdp.bow_rise_progress_reward(
                    env, **_cfg_params("rise_progress"))[0]),
            "risen": w("risen") * float(microduck_mdp.bow_risen_bonus(env)[0]),
            "terminated": w("terminated") * float(
                microduck_mdp.bow_termination_penalty(env, **_cfg_params("terminated"))[0]),
            # Standing still and level pays every strategy the same: both track
            # terms at ~1.0. Counted for all of them so the totals are honest.
            "track": 2 * _CFG.rewards["track_linear_velocity"].weight,
        }
        for k, v in step.items():
            terms[k] = terms.get(k, 0.0) + v
        ph = int(env._bow_phase[0])
        phases[ph] = phases.get(ph, 0.0) + sum(step.values())
        if dead:
            break                     # a terminated episode collects nothing more
    return terms, phases


def test_the_four_strategies():
    """The v4 arithmetic on the real reward functions (module docstring).

    Three things have to be true at once, and v1-v3 each got only two:
      1. the textbook bow is the argmax;
      2. parking in the crouch — the v1 policy — loses clearly;
      3. COLLAPSING is the worst outcome of the four, so the escape hatch v3
         taught the policy to take is closed."""
    book, book_ph = _episode(_textbook)
    high, _ = _episode(_park_high)
    low, low_ph = _episode(_park_low)
    dead, _ = _episode(_collapse, terminate_s=1.5)
    book_t, high_t, low_t, dead_t = (
        sum(x.values()) for x in (book, high, low, dead))
    steps = int(STAND_END_S / 0.02)

    # 1. The textbook bow wins, and it is the only strategy that collects the
    #    rise at all.
    assert book_t > max(high_t, low_t, dead_t)
    assert math.isclose(book["rise_progress"],
                        RISE_CREDITS * _CFG.rewards["rise_progress"].weight,
                        rel_tol=1e-5), book["rise_progress"]
    assert book["risen"] == _CFG.rewards["risen"].weight
    assert book["terminated"] == 0.0
    for other in (high, low, dead):
        assert other["rise_progress"] == 0.0
        assert other["risen"] == 0.0

    # 2. Parking in the crouch (v1's behaviour) keeps what its dip and hold
    #    honestly earn — those two phases are worth exactly the textbook's —
    #    and loses the entire second half.
    assert math.isclose(low_ph[microduck_mdp.BOW_PHASE_DIP],
                        book_ph[microduck_mdp.BOW_PHASE_DIP], rel_tol=1e-6)
    assert math.isclose(low_ph[microduck_mdp.BOW_PHASE_HOLD],
                        book_ph[microduck_mdp.BOW_PHASE_HOLD], rel_tol=1e-6)
    assert low_t < 0.7 * book_t, (low_t, book_t)
    # Every park stays POSITIVE overall: a half-done trick is worth less than a
    # whole one, but it must never be worth less than quitting (that inversion
    # is precisely what v3 built, and the policy quit).
    assert low_t > 0.0 and high_t > 0.0

    # 3. Collapsing is the WORST of the four, by a wide margin. It does not go
    #    negative — an episode that ends simply stops earning, and no plausible
    #    penalty makes 1.5 s of an honest bow worth less than nothing — but it
    #    is barely half of the next-worst strategy, which is the whole point.
    assert dead_t == min(book_t, high_t, low_t, dead_t), dead_t
    assert dead_t < 0.5 * min(book_t, high_t, low_t), (dead_t, low_t)
    assert dead["terminated"] == _CFG.rewards["terminated"].weight
    # The -20 is insurance, not the mechanism: what makes quitting lose is that
    # it forfeits 125 steps of income. Remove the penalty and it must STILL be
    # the worst — v3's failure was that it was not.
    assert dead_t - dead["terminated"] < min(book_t, high_t, low_t)

    # 4. The margins, per step, in wandb's Episode_Reward units. The textbook
    #    bow leads the low park by 3.6/step and the high park by 3.4/step —
    #    thinner than v3's paper margins, and deliberately so: v3 bought its
    #    margins with costs that made quitting cheap. The mechanism here is the
    #    per-step GRADIENT out of the crouch (rise_progress), not the size of
    #    the gap at the end.
    per_step = {k: v / steps for k, v in
                (("book", book_t), ("high", high_t), ("low", low_t))}
    assert per_step["book"] > per_step["low"] + 3.0, per_step
    assert per_step["book"] > per_step["high"] + 3.0, per_step


def test_rising_out_of_the_crouch_pays_at_every_millimetre():
    """The v4 thesis, measured: from the parked crouch, EVERY step of climbing
    is worth strictly more than staying put — which was never true before.

    v1 parked at 0.088 m and stayed for 2.7 s. Under v1's stack the marginal
    step of rising bought only the difference between two Gaussians; here it
    also banks progress credits that can never be taken back."""
    def marginal(rise_per_step):
        """Total pay over the rise window for climbing at this rate from the
        crouch, minus what staying in the crouch would have paid."""
        got = []
        for per_step in (0.0, rise_per_step):
            env = _FakeEnv()
            env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
            env.run_to(HOLD_END_S)
            total, z = 0.0, BOW_Z
            while (float(env.episode_length_buf[0]) * env.step_dt
                   < RISE_PAY_END_S - 1e-9):
                z = min(z + per_step, STAND_Z)
                env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, z]])
                env.tick()
                total += (
                    _CFG.rewards["rise_progress"].weight * float(
                        microduck_mdp.bow_rise_progress_reward(
                            env, **_cfg_params("rise_progress"))[0])
                    + _CFG.rewards["bow_height"].weight * float(
                        microduck_mdp.bow_height_reward(
                            env, **_cfg_params("bow_height"))[0]))
            got.append(total)
        return got[1] - got[0]

    # Positive at every rate from the very slowest crawl to the rate cap:
    # there is no speed at which starting to rise is worse than staying put.
    rates = (0.0001, 0.0002, 0.0005, 0.001, 0.0012)
    gains = [marginal(r) for r in rates]
    assert all(g > 0.0 for g in gains), gains
    # And the best-paid rate is the one that TRACKS THE RAMP (0.6 mm/step =
    # 0.03 m/s), not the fastest — the progress money is speed-invariant, so
    # the height Gaussian is free to price the schedule. Rising faster than the
    # ramp buys nothing extra, which is AGENTS.md's "no jackpots" again.
    best = rates[gains.index(max(gains))]
    assert best == 0.0005, (rates, gains)
    assert gains[-1] < max(gains), "sprinting up overshoots the height ramp"
    # A tenth of the climb is worth about a tenth of the progress money, which
    # is the property none of v1-v3 had: partial credit for a partial rise.
    tenth = 0.1 * BOW_DIP_M
    env = _FakeEnv()
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
    env.run_to(HOLD_END_S)
    banked, z = 0.0, BOW_Z
    while z < BOW_Z + tenth - 1e-9:
        z += 0.0005
        env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, z]])
        env.tick()
        banked += _CFG.rewards["rise_progress"].weight * float(
            microduck_mdp.bow_rise_progress_reward(
                env, **_cfg_params("rise_progress"))[0])
    full = RISE_CREDITS * _CFG.rewards["rise_progress"].weight
    assert math.isclose(banked, 0.1 * full, rel_tol=1e-5), (banked, full)


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
