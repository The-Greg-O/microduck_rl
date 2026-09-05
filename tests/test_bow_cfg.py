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
    HOLD_END_S,
    NOT_RISEN_BAND,
    RISE_END_S,
    RISEN_TOL,
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
                "stand_pose", "risen", "track_linear_velocity",
                "track_angular_velocity"):
        assert cfg.rewards[pos].weight > 0, pos
    for cost in ("foot_lift", "not_risen", "drift", "foot_slip",
                 "action_rate_l2", "dof_pos_limits", "self_collisions"):
        assert cfg.rewards[cost].weight < 0, cost
    # Feet planted is the hardest constraint of the trick — heavier than the
    # slope that pushes the robot back up out of the crouch.
    assert cfg.rewards["foot_lift"].weight <= -3.0
    assert cfg.rewards["foot_lift"].weight < cfg.rewards["drift"].weight
    assert cfg.rewards["foot_lift"].weight < cfg.rewards["not_risen"].weight
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


def test_not_risen_cost_ramps_over_the_stand_phase():
    """Bounded on both axes — a slope out of the crouch, not a cliff."""
    env = _FakeEnv()
    cost = lambda: float(microduck_mdp.bow_not_risen_penalty(
        env, stand_z=STAND_Z, crouch_z=BOW_Z, band=NOT_RISEN_BAND,
        start_s=RISE_END_S, end_s=STAND_END_S)[0])
    env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, BOW_Z]])
    env.tick()
    assert cost() == 0.0, "must not tax the dip"
    env.run_to(HOLD_END_S)
    assert cost() == 0.0, "must not tax the hold"
    env.run_to(RISE_END_S)
    assert cost() == 0.0, "the ramp starts at zero where the stand phase does"
    env.run_to(RISE_END_S + 0.5)
    half = cost()
    assert math.isclose(half, 0.5, abs_tol=1e-5)
    env.run_to(STAND_END_S)
    assert math.isclose(cost(), 1.0, abs_tol=1e-5)
    # Monotone in height at a fixed time: every millimetre up pays less.
    heights, costs = [BOW_Z, 0.088, 0.092, STAND_Z - NOT_RISEN_BAND, STAND_Z], []
    for z in heights:
        env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, z]])
        costs.append(cost())
    assert costs == sorted(costs, reverse=True)
    assert math.isclose(costs[0], 1.0, abs_tol=1e-5), "saturates at the crouch"
    assert math.isclose(costs[-1], 0.0, abs_tol=1e-5), "free at standing height"
    # A cliff would have no intermediate values; this is a slope in z.
    assert math.isclose(costs[1], 0.7, abs_tol=1e-5)   # the height v1 parked at
    assert math.isclose(costs[2], 0.3, abs_tol=1e-5)


def test_rising_beats_parking_in_the_crouch_per_step():
    """The arithmetic (b) of the v2 fix, on the real cfg weights: in the stand
    phase, a finished stand must pay CLEARLY more per step than the crouch
    bow-v1 parked in (trunk 0.088 m, still nose-down, head still down)."""
    cfg = make_microduck_bow_env_cfg()
    legs = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

    def stand_phase_pay(z, y_rot, head, leg_offsets):
        # y_rot is the rotation about +y; nose_up = -y_rot (see
        # test_measured_signs_are_not_flipped), so nose-DOWN is y_rot > 0.
        env = _FakeEnv()
        env.run_to(STAND_END_S)
        env._asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, z]])
        env._asset.data.root_link_quat_w = torch.tensor(
            [[math.cos(y_rot / 2), 0.0, math.sin(y_rot / 2), 0.0]])
        q = torch.zeros(1, 14)
        q[0, 5], q[0, 6] = head
        for i, v in leg_offsets.items():
            q[0, i] = v
        env._asset.data.joint_pos = q
        env.tick()
        w = lambda n: cfg.rewards[n].weight
        return (
            w("bow_height") * float(microduck_mdp.bow_height_reward(env)[0])
            + w("bow_pitch") * float(microduck_mdp.bow_pitch_reward(env)[0])
            + w("bow_head") * float(microduck_mdp.bow_head_reward(env)[0])
            + w("bow_upright") * float(microduck_mdp.bow_upright_reward(env)[0])
            + w("stand_pose") * float(
                microduck_mdp.bow_stand_pose_reward(env, joint_indices=legs)[0])
            + w("not_risen") * float(microduck_mdp.bow_not_risen_penalty(
                env, stand_z=STAND_Z, crouch_z=BOW_Z, band=NOT_RISEN_BAND,
                start_s=RISE_END_S, end_s=STAND_END_S)[0])
        )

    # A 3 cm crouch bends knee ≈ 0.5 rad and hip_pitch / ankle ≈ 0.25 rad.
    crouched = {2: 0.25, 3: 0.5, 4: 0.25, 11: 0.25, 12: 0.5, 13: 0.25}
    parked = stand_phase_pay(
        0.088, BOW_PITCH_RAD, microduck_mdp.BOW_HEAD_DOWN, crouched)
    risen = stand_phase_pay(STAND_Z, 0.0, microduck_mdp.BOW_HEAD_UP, {})

    # Every factor is ≈ 1 at the finish, and `not_risen` is zero there.
    assert math.isclose(risen, 15.0, rel_tol=1e-3)   # 4+2+2+2+5, track excluded
    assert parked < 5.0
    assert risen - parked > 10.0, (
        f"rising must be the argmax by a margin (risen {risen:.2f} vs parked "
        f"{parked:.2f}); v1's gap was 7.4/step and it parked anyway")
    # The one-shot is a milestone on top, not the thing doing the work.
    assert cfg.rewards["risen"].weight < risen - parked


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
