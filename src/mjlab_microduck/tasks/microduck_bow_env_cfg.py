"""Microduck BOW task — the play bow, a dog's invitation to play.

Episodic skill: from a standing start, dip the trunk down and forward (a
~3 cm crouch, trunk slightly nose-down), head lowered, hold about a second,
then rise back to the standing pose — all inside a 4 s episode, with BOTH FEET
PLANTED the whole time. Deploys as a constant-command `episodic` skill
(`robotctl policy add bow <repo>`), duration = EPISODE_LENGTH_S, bindable to a
gamepad button, and the natural partner to tippy-taps in the social layer.

Built on the velocity recipe (AGENTS.md workflow step 1) so DR, observation
noise, delays, the BAM actuator and the 61D obs contract come for free, and
structured exactly like the tippy-taps port: a pinned command, a short
episode, one per-env memory refreshed by every reward term, and symmetry ON
(a bow is left/right symmetric).

── The phase machine ────────────────────────────────────────────────────────
`microduck_mdp._bow_update` keeps the per-env memory and derives, from episode
TIME alone, a phase (dip 0–1 s, hold 1–2 s, rise 2–3 s, stand 3–4 s) plus two
blends: `s` ramps 0→1→0 across dip/hold/rise and drives the trunk height and
pitch targets; `r` ramps 0→1 across the rise and gates the head LIFT. Both
targets are SLEWED ramps, not per-phase steps, so diving early pays nothing
(AGENTS.md "No jackpots") — the trajectory of the crouch is still the
policy's to discover, only its schedule is fixed.

── Physics verified before training (AGENTS.md workflow step 2) ─────────────
Forward kinematics on robot_walk.xml, feet flat:
  * Head sign: positive neck_pitch TUCKS THE CHIN AND RAISES THE BEAK. Head
    DOWN is neck_pitch NEGATIVE with head_pitch POSITIVE — BOW_HEAD_DOWN
    (-0.30, +0.35) drops the mouth_tip site 3.9 cm and pushes it 1.5 cm
    forward; BOW_HEAD_UP (+0.25, -0.25) lifts it 1.8 cm. The naive sign is
    backwards; measured, not guessed.
  * Trunk pitch sign: nose_up = asin(R[2,0]) is positive nose-UP, so the bow
    target is NEGATIVE (BOW_PITCH_RAD = 10° nose-down).
  * Reachability: the 3 cm crouch at 10° nose-down with both feet level is
    deep inside the joint envelope (the minimum trunk z at that pitch is
    ~5 cm, well below the 8.5 cm target).
  * Static margin: a CoM-over-support sweep (crouch depth × trunk pitch ×
    head dip, symmetric leg chain, trunk xy pinned) puts the bow pose within
    ±1 cm of the foot polygon edge, and the CoM falls off the BACK as the
    crouch deepens. Both the head dip and the nose-down pitch move the CoM
    FORWARD and improve the margin, which is why the bow keeps them rather
    than being a bare squat. The sweep pins the trunk xy, which the real
    policy will not — it can shift its weight over the feet — so these are
    pessimistic numbers, but they are why BOW_PITCH_DEG is 10 and not 15
    (the 15° branch loses ~3 cm of margin) and why the foot-lift cost below
    is the heaviest single term.

── Reward design, in the playbook's terms ───────────────────────────────────
  * height / pitch / head track a slewed ramp (Gaussian): no jackpot, no
    waypoint camping, and the reward is zero-mean-safe at every phase.
  * stand_pose is a TIME window (the stand phase), not a state gate — the
    policy cannot park in it early, and it is what makes the episode finish
    in a clean stand instead of leaving the robot folded.
  * upright prices LATERAL tilt only. The stock `upright` term prices total
    tilt and would fight the intended nose-down pitch, so it is dropped.
  * foot_lift is the anti-fall constraint that matters most: Microducks fall
    while RISING out of a crouch, and every one of those falls starts with a
    foot leaving the floor. Charged per foot, so a single lift already costs —
    but only once a foot has been airborne for 40 ms, so the heel flicker of a
    legitimate push-up is free (see the v2 section below).
  * phase_match pays one credit per phase for tracking the ramp end to end —
    a milestone, not per-step pay, and the one term no static height can earn.
  * not_dipped / not_risen are the two halves of "a static trunk is wrong in
    every phase": above the ramp through the dip and hold, below it through the
    rise and the finish. Both measure against the ramp, so they never charge a
    policy that is where the trajectory says it should be.
  * risen: a one-shot milestone for getting back up, feet planted, having
    actually reached the crouch.
  * collapse termination: a trunk below BOW_COLLAPSE_Z is a collapse, not a
    bow — terminating stops the policy from farming the crouch reward's tail
    from a heap on the floor.

── v1 failed by PARKING IN THE CROUCH; what v2 changes ──────────────────────
The rendered rollout of bow-v1 (checkpoint 1999, BAM servos) got the whole
first half right: trunk 0.127 → 0.088 m by 1.3 s, 10° nose-down, head down to
0.202 m, both feet planted, lateral tilt 10°, and a clean hold. Then it simply
stayed there — 0.088 m from 1.3 s all the way to 4.0 s, straight through the
rise and the stand phase. It had found that parking is cheap: the +2.0
stand-phase pose term and the height ramp back up did not outweigh what rising
RISKS (a foot-lift cost at -3.0 that fires on the heel flicker every push-up
produces, and a collapse termination that forfeits the remainder of a 4 s
episode). Rising was the scary part, and nothing made it the argmax.

Four changes, and nothing else:
  (a) `risen` (+5.0), ONE step per episode — the first step at/after 2 s with
      the trunk back within 1 cm of STAND_Z, both feet planted, and the crouch
      actually reached. Modelled on happy-spin's `spin_complete`: one-shot, so
      it is a milestone and not a jackpot to camp on.
  (b) the stand-phase stack is heavier — W_HEIGHT 3.0 → 4.0 and W_STAND_POSE
      2.0 → 5.0 — so per-step pay in the stand phase clearly favours standing.
  (c) `not_risen` (-2.5), a bounded double ramp (time 3→4 s × depth below
      STAND_Z - 2 cm), so the crouch gets steadily more expensive at the
      finish. A slope, not a cliff: 5 mm higher always pays strictly less.
  (d) `foot_lift` keeps its -3.0 but exempts airborne spells under 40 ms, so
      the unweighted heel of a legitimate rise is not priced as a hop.

The per-step arithmetic in the stand phase (dt = 0.02, 50 steps), comparing
the parked crouch (z = 0.088, still nose-down, head still down, legs folded:
height 0.162, pitch 0.121, head 0.027, pose 0.405, upright 1.0) against a
finished stand (every factor 1.0):

      weights (H,P,Hd,U,SP,track)   crouch-parked   risen    gap/step
  v1  (3, 2, 2, 2, 2, 0.5+0.5)          4.59        12.00      7.41
  v2  (4, 2, 2, 2, 5, 0.5+0.5)          5.09        16.00     10.91
      (v2 crouch = 5.97 of pay minus 0.88 of mean `not_risen` cost: the time
       ramp averages 0.5 over the stand phase and 0.088 m is 0.70 of the way
       down the depth ramp)

Over the 50-step stand phase that is 371 → 545, plus the one-shot +5. Over the
whole 4 s episode, a textbook bow beats the v1-parked trajectory by 603 points
under the old weights and by 821 under the new ones (2460 vs 1639) — a 40% →
50% margin. The marginal step of staying down now costs ~11 instead of ~7,
and the rise itself got cheaper to attempt (d).

── v2 failed by PARKING HIGHER; what v3 changes ─────────────────────────────
The rendered rollout of bow-v2 (checkpoint 1999, BAM, twist zero, 4 s) never
bowed at all. The trunk went from the 0.127 m spawn to 0.101 m by 0.5 s and
STAYED THERE for the remaining 3.5 s: pitch -4°, head 0.218 m, feet planted,
no dip, no rise. 0.101 m is the compromise — close enough to the 0.085 crouch
target and to the 0.115 stand target to collect partial credit from BOTH
phases' Gaussians at v2's std of 2 cm (0.53 against the crouch, 0.61 against
the stand), while never paying the action-rate cost of moving and never
risking the rise. A static pose that half-satisfies every phase was the
optimum, and the +5 one-shot was nowhere near enough to buy the motion.

The v2 log (iteration 1999; `Episode_Reward/x` is the MEAN WEIGHTED VALUE PER
STEP — the terms sum to 7.15 and 4 × 7.15 = 28.50 = the reported mean reward)
says the same thing term by term, for a robot that did nothing:

    bow_height 2.83 (0.71 of max)   bow_upright 1.94 (0.97)
    bow_pitch  1.40 (0.70)          track_lin/ang 0.42 / 0.16
    bow_head   1.22 (0.61)          stand_pose 0.26 (0.05)
    action_rate_l2 -0.71            drift -0.20   foot_lift -0.08
    risen 0.0005  →  the one-shot fired in ~2% of episodes
    not_risen -0.0003  →  ZERO. Its dead band ended at 0.095 m and the
                          policy parked at 0.101: v2's anti-parking term
                          never touched the pose it was written to stop.

Six changes:
  (a) TOLERANCES SMALLER THAN THE MOTION THEY MEASURE. height std 0.02 →
      0.006 m, pitch 0.12 → 0.05 rad, head 0.30 → 0.10 rad. At 6 mm the 0.101
      compromise scores 0.001 against the crouch target and 0.004 against the
      stand target instead of 0.53 and 0.61. A std as wide as the amplitude
      being asked for pays for not moving.
  (b) action_rate_l2 -0.2 → -0.02, still with no curriculum. v2 paid -0.71 per
      step for the jitter of a robot standing STILL; a 3 cm dip and rise cost
      more than that, so the tax alone bought stillness. Motion is the trick.
  (c) `phase_match` (+25, at most 4 per episode): one credit per PHASE, paid
      only if the trunk stayed within 1 cm of the SLEWED ramp for every step
      of that phase. A parked trunk is outside the corridor at some point in
      every phase, so it collects nothing; being early is equally worthless.
      The whole-phase latch matters — a per-step 1 cm window would be farmable,
      because a 3 cm/s ramp passes within 1 cm of ANY parked height for 0.67 s
      of the dip and again for 0.67 s of the rise.
  (d) TWO SYMMETRIC OFF-RAMP COSTS, both -5.0 and both measured against the
      ramp rather than against a fixed height (which is what made v2's version
      a no-op): `not_dipped` charges being >1 cm ABOVE the ramp from 0.8 s to
      2.0 s, `not_risen` charges being >1 cm BELOW it from 2.5 s to the end,
      each saturating 5 mm further out. Parking high now loses the dip and the
      hold; parking low loses the rise and the stand; a trunk ON the ramp is
      free in both windows, so neither cost can ever fight the trajectory.
  (e) SURVIVAL INCOME HALVED: bow_upright 2.0 → 1.0, track 0.5 → 0.25 each.
      These paid v2's frozen pose 2.97 of its 7.15 per step for having a level
      roll and not moving — 42% of the take for doing nothing. They stay (roll
      safety, "don't travel"), but they no longer fund parking.
  (f) `stand_pose` becomes a MULTIPLICATIVE composite (leg pose × trunk back at
      STAND_Z within 1 cm), per AGENTS.md's rule for goal states. Additively it
      leaked: four of the ten leg joints sit at HOME in any sagittal crouch, so
      its mean floored at 0.4 and a trunk parked 1.4 cm low still collected
      0.79 of a term weighted 5.0.
  Kept unchanged: the `risen` one-shot (+5), `foot_lift` (-3.0) with the 40 ms
  flicker exemption, `drift`, the collapse termination, symmetry, 2000 iters.

THE ARITHMETIC (`tests/test_bow_cfg.py::test_the_three_strategies`, real
reward functions, all 200 steps of the episode, sums of weighted values before
mjlab's ×dt — divide by 200 for wandb's `Episode_Reward` units):

                       textbook   park@0.101   park@0.088
      bow_height          800.0        197.8        433.4
      bow_pitch           400.0        127.9        224.4
      bow_head            400.0         57.5        214.4
      bow_upright         200.0        200.0        200.0
      stand_pose          255.0         20.1          0.0
      phase_match         100.0          0.0         50.0
      risen                 5.0          0.0          0.0
      not_dipped            0.0       -274.2          0.0
      not_risen             0.0       -215.4       -365.0
      track (≈1.0 each)   100.0        100.0        100.0
      ------------------------------------------------------
      TOTAL              2260.0        213.7        857.3
      per phase   dip     465.5        264.4        465.5
                  hold    500.0       -173.6        500.0
                  rise    505.0        214.7         70.3
                  stand   789.5        -91.8       -178.5
      per step             11.30         1.07         4.29

park@0.101 is the trajectory v2 actually learned (settle to 0.101 by 0.5 s,
then frozen); park@0.088 is v1's (a textbook dip and hold, then frozen in the
crouch — the strongest version of the low park, credited in full for the half
of the bow it really does). The textbook bow wins by 10.6× and 2.6×, and each
park is NEGATIVE in exactly the phases its parking violates: the high park
bleeds through the hold (-174) and the stand (-92), the low park through the
stand (-179) and its whole second half (-108). Measured against the run we are fixing: the v2 policy collected
7.15 per step for that frozen pose; the same trajectory collects 1.07 under
these weights, while a textbook bow collects 11.3.

The low park stays positive over the whole episode (+857) and no weight of -5
can change that — its first two seconds are a genuine bow and honestly earn
~950, which -5 over the 100 remaining steps cannot erase (it would take about
-13, which would also make COLLAPSING beat parking, since a terminated episode
scores zero from then on). Losing by 2.6× with a negative finish is the right
shape for PPO; making a half-done trick net-negative is not.

WATCH FOR (the new failure mode this trades into): a policy that cannot dip
is now worse off than one that terminates during the not_dipped window
(-3.5/step vs 0). The descent gradient points through the crouch corridor, not
past it — the collapse floor is 2.5 cm BELOW the crouch target — but if `Mean
episode length` falls under ~195 or `Episode_Reward/not_dipped` stays pinned
while episodes shorten, the policy is buying its way out by falling over, and
W_NOT_DIPPED should come down before anything else is touched.

Reward MASS check (AGENTS.md: compare mass, not weights, when regularizers are
shared): stand_pose only pays in the last quarter of the episode, so the
episode-average positive stack goes 10.5 → 12.25 → 11.8 (v1 → v2 → v3), still
in line with the velocity recipe's ~11 that the inherited regularizers were
tuned against — but action_rate_l2 is deliberately 10× below that scaling.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

# A bow is left/right symmetric: the mirror loss is a valid prior and buys
# sample efficiency (unlike the one-footed tricks, where it must be off).
ENABLE_SYMMETRY = True

EPISODE_LENGTH_S = 4.0

# ── Phase boundaries (seconds from episode start) ────────────────────────────
DIP_END_S = microduck_mdp.BOW_DIP_END_S      # 1.0 — trunk down and forward
HOLD_END_S = microduck_mdp.BOW_HOLD_END_S    # 2.0 — hold the bow
RISE_END_S = microduck_mdp.BOW_RISE_END_S    # 3.0 — back up
STAND_END_S = microduck_mdp.BOW_STAND_END_S  # 4.0 — settle standing

# ── Geometry (measured; see the module docstring) ────────────────────────────
STAND_Z = microduck_mdp.BOW_STAND_Z          # 0.115 measured standing trunk z
BOW_DIP_M = 0.030                            # the trunk drops 3 cm
BOW_Z = microduck_mdp.BOW_CROUCH_Z           # 0.085 = STAND_Z - BOW_DIP_M
assert abs((STAND_Z - BOW_DIP_M) - BOW_Z) < 1e-9
# Below this the robot has collapsed, not bowed (2.5 cm under the crouch, so a
# legitimate overshoot of the height ramp does not terminate).
BOW_COLLAPSE_Z = 0.060
BOW_PITCH_DEG = 10.0
BOW_PITCH_RAD = microduck_mdp.BOW_PITCH_RAD
assert abs(math.radians(BOW_PITCH_DEG) - BOW_PITCH_RAD) < 1e-3

# ── Rising out of the crouch (see "v1 failed by PARKING" above) ──────────────
RISEN_TOL = microduck_mdp.BOW_RISEN_TOL          # 0.01 — "back within 1 cm"
FLICKER_S = microduck_mdp.BOW_FLICKER_S          # 0.04 — 2 steps of contact loss

# ── Staying ON THE RAMP (v3; see "v2 failed by PARKING HIGHER" above) ────────
MATCH_TOL = microduck_mdp.BOW_MATCH_TOL              # 0.01 — the ramp corridor
MATCH_GRACE_S = microduck_mdp.BOW_MATCH_GRACE_S      # 0.3 — the spawn drop is free
OFF_RAMP_BAND = microduck_mdp.BOW_OFF_RAMP_BAND      # 0.01 forgiven off the ramp
OFF_RAMP_SLOPE = microduck_mdp.BOW_OFF_RAMP_SLOPE    # then 5 mm to full cost
NOT_DIPPED_START_S = microduck_mdp.BOW_NOT_DIPPED_START_S  # 0.8
NOT_DIPPED_END_S = microduck_mdp.BOW_NOT_DIPPED_END_S      # 2.0
NOT_RISEN_START_S = microduck_mdp.BOW_NOT_RISEN_START_S    # 2.5

# ── Tolerances: each std is SMALLER than the motion it measures (v3) ─────────
HEIGHT_STD = 0.006          # m,   vs a 3 cm dip      (v2: 0.02 — half the dip)
PITCH_STD = 0.05            # rad, vs a 0.175 rad tilt (v2: 0.12)
HEAD_STD = 0.10             # rad, vs 0.25–0.35 rad deltas (v2: 0.30)
STAND_POSE_STD = 0.12       # rad, leg joints at HOME
STAND_POSE_HEIGHT_STD = 0.01  # m, the height factor of the stand composite

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# ── Reward weights ───────────────────────────────────────────────────────────────────────────
# Episode-average positive task mass ≈ 11.8 for a textbook bow (stand_pose
# pays only in the last quarter; the phase credits are worth 0.5/step spread
# over four milestones), comparable to velocity's ~11, so the shared
# regularizers keep their relative strength (AGENTS.md: compare reward MASS,
# not weights, when copying regularizers between envs) — with the deliberate
# exception of action_rate_l2, 10× below that scaling. The full three-strategy
# arithmetic is in the module docstring and in
# `tests/test_bow_cfg.py::test_the_three_strategies`.
W_HEIGHT = 4.0        # the shape of the bow — and the ramp back UP (v1: 3.0)
W_PITCH = 2.0         # nose-down: what makes the crouch read as a bow
W_HEAD = 2.0          # head down, then lifted
W_STAND_POSE = 5.0    # the finish, stand phase only (v1: 2.0 — too cheap to
                      # outbid parking in the crouch)
W_PHASE_MATCH = 25.0  # v3: one credit per PHASE for tracking the ramp end to
                      # end — 4 per episode, ~100 of a textbook bow's ~2200
                      # return, and zero for any static height. Milestones,
                      # not per-step pay, so there is nothing to camp on.
W_UPRIGHT = 1.0       # lateral tilt only — the pitch is intended. HALVED in
W_TRACK = 0.25        # v3 with the twist tracking: at 2.0 / 0.5+0.5 these paid
                      # a motionless robot 2.97 of the 7.15 reward/step that
                      # v2's parked pose collected. Survival income is what
                      # funds parking; the trick has to be where the money is.
W_RISEN = 5.0         # ONE step per episode: back up, feet planted. Same size
                      # and same one-shot shape as happy-spin's completion.
W_NOT_DIPPED = -5.0   # v3: parked HIGH loses the dip and the hold …
W_NOT_RISEN = -5.0    # … and parked LOW loses the rise and the stand (v2: -2.5
                      # and, measured against a fixed height, ~0 in practice)
W_FOOT_LIFT = -3.0    # both feet planted: still the hardest per-step rule
                      # inside the corridor, but no longer able to make holding
                      # still the safest option — being off the ramp costs more
W_DRIFT = -1.5        # a bow happens on the spot
W_SLIP = -0.2
W_ACTION_RATE = -0.02  # v3: 10× lighter. v2 paid -0.71/step for the jitter of
                       # a robot standing STILL; moving through a 3 cm dip and
                       # back cost more, and that tax was the whole reason a
                       # frozen pose won. Motion is the trick: make it cheap.


def make_microduck_bow_env_cfg(
    play: bool = False, rough: bool = False
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Command: pinned near zero (obs-shape parity; being selected IS the
    # trigger, as for tippy-taps and the kicks) ──────────────────────────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    if hasattr(command, "rel_turn_in_place_envs"):
        command.rel_turn_in_place_envs = 0.0
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # Curricula that drive things this env pins or replaces. The head/body pose
    # COMMANDS stay (tiny ranges, obs slots alive per the 61D contract) — only
    # their widening curricula and their reward go, since the bow drives the
    # head itself. com_range / head_com_range (DR) are kept.
    for gone in (
        "standing_envs",        # drives a command we no longer vary
        "head_pose_range",      # head is phase-driven here, not commanded
        "body_pose_range",      # body_pose tracking is weight 0 (as in velocity)
        "head_pose_bias_weight",# ramps a term this env deletes
        "action_rate_weight",   # would ramp -0.1 → -1.0; see W_ACTION_RATE
    ):
        if gone in cfg.curriculum:
            del cfg.curriculum[gone]

    # ── Rewards: drop the gait / walking terms ──────────────────────────────
    for name in (
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "pose",         # replaced by the stand-phase pose below
        "upright",      # replaced by bow_upright (lateral tilt only)
        "head_pose_bias",
    ):
        if name in cfg.rewards:
            del cfg.rewards[name]

    # The head is driven by the bow phase, not by the head_pose command. The
    # term stays at weight 0 so the obs slot's role is visible in the log
    # (same pattern velocity uses for body_pose_tracking).
    cfg.rewards["head_pose_tracking"].weight = 0.0

    # Tracking a ~zero twist = "don't travel, don't spin", kept light.
    cfg.rewards["track_linear_velocity"].weight = W_TRACK
    cfg.rewards["track_angular_velocity"].weight = W_TRACK

    # foot_slip is command-gated in the velocity recipe and would be dead at a
    # pinned ~zero command; -1.0 turns the gate permanently on.
    cfg.rewards["foot_slip"].weight = W_SLIP
    cfg.rewards["foot_slip"].params["command_threshold"] = -1.0

    # Smoothness: FIXED and light. The velocity curriculum ramps this to -1.0
    # by iter 1500, which over a 2000-iter episodic run would spend most of
    # training taxing a slow, large, deliberate motion.
    cfg.rewards["action_rate_l2"].weight = W_ACTION_RATE

    sensor_name = "feet_ground_contact"
    trunk = SceneEntityCfg("robot", body_names=("trunk_base",))

    cfg.rewards["bow_height"] = RewardTermCfg(
        func=microduck_mdp.bow_height_reward,
        weight=W_HEIGHT,
        params={
            "sensor_name": sensor_name,
            # The ramp itself lives in `_bow_update` (STAND_Z → BOW_Z → STAND_Z);
            # one source of truth for the height reward, both off-ramp costs and
            # the phase latch.
            "std": HEIGHT_STD,
        },
    )
    cfg.rewards["bow_pitch"] = RewardTermCfg(
        func=microduck_mdp.bow_pitch_reward,
        weight=W_PITCH,
        params={
            "sensor_name": sensor_name,
            "pitch_rad": BOW_PITCH_RAD,
            "std": PITCH_STD,   # ~3°, well under the 10° target
        },
    )
    cfg.rewards["bow_head"] = RewardTermCfg(
        func=microduck_mdp.bow_head_reward,
        weight=W_HEAD,
        params={
            "sensor_name": sensor_name,
            "down_deltas": microduck_mdp.BOW_HEAD_DOWN,
            "up_deltas": microduck_mdp.BOW_HEAD_UP,
            "std": HEAD_STD,
        },
    )
    cfg.rewards["bow_upright"] = RewardTermCfg(
        func=microduck_mdp.bow_upright_reward,
        weight=W_UPRIGHT,
        params={"sensor_name": sensor_name, "std": 0.15},
    )
    cfg.rewards["stand_pose"] = RewardTermCfg(
        func=microduck_mdp.bow_stand_pose_reward,
        weight=W_STAND_POSE,
        params={
            "sensor_name": sensor_name,
            "joint_indices": _LEG_JOINTS,   # head is bow_head's job
            "std": STAND_POSE_STD,          # tight: this IS the standing pose
            "stand_z": STAND_Z,             # … MULTIPLIED by being back up …
            "height_std": STAND_POSE_HEIGHT_STD,   # … within a centimetre
        },
    )
    # Following the ramp end-to-end, one credit per phase. This is the term a
    # static pose cannot touch at any height (module docstring, v3 (c)).
    cfg.rewards["phase_match"] = RewardTermCfg(
        func=microduck_mdp.bow_phase_match_reward,
        weight=W_PHASE_MATCH,
        params={"sensor_name": sensor_name},
    )
    # The finish: a one-shot for getting back up, plus the two symmetric
    # off-ramp costs that make a static trunk lose in EVERY phase.
    cfg.rewards["risen"] = RewardTermCfg(
        func=microduck_mdp.bow_risen_bonus,
        weight=W_RISEN,
        params={"sensor_name": sensor_name},
    )
    cfg.rewards["not_dipped"] = RewardTermCfg(
        func=microduck_mdp.bow_not_dipped_penalty,
        weight=W_NOT_DIPPED,
        params={
            "sensor_name": sensor_name,
            "band": OFF_RAMP_BAND,
            "slope": OFF_RAMP_SLOPE,
            "start_s": NOT_DIPPED_START_S,  # commit to the crouch by 0.8 s …
            "end_s": NOT_DIPPED_END_S,      # … off again for the rise
        },
    )
    cfg.rewards["not_risen"] = RewardTermCfg(
        func=microduck_mdp.bow_not_risen_penalty,
        weight=W_NOT_RISEN,
        params={
            "sensor_name": sensor_name,
            "band": OFF_RAMP_BAND,
            "slope": OFF_RAMP_SLOPE,
            "start_s": NOT_RISEN_START_S,   # committed to standing by 2.5 s
            "end_s": None,                  # … until the episode ends
        },
    )
    cfg.rewards["foot_lift"] = RewardTermCfg(
        func=microduck_mdp.bow_foot_lift_penalty,
        weight=W_FOOT_LIFT,
        params={
            "sensor_name": sensor_name,
            # 2 control steps: the heel flicker of a push-up is free, a hop
            # is not. Without this, -3.0 made the rise itself look expensive.
            "flicker_s": FLICKER_S,
        },
    )
    cfg.rewards["drift"] = RewardTermCfg(
        func=microduck_mdp.bow_drift_penalty,
        weight=W_DRIFT,
        params={"sensor_name": sensor_name},
    )

    # ── Terminations: the base fall check, plus "a collapse is not a bow" ────
    cfg.terminations["collapsed"] = TerminationTermCfg(
        func=microduck_mdp.root_height_below,
        time_out=False,
        params={"min_height": BOW_COLLAPSE_Z, "asset_cfg": trunk},
    )
    return cfg


MicroduckBowRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="bow",
    run_name="bow",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=2_000,
)
