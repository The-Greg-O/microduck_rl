"""Microduck PIVOT task — one full turn about a single planted foot.

Episodic skill: from a standing start the PLANTED foot stays exactly where it
is, in contact, as the pin; the other foot paddles — lift, reach sideways,
plant, push — so the trunk pinwheels around the pin through a full 360°, and
then the robot settles back onto both feet in the STAND pose. All inside a 4 s
episode. Deploys as a constant-command `episodic` skill
(`robotctl policy add pivot-left <repo>`), duration = EPISODE_LENGTH_S,
bindable to a gamepad button.

Built on the velocity recipe (AGENTS.md workflow step 1) so DR, observation
noise, delays, the BAM actuator and the 61D obs contract come for free, and
structured exactly like happy-spin and bow: a command pinned for the whole
episode, a short episode, one per-env memory ticked once per step by every
reward term, potential-based progress, a one-shot completion bonus and a settle
phase.

── Direction: ONE network, both ways (a daemon finding) ─────────────────────
The premise happy-spin was written under — "the 61D observation contract has no
free slot to tell the policy which way to turn" — is not what the daemon
actually does. During a skill window `robotd/src/control.rs` does not pass the
live twist AND does not necessarily zero it: it builds a fresh command block
from the skill's CONFIGURED CONSTANT twist,
`robotd_params::SkillDef::command`, with head and body zeroed. That constant is
zeros for most one-shots, but carrying a policy's own encoding is exactly what
it is for — the published flamingo reads `[flag, side, 0]`, so `[1, 1, 0]`
means "lift the left leg".

So the turn direction rides the twist YAW slot (obs[50]) as a ±1 FLAG:

  * in training, `microduck_mdp.PivotCommand` samples the sign per env at
    reset and writes it into the pinned command, so the network sees it in the
    observation exactly as it will on the robot;
  * `_pv_update` latches that sign for the episode, folds it into ONE signed
    rate (`_pv_omega` = dir × body-frame omega_z, + = the commanded way) that
    every direction-aware term reads, and derives the pin foot from it — the
    pin is the INSIDE foot of the turn, so +1 (counter-clockwise, turning left)
    pivots on the LEFT foot (site/contact slot 0, measured at +4.18 cm, the
    robot's left) and -1 on the right;
  * at deploy time ONE onnx installs as two skill entries, `pivot-left` with
    `command = [0, 0, 1]` and `pivot-right` with `[0, 0, -1]`.

`PIVOT_DIRECTION = None` is the bidirectional task. Setting it to ±1 pins one
direction (the factory takes a `direction` argument for the same reason), which
is the fallback if a bidirectional policy proves hard to train — but it costs
two training runs and two onnx files for what the command slot already carries.

Because the two feet have DIFFERENT ROLES — one pins, one paddles — the mirror
symmetry loss is a false prior here and is OFF. (Under a bidirectional policy
the mirror is not even a different task, it is the same policy under the
opposite flag, which the network has to learn as a response to the flag rather
than as a hard-wired equivariance.)

── Physics measured before training (AGENTS.md workflow step 2) ─────────────
On robot_walk.xml at the STAND keyframe, feet flat:
  * the foot sites sit ±4.18 cm off the trunk centreline → the trunk orbits the
    pin at a radius of ~4.2 cm (`PV_RADIUS_M`), which is why the drift cost
    here prices leaving that CIRCLE rather than leaving the spawn point: a
    happy-spin-style drift-from-home cost would charge the pivot itself.
  * the foot's collision mesh is ~4.1 × 5.4 cm → a foot rotating about a point
    inside its own sole moves its site by ≲1 cm, while a STEP moves it 4+ cm.
    Hence `PV_PIN_STD_M = 1.5 cm` for the pin pay and `PV_PIN_SAT_M = 4 cm`
    for the pin cost.
  * measured standing trunk z is 0.115 m (as standup / bow / happy-spin).

── Reward design, in the playbook's terms ───────────────────────────────────
  * `pivot_progress` pays Δ(accumulated yaw) — potential-based, so holding
    still pays zero and nothing can be farmed; capped per step (no pay for
    violence past ~1.5 turns/s) and capped in total at one turn.
  * `pivot_complete` is ONE step, not a per-step "you are done" — that would be
    the jackpot the playbook warns about and would buy a ballistic whip. Speed
    is bought by the SETTLE phase instead: finish early and more of the 4 s is
    spent collecting settle pay, which only pays while upright, stopped, on
    both feet and back in the STAND pose.
  * `pin` is what makes this a PIVOT and not a spin, and it is weighted to say
    so — but it pays ONLY WHILE THE POTENTIAL IS RISING (it is multiplied by
    the step's progress rate). Version 1 paid it unconditionally, which made
    standing still the argmax of the whole stack: contact × stay is maximal for
    a robot that never moves, so "pin 3.0 + height 1.0 + upright 2.0 ≈ 6/step
    for doing nothing" beat every attempt at a turn, and checkpoint 2499 duly
    stood there with both feet down 99% of frames. Happy spin never had this
    problem because it has no pin term to make stillness lucrative. Gated on
    progress, a pivot collects pin 2.0 + paddle ~1.5 per step of the turn while
    a two-footed spin collects ~0 of the pin pay AND pays the two pin costs.
  * `paddle` is tippy-taps' tap term with the roles split: it pays the FREE
    foot for being mid-stroke, in an air-time window, while the pin is planted
    and the turn is unfinished. Air time resets on contact, so every paid
    window has to be bought with a fresh plant — that is the alternation — and
    a foot parked in the air runs past the window and stops paying (that would
    be a flamingo).
  * `settle` is a PRODUCT (stopped × pose × both feet down), not a sum: an
    additive stack has a compromise basin where a still-turning one-legged
    crouch keeps most of it.
  * `stall` makes standing still LOSE rather than merely not-win: a bounded
    cost that ramps in after 0.5 s without progress, during the spin phase
    only, resetting the instant the turn advances. AGENTS.md's rule is that an
    attempt-TAX during discovery makes "do nothing" win; the inverse is a
    do-nothing tax, which makes attempting win.
  * `counter_yaw` is the SIGN FIX. The ±1 flag in the twist yaw slot was
    one-sided — turning the commanded way paid, turning the opposite way was
    free, and (through an un-floored raw integral) it also buried the potential
    where the policy could not reach it. So the only direction-dependent
    pressure on a policy that had not found the turn came from the pin/paddle
    loading, which yaws the trunk AGAINST the command: measured -21°/-17° on
    direction +1 and +28°/+33° on -1. The raw integral is now floored at zero
    and the wrong-way rate is charged per step, bounded.
  * `pin_broken` (v3) is what stops the pin being FOR SALE. v2 turned — and
    turned well — but as a two-footed shuffle-spin: contact fractions 0.91 and
    0.85, 6–17 cm of trunk drift. Every v2 pin pressure was graded and bounded,
    so trading ~2.0 of pin pay and ~2.5 of pin costs for ~4.0 of progress was
    simply a good deal. The pin is now a REQUIREMENT: off the floor for more
    than 40 ms or more than 3 cm from its anchor and it is broken, charged
    every step until re-planted within 2 cm of that anchor — and, decisively,
    THE PROGRESS POTENTIAL DOES NOT ADVANCE WHILE IT IS BROKEN. A shuffle earns
    no progress past its first 3 cm, which drops it below standing still.
    Gating the potential is safe now and was not in v1: the raw integral is
    floored and the potential is a running maximum, so a gate cannot burn
    banked degrees, only stop new ones accruing.
  * `paddle_reach` (v3) makes the free foot REACH rather than tap. The air-time
    window says the foot left the floor; the reach scores how far its contact
    point travelled between consecutive plants — a plateau over 4–8 cm with a
    Gaussian skirt, paid ONE SHOT on the plant so it cannot be farmed by
    parking the foot out wide. The window floor also rises 0.06 → 0.08 s: the
    v2 shuffle's alternating flicks cleared 0.06 s, so the paddle was paying
    for a twitch.
  * `two_feet_still` (v3) names the shuffle in one predicate: both feet in
    contact while the trunk yaws past 0.5 rad/s. Two planted feet and a turning
    trunk means the feet are sliding. It also charges a real pivot's
    double-support push (a third of the cadence, -0.67/step), which is
    deliberate — it pushes the cadence toward more air and shorter plants.
  * DISCOVERY IS MADE CHEAP. The pin reward's std starts at 4 cm (a whole
    footprint of slack) with the displacement cost at -0.5, and both tighten to
    1.5 cm / -2.0 by iteration 1500 on phase-aligned stages; the paddle opens
    at +3.0 so lifting the free foot pays from step one, instead of only
    risking the pin, tilt and radius costs.
  * `foot_slip` is DELETED as a global term and re-added for the PIN ONLY. The
    paddle foot is supposed to scuff — that is the push. For the pin, slip is
    exactly what "planted" forbids.
  * `angular_momentum` is DELETED (its 3D norm fights the trick head-on).
    `body_ang_vel` stays: x/y only, so it mates roll/pitch thrash without
    touching yaw.
  * `track_linear_velocity` / `track_angular_velocity` are DELETED. Both are
    actively wrong here: the trunk MUST travel (it orbits the pin at ~0.26 m/s
    during a 1 turn/s pivot) and the yaw slot is a FLAG, so tracking it as a
    rate would command a 1 rad/s crawl. Position is anchored by
    `pivot_radius` instead, which is what a pivot actually constrains.
  * The gait terms (`air_time`, `foot_clearance`, `foot_swing_height`) go too —
    and note they would NOT have been silently inert here as they are in the
    other pinned-command skills: they gate on command magnitude, and this
    task's command has a ±1 yaw slot, so they would have been live and would
    have demanded the PIN foot swing.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

# The pin foot and the paddle foot have DIFFERENT ROLES: the mirror loss would
# be a false prior. OFF (as for every one-footed trick).
ENABLE_SYMMETRY = False

EPISODE_LENGTH_S = 4.0
STAND_Z = microduck_mdp.PV_STAND_Z          # 0.115 measured standing trunk z

# None = the direction is sampled per env and read from the twist yaw slot
# (obs[50]); ±1 pins it. See the module docstring — the daemon feeds a skill a
# configured CONSTANT twist, so a flag in that slot is deployable.
PIVOT_DIRECTION: float | None = None

PV_TARGET_YAW = microduck_mdp.PV_TARGET_YAW      # 2π, one full turn
PV_DIRECTION_CCW = microduck_mdp.PV_DIRECTION_CCW  # +1 = ccw, pin = LEFT foot
PV_DIRECTION_CW = microduck_mdp.PV_DIRECTION_CW    # -1 = cw,  pin = RIGHT foot

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# ── Reward weights ───────────────────────────────────────────────────────────
# THE ARITHMETIC (dt = 0.02, 200 steps, rate cap = 1.5 turns/s so a 1 turn/s
# pivot scores `rate` = 2π/3π = 0.667 on every rate-scaled term).
#
# v1 was wrong because it gave "standing" the full pin pay; gating the pin on
# progress fixed that and v2 duly TURNED. v2 was wrong in a subtler place: every
# pin pressure was GRADED, so the pin was for sale. Checkpoint 2499 bought its
# way out — it turned +421°/+362° (+1) and -365°/-303° (-1), 360° in 0.7–1.1 s,
# no falls, correct direction, but as a TWO-FOOTED SHUFFLE-SPIN: per-foot
# contact fractions 0.91 / 0.85 and 6–17 cm of trunk drift. Giving up ~2.0 of
# pin pay and ~2.5 of bounded pin costs to keep ~4.0 of progress is a trade
# worth taking, so it took it.
#
# v3 makes the pin a REQUIREMENT rather than a price. Mean reward per step of
# the SPIN phase, PIVOT-SPECIFIC terms only (height 1.0 + upright 2.0 + head
# tracking are common to every row and cancel). These are MEASURED, not
# estimated — `test_the_shuffle_loses_per_step_during_the_spin_phase` and
# `test_the_arithmetic_pivoting_wins` run all three strategies through the real
# reward functions at these weights and print exactly this table. The pivot row
# is a 0.2 s up / 0.1 s down cadence at 1 turn/s with the free foot carrying its
# contact point 6 cm per stroke; the shuffle row is WHAT v2 ACTUALLY LEARNED —
# both feet down, both sliding (pin at 0.1 m/s), trunk yawing 1.5 rad/s:
#
#                                    still   pivoting   two-footed shuffle
#   progress      6.0 × rate          0.00     +4.00      +0.19   (→ 0 at 0.3 s)
#   pin           3.0 × stay × rate   0.00     +2.00      +0.05   (→ 0)
#   paddle        3.0, in-window      0.00     +0.92       0.00
#   paddle_reach  3.0, one shot/plant 0.00     +0.18       0.00
#   pin_broken   (-1.0 → -3.0)        0.00      0.00      -2.40
#   pin displacement (-0.5 → -2.0)    0.00      0.00      -1.63
#   two_feet_still   -2.0             0.00     -0.61      -0.80
#   pin slip         -0.5             0.00      0.00      -0.49
#   stall            -2.0            -0.99      0.00      -0.61
#   counter_yaw      -1.0             0.00      0.00       0.00
#                                   ───────   ───────    ────────
#   mean/step over the spin phase    -0.99     +6.49      -5.70
#   ...and once the pin is broken                         -7.98
#
# The decisive line is `progress → 0`. The shuffle's pin passes 3 cm from its
# anchor after ~0.3 s, `_pv_broken` latches, and from then on the potential does
# not advance at all: no progress means no pin pay either (it is rate-scaled)
# and the stall timer runs. The shuffle stops being SECOND BEST — which is what
# it was under v2's weights, and precisely why checkpoint 2499 learned it — and
# becomes the worst row on the board, below standing still.
#
# Settle phase (after the one-shot completion): progress / pin / paddle /
# paddle_reach / stall / pin displacement / pin_broken / two_feet_still are all
# gated off by `_pv_done`, and settle 4.0 takes over → +4.0/step.
#
# Whole 4 s episodes (200 steps), the same three strategies end to end:
#   pivot     49 steps of turn + the completion bonus + 151 of settle   +929
#   still     200 steps of a ramping stall                              -324
#   shuffle   never completes; ~185 steps at -8                        -1465
# Pivoting beats standing by ~1250 and the shuffle by ~2400, and finishing early
# still wins because every step saved from the turn is a step of settle. The
# ranking holds at BOTH ends of the pin curriculum (loose: +929 / -324 / -816),
# which is what stops the loose early stage from teaching a shuffle — the pin
# COSTS are made cheap for discovery, but the progress GATE never is.
W_PROGRESS = 6.0
W_COMPLETE = 5.0
W_PIN = 3.0
W_SETTLE = 4.0
W_PADDLE = 3.0
# The reach bonus matches the paddle's weight but fires ONCE per plant, so on a
# 0.3 s cadence it is worth ~+0.2/step: a directional nudge toward strokes that
# actually travel 4–8 cm, not a term that can out-shout progress.
W_PADDLE_REACH = 3.0
W_HEIGHT = 1.0
# Opens loose (-0.5) so the first clumsy attempt at lifting the free foot is
# affordable, tightened to -2.0 by iteration 1500 — phase-aligned with the pin
# std curriculum below.
W_PIN_DISPLACEMENT_START = -0.5
W_PIN_DISPLACEMENT = -2.0
W_PIN_SLIP = -0.5
W_RADIUS = -1.5
W_TILT = -0.5
W_STALL = -2.0
W_COUNTER_YAW = -1.0
# The pin-broken cost. Its real teeth are the progress gate in `_pv_update`, not
# this number, so the weight follows the same discovery-cheap ramp as the other
# two pin pressures: a first clumsy attempt that scuffs the pin costs -1.0/step
# while the turn is being discovered, -3.0/step once it exists. Phase-aligned
# with `pin_std` and `pin_displacement_weight` — tightening one pin pressure
# while the others are slack is just a tax on attempting.
W_PIN_BROKEN_START = -1.0
W_PIN_BROKEN = -3.0
# Both feet planted while the trunk yaws hard: the shuffle-spin, charged
# directly. Flat, not curriculum'd — it is the one pressure that must be present
# from step one, because the shuffle basin is what the policy falls into FIRST.
W_TWO_FEET_STILL = -2.0
W_ACTION_RATE = -0.05

# Curriculum boundary: the pin is loose until the turn exists, tight after.
PIN_TIGHTEN_ITER = 1500


def make_microduck_pivot_env_cfg(
    play: bool = False,
    rough: bool = False,
    direction: float | None = PIVOT_DIRECTION,
) -> ManagerBasedRlEnvCfg:
    """The pivot env.

    `direction=None` samples ±1 per env at reset (one network, both ways);
    `direction=±1` pins the sign for a one-way task. Either way the sign is
    written into the twist yaw slot, so the policy READS it — a pinned task is
    the bidirectional one restricted to half its command distribution, not a
    different observation contract.
    """
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Command: linear slots pinned near zero (obs-shape parity, neurons kept
    # alive), yaw slot carries the ±1 DIRECTION FLAG for the whole episode ────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-1.0, 1.0)   # the flag's two values
    if hasattr(command, "rel_turn_in_place_envs"):
        command.rel_turn_in_place_envs = 0.0
    fields = {f.name for f in microduck_mdp.PivotCommandCfg.__dataclass_fields__.values()}
    params = {k: v for k, v in vars(command).items() if k in fields}
    params.pop("direction", None)
    cfg.commands["twist"] = microduck_mdp.PivotCommandCfg(
        direction=direction, **params
    )

    # The standing-fraction curriculum drives a command we no longer vary.
    if "standing_envs" in cfg.curriculum:
        del cfg.curriculum["standing_envs"]

    # ── Rewards: drop the gait terms and everything that fights a pivot ──────
    # air_time / foot_clearance / foot_swing_height: gait shaping, and — unlike
    #   in the other pinned-command skills — NOT inert here, since they gate on
    #   command magnitude and this command's yaw slot is ±1. They would have
    #   demanded the PIN foot swing.
    # foot_slip: re-added below for the PIN foot only. The paddle scuffs.
    # angular_momentum: 3D norm, fights the trick head-on.
    # track_*_velocity: the trunk must orbit, and the yaw slot is a flag, not
    #   a rate (see the module docstring).
    for name in (
        "air_time", "foot_clearance", "foot_swing_height", "foot_slip", "pose",
        "angular_momentum", "track_angular_velocity", "track_linear_velocity",
    ):
        if name in cfg.rewards:
            del cfg.rewards[name]

    sensor_name = "feet_ground_contact"

    def shared() -> dict:
        # A FRESH SceneEntityCfg per term: the managers resolve these in place
        # (site_names → site_ids), and a module-level instance shared across
        # terms — and across the train / play / backlash copies of this cfg —
        # would be resolved against whichever scene got there first.
        return {
            "sensor_name": sensor_name,
            "feet_cfg": SceneEntityCfg(
                "robot", site_names=("left_foot", "right_foot")
            ),
            "command_name": "twist",
            "direction": direction,
            "target_yaw": PV_TARGET_YAW,
        }

    cfg.rewards["pivot_progress"] = RewardTermCfg(
        func=microduck_mdp.pv_progress_reward,
        weight=W_PROGRESS,
        params=shared(),
    )
    cfg.rewards["pivot_complete"] = RewardTermCfg(
        func=microduck_mdp.pv_complete_bonus,
        weight=W_COMPLETE,
        params=shared(),
    )
    cfg.rewards["pin"] = RewardTermCfg(
        func=microduck_mdp.pv_pin_reward,
        weight=W_PIN,
        # Starts at a whole footprint of slack; `pin_std` curriculum tightens
        # it to the measured PV_PIN_STD_M by PIN_TIGHTEN_ITER. The pay is
        # multiplied by this step's progress inside the function, so a still
        # robot collects nothing however loose the std is.
        params={**shared(), "pin_std": microduck_mdp.PV_PIN_STD_START_M},
    )
    cfg.rewards["paddle"] = RewardTermCfg(
        func=microduck_mdp.pv_paddle_reward,
        weight=W_PADDLE,
        params={
            **shared(),
            "min_air": microduck_mdp.PV_MIN_AIR_S,
            "max_air": microduck_mdp.PV_MAX_AIR_S,
        },
    )
    # v3: air time says the foot LEFT the floor, not that it went anywhere. The
    # v2 shuffle lifted a foot and set it back down where it was. This scores
    # the contact point's travel between consecutive plants, one shot per plant.
    cfg.rewards["paddle_reach"] = RewardTermCfg(
        func=microduck_mdp.pv_paddle_reach_reward,
        weight=W_PADDLE_REACH,
        params={
            **shared(),
            "reach_min": microduck_mdp.PV_REACH_MIN_M,
            "reach_max": microduck_mdp.PV_REACH_MAX_M,
            "reach_std": microduck_mdp.PV_REACH_STD_M,
        },
    )
    cfg.rewards["settle"] = RewardTermCfg(
        func=microduck_mdp.pv_settle_reward,
        weight=W_SETTLE,
        params={
            **shared(),
            # rate_std 1.0 rad/s: "stopped" for a robot that was just doing
            # ~2π rad/s, wide enough that the first policy to finish a turn
            # scores visibly (invisible gradients change nothing).
            "rate_std": 1.0,
            # pose_std 0.4: the legs come back near HOME, loosely — recovering
            # from a pivot is a big transient and must stay affordable.
            "pose_std": 0.4,
            "joint_indices": _LEG_JOINTS,
        },
    )
    # Trunk at standing height throughout: a pivot, not a squat-and-shuffle.
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=W_HEIGHT,
        params={
            "std": 0.04,
            "target_height": STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["pin_displacement"] = RewardTermCfg(
        func=microduck_mdp.pv_pin_displacement_penalty,
        weight=W_PIN_DISPLACEMENT_START,
        params={**shared(), "saturate_m": microduck_mdp.PV_PIN_SAT_M},
    )
    # v3: THE term that stops the pin being for sale. Broken = off the floor for
    # more than 40 ms or more than 3 cm from its anchor; charged every step until
    # re-planted within 2 cm of that anchor. Its real teeth are in `_pv_update`:
    # while it is set the progress potential does not advance at all, so a turn
    # taken off the pin earns nothing and the stall timer runs.
    cfg.rewards["pin_broken"] = RewardTermCfg(
        func=microduck_mdp.pv_pin_broken_penalty,
        weight=W_PIN_BROKEN_START,
        params=shared(),
    )
    # v3: the shuffle-spin named directly — both feet planted while the trunk
    # yaws past 0.5 rad/s. Charges a genuine pivot's double-support push too
    # (~a third of the cadence = -0.67/step), which is intended: it pushes the
    # cadence toward more air and shorter plants.
    cfg.rewards["two_feet_still"] = RewardTermCfg(
        func=microduck_mdp.pv_two_feet_still_penalty,
        weight=W_TWO_FEET_STILL,
        params={
            **shared(),
            "yaw_gate": microduck_mdp.PV_TWOFOOT_YAW_GATE,
            "yaw_cap": microduck_mdp.PV_TWOFOOT_YAW_CAP,
        },
    )
    cfg.rewards["pin_slip"] = RewardTermCfg(
        func=microduck_mdp.pv_pin_slip_penalty,
        weight=W_PIN_SLIP,
        params={**shared(), "saturate_mps": microduck_mdp.PV_PIN_SLIP_SAT},
    )
    cfg.rewards["pivot_radius"] = RewardTermCfg(
        func=microduck_mdp.pv_radius_penalty,
        weight=W_RADIUS,
        params={
            **shared(),
            "radius_m": microduck_mdp.PV_RADIUS_M,
            "tol_m": microduck_mdp.PV_RADIUS_TOL_M,
            "saturate_m": microduck_mdp.PV_RADIUS_SAT_M,
        },
    )
    cfg.rewards["tilt"] = RewardTermCfg(
        func=microduck_mdp.pv_tilt_penalty,
        weight=W_TILT,
        params=shared(),
    )
    # Standing still must LOSE, not merely fail to win: the pin no longer pays
    # a motionless robot, and this charges it. Bounded, spin-phase only, timer
    # resets the instant the potential moves.
    cfg.rewards["stall"] = RewardTermCfg(
        func=microduck_mdp.pv_stall_penalty,
        weight=W_STALL,
        params={
            **shared(),
            "grace_s": microduck_mdp.PV_STALL_GRACE_S,
            "ramp_s": microduck_mdp.PV_STALL_RAMP_S,
        },
    )
    # THE SIGN FIX. The ±1 flag in the twist yaw slot used to be one-sided —
    # the commanded direction paid, the opposite direction was free — so the
    # only direction-dependent pressure on a policy that had not found the turn
    # came from the pin/paddle loading, which yaws the trunk the WRONG way.
    # Charging the wrong-way rate makes obs[50] mean something from step one.
    cfg.rewards["counter_yaw"] = RewardTermCfg(
        func=microduck_mdp.pv_counter_yaw_penalty,
        weight=W_COUNTER_YAW,
        params={**shared(), "saturate_rate": microduck_mdp.PV_COUNTER_CAP},
    )

    # ── Discovery curricula: the pin is LOOSE until the turn exists ─────────
    # AGENTS.md's proven split — `reward_weight` for weights, a dedicated params
    # curriculum for everything else — and both ramps land on the same
    # iterations so the pay and the cost tighten together (a phase-misaligned
    # pair would tighten the cost while the pay was still slack, which is just a
    # tax on attempting).
    cfg.curriculum["pin_std"] = CurriculumTermCfg(
        func=microduck_mdp.pv_pin_std_curriculum,
        params={
            "reward_name": "pin",
            "std_stages": [
                {"step": 0, "std": microduck_mdp.PV_PIN_STD_START_M},   # 4.0 cm
                {"step": 600 * 24, "std": 0.030},
                {"step": 1000 * 24, "std": 0.022},
                {"step": PIN_TIGHTEN_ITER * 24, "std": microduck_mdp.PV_PIN_STD_M},
            ],
        },
    )
    cfg.curriculum["pin_displacement_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "pin_displacement",
            "weight_stages": [
                {"step": 0, "weight": W_PIN_DISPLACEMENT_START},
                {"step": 600 * 24, "weight": -1.0},
                {"step": 1000 * 24, "weight": -1.5},
                {"step": PIN_TIGHTEN_ITER * 24, "weight": W_PIN_DISPLACEMENT},
            ],
        },
    )
    # The third pin pressure, on the SAME iterations. The progress gate that
    # goes with it is NOT curriculum'd — it is on from step one, because a
    # policy that can earn progress off the pin will learn to earn progress off
    # the pin. Only the cost of scuffing is made cheap while the turn is being
    # discovered.
    cfg.curriculum["pin_broken_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "pin_broken",
            "weight_stages": [
                {"step": 0, "weight": W_PIN_BROKEN_START},
                {"step": 600 * 24, "weight": -1.5},
                {"step": 1000 * 24, "weight": -2.0},
                {"step": PIN_TIGHTEN_ITER * 24, "weight": W_PIN_BROKEN},
            ],
        },
    )

    # Action smoothness stays LIGHT: the velocity recipe ramps this to -1.0 by
    # iter 1500, which is a motion-blocker for a whole-body turn, and an
    # attempt-tax active during skill discovery makes "do nothing" win
    # (AGENTS.md). Introduced only after the skill should exist.
    cfg.rewards["action_rate_l2"].weight = W_ACTION_RATE
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.05},
                {"step": 1000 * 24, "weight": -0.10},
                {"step": 1800 * 24, "weight": -0.20},
            ],
        },
    )
    return cfg


MicroduckPivotRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="pivot",
    run_name="pivot",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=2_500,
)

assert math.isclose(PV_TARGET_YAW, 2.0 * math.pi)
