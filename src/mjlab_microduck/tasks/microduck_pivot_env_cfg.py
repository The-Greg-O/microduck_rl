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
  * `alive` (v5) is a CONSTANT +3.0 paid every step the episode is alive with
    the trunk within 15° of vertical. It is the term that makes this version
    different in kind rather than in degree: 200 alive steps = 600, larger than
    the whole progress budget of a perfect turn (400), so BY CONSTRUCTION no
    turn can pay for the episode it ends. It has no shape, no optimum and
    nothing to farm — which is exactly why it can be this large: it changes the
    value of surviving, not the ranking of anything done while alive. v4 made
    the same argument from `upright` + `height_stand` = 3.0/step and lost it on
    the numbers, against a progress term paying up to 6.0/step.
  * `pivot_progress` pays Δ(accumulated yaw) — potential-based, so holding
    still pays zero and nothing can be farmed; capped per step (no pay for
    violence past 0.5 turns/s — v5, halved again from v4's 0.75, which the
    policy read as a target) and capped in total at one turn. The potential only
    advances within 15° of vertical, so degrees turned on the way to the floor
    are not progress.
  * `pivot_complete` is ONE step, not a per-step "you are done" — that would be
    the jackpot the playbook warns about and would buy a ballistic whip. Speed
    is bought by the SETTLE phase instead: finish early and more of the 4 s is
    spent collecting settle pay, which only pays while upright, stopped, on
    both feet and back in the STAND pose. v5 raises both (+5 → +15 and
    +4 → +6) and re-gates both from the 40° fallen line to 15°: the money moves
    from turning FAST to FINISHING UPRIGHT, which is the substitution the v4
    log demanded (at iteration 3999 `pivot_complete` was 0.0014 and `settle`
    0.0009 — the policy had stopped finishing anything at all).
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
  * `terminated` (v4, re-sized in v5 to -100) charges the fall itself, one shot.
    v3 pivoted — the foot roles were finally right — and then fell over in every
    single episode (mean length 43 of 200, `fell_over` the only termination that
    ever fired). Nothing in v3 priced that: a fall ends the episode, and ending
    the episode also ends the stall bill a standing robot pays for the full 4 s,
    so dying at 0.6 s was worth ~+300 against standing still. Most of the swing
    is the FORFEITED income, not this penalty — see the arithmetic below — but a
    terminal state that merely stops the meter is an exit, and v3 proved the
    policy will take it. v4 set this at -20 and the arithmetic said that was
    enough; the log said otherwise (-0.09/step averaged, against progress at
    +0.90/step, while falls per iteration went 1.6 → 18 → 58). v5 raises it to
    -100 AND raises the income it forfeits (`alive`), because the mechanism was
    always right and only the sizing was wrong. It is still a tenth of the real
    price and still one step in 200 — a marker, not a spike.
  * `PV_PROGRESS_TILT_DEG` (v4) is the term that actually removes the faller's
    money. The potential only advances within 15° of vertical; the 40° gate the
    other terms use is the FALLEN gate, and by 40° the fall is already lost. v3
    collected ~0.88 of the per-step progress cap all the way down. Safe for the
    same reason the v3 pin gate is: the potential is a running maximum and the
    raw integral is floored, so a gate can only stop new degrees accruing — a
    robot that leans hard, recovers and carries on keeps everything it earned.
  * `tilt` (-0.5 → -2.0) and `body_ang_vel` (-0.05 → -0.2, x/y only) are v4's
    two lean/rock costs: the precursor to the fall, priced before it becomes
    one. `1 - cos(tilt)` is gentle near vertical (0.03 at 15°), so -2.0 is still
    under -0.07/step for a working lean and only bites once the robot is going
    over. Neither touches the yaw axis, which is the trick.
  * DISCOVERY IS MADE CHEAP AND THE CURRICULUM STOPS EARLY (v5). The pin
    reward's std starts at 4 cm (a whole footprint of slack) with the
    displacement cost at -0.5, and both tighten on phase-aligned stages to
    2.5 cm / -1.5 by iteration 1600 — and then HOLD for the remaining 2400
    iterations. v4 ran the same ramp to 1.5 cm / -2.0 at iteration 2400, and
    that window is exactly where its falls went 1.6 → 18 → 58 per iteration
    while episode length went 191 → 133 → 74. A curriculum is meant to
    consolidate a skill the policy has; that one was squeezing a policy that
    could barely hold a pivot into a tolerance it could not hold at all. The
    paddle still opens at +3.0 so lifting the free foot pays from step one.
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
# THE ARITHMETIC (dt = 0.02, 200 steps, v5 rate cap = 0.5 turns/s, so a
# 0.5 turns/s pivot scores `rate` = 1.0 on every rate-scaled term and completes
# the turn in 2.0 s — exactly half the episode, leaving the other half to
# settle).
#
# The lesson arc, one defect per version:
#   v1  gave "standing" the full pin pay → it stood there. Gating the pin on
#       progress fixed it and v2 duly TURNED.
#   v2  priced the pin GRADEDLY, so the pin was for sale: +421°/-365°, no falls,
#       but a TWO-FOOTED SHUFFLE-SPIN (contact 0.91 / 0.85, 6–17 cm of drift).
#       v3 made the pin a REQUIREMENT — `_pv_broken` freezes the potential.
#   v3  got the FOOT ROLES RIGHT and FELL OVER IN 4 EPISODES OUT OF 4 (mean
#       episode length 43 of 200, `fell_over` the only termination that ever
#       fired). It never priced FALLING at all.
#   v4  priced falling — and priced it too cheaply. THE CURVE WENT THE WRONG
#       WAY, monotonically, and the log is unambiguous about the direction:
#
#         iter 1000  length 191 of 200, falls 1.6/iter, progress 0.59/step
#         iter 2000  length 133,        falls  18/iter
#         iter 3000  length  74,        falls  58/iter, progress 0.82
#         iter 3999  length  77,        falls  47/iter, progress 0.90,
#                    `terminated` -0.09/step, `pivot_complete` 0.0014,
#                    `settle` 0.0009
#
#       Progress ROSE the whole way while episode length collapsed: the policy
#       was not failing to learn, it was learning that a turn which falls at
#       ~0.9 s out-earns a careful one. The lab eval says the SKILL is there
#       (BAM, noise + DR, 3 seeds per direction: 2 of 6 episodes complete a full
#       turn on one foot without falling; pin contact 0.76–0.97, free foot
#       0.31–0.49, pin drift 2–6 cm) and that the REWARD makes falling
#       profitable past ~iteration 1500.
#
#       v4's own diagnosis was right in kind and wrong in size. It said a fall
#       is paid for mostly in FORFEITED INCOME rather than in its one-shot
#       penalty — true — and then made that income `upright` 2.0 +
#       `height_stand` 1.0 = 3.0/step, HALF the 6.0/step the progress term paid
#       the thing it was supposed to outweigh. -20 averaged to -0.09/step
#       against progress at 0.90/step. Two of v4's own terms were also actively
#       pushing: a 0.75 turns/s pay cap the policy read as a target, and a pin
#       curriculum still tightening toward 1.5 cm / -2.0 across iterations
#       1600–2400 — precisely the window where falls went 1.6 → 18 → 58.
#
# v5 MAKES STAYING UP WORTH MORE THAN ANY TURN, BY CONSTRUCTION:
#   a. `alive` +3.0/step, a CONSTANT paid every step the episode is alive with
#      the trunk within 15° of vertical. 200 alive steps = 600 > 400, the entire
#      progress budget of a perfect turn (4.0 × 100 steps). Nothing to farm — no
#      shaping, no pose to seek — so it can be this large without distorting the
#      ranking of anything the policy does WHILE alive; it changes only the
#      value of still being alive to do it.
#   b. `terminated` -100 (was -20), sized against the term it competes with.
#      The fall's real price is still the forfeit (see below) — this is a marker
#      that is now visible rather than one that averages to noise.
#   c. rate cap 0.75 → 0.5 turns/s and progress 6.0 → 4.0: a 2 s pivot, not a
#      1.3 s one, and a progress budget that cannot out-earn survival.
#   d. the pin curriculum stops at 2.5 cm / -1.5 and gets there by iteration
#      1600, then HOLDS for the remaining 2400 iterations.
#   e. the money moves to the FINISH: `pivot_complete` +5 → +15 and `settle`
#      +4 → +6, both re-gated from the 40° fallen line to 15°, so finishing
#      upright is the jackpot rather than turning fast.
#   f. everything else is v4's: upright-gated progress, tilt -2.0, body_ang_vel
#      -0.2, counter-yaw, stall, two_feet_still, the paddle terms.
#
# Both tables below are MEASURED — `test_the_arithmetic_pivoting_wins` and
# `test_the_shuffle_loses_per_step_during_the_spin_phase` run all FOUR
# strategies through the real reward functions at these weights and produce
# exactly these numbers.
#
#   pivot     — the textbook, SLOW: 0.5 turns/s (full pay), a 2 s turn, pin
#               planted, free foot on a 0.24 s up / 0.12 s down cadence carrying
#               its contact point 6 cm per stroke, trunk upright — then ~2 s of
#               settle.
#   still     — the v1 argmax: nothing moves.
#   shuffle   — WHAT v2 LEARNED: both feet down and sliding, trunk at 1.5 rad/s.
#   fastfall  — WHAT v4 LEARNED, given every benefit of the doubt: a 2 turns/s
#               whip with the pin held perfectly planted and a textbook paddle
#               cadence, the trunk tipping to 90° over 0.9 s and `fell_over`
#               firing there. 0.9 s is v4's OWN measured mean episode length at
#               iteration 3999 (77 steps of 200), not a strawman. The episode
#               ENDS at the fall.
#
# (1) Per step of the SPIN phase (or, for the faller, up to the fall):
#
#                                    still   pivot   shuffle  fastfall
#   progress      4.0 × rate          0.00   +4.00    +0.14    +0.62
#   pivot_complete 15.0, one shot     0.00   +0.15     0.00     0.00
#   pin           3.0 × stay × rate   0.00   +3.00    +0.07    +0.47
#   paddle        3.0, in-window      0.00   +1.26     0.00     0.00
#   paddle_reach  3.0, one shot/plant 0.00   +0.15     0.00    +0.07
#   alive         3.0, upright ≤ 15°  +3.00  +3.00    +3.00    +0.47
#   pin_displacement (-0.5 → -1.5)    0.00    0.00    -1.40     0.00
#   pin_slip         -0.5             0.00    0.00    -0.50     0.00
#   pin_broken       (-1.0 → -3.0)    0.00    0.00    -2.77     0.00
#   two_feet_still   -2.0             0.00   -0.60    -0.80    -0.53
#   stall            -2.0            -1.62    0.00    -1.48    -0.06
#   counter_yaw      -1.0             0.00    0.00     0.00     0.00
#   tilt             -2.0             0.00    0.00     0.00    -0.75
#   terminated      -100.0, one shot  0.00    0.00     0.00    -2.22
#   common (upright 2.0 + height 1.0) +3.00  +3.00    +3.00    +1.33
#                                   ──────  ──────   ───────  ────────
#   mean/step                        +4.38  +13.96    -0.73    -0.61
#   steps it lasts                     200     100      200        45
#
# Note the still column. v4 wanted standing still to be NEGATIVE and got its
# wish (-1.38/step at the equivalent row), which is exactly the shape that makes
# ending the episode an ESCAPE — the stall bill stops the moment the episode
# does. Under v5 standing still is in profit at +4.38/step and pivoting is still
# worth three times that.
#
# (2) Whole 4 s episodes, end to end — and THIS is where the fall loses:
#
#                     pivot     still    fastfall    shuffle
#   progress         +400.0       0.0       +28.0      +28.6
#   pin              +300.0       0.0       +21.0      +15.0
#   paddle           +126.0       0.0         0.0        0.0
#   paddle_reach      +15.0       0.0        +3.0        0.0
#   pivot_complete    +15.0       0.0         0.0        0.0
#   settle           +600.0       0.0         0.0        0.0
#   alive            +600.0    +600.0       +21.0     +600.0
#   pin_displacement     0.0      0.0         0.0     -279.3
#   pin_slip             0.0      0.0         0.0      -99.5
#   pin_broken           0.0      0.0         0.0     -555.0
#   two_feet_still     -60.0      0.0       -24.0     -160.0
#   stall                0.0   -324.0        -2.9     -296.0
#   tilt                 0.0      0.0       -33.7        0.0
#   terminated           0.0      0.0      -100.0        0.0
#   common            +600.0   +600.0       +60.0     +600.0
#                    ───────  ───────    ────────   ────────
#   TOTAL             +2596     +876       -27.6     -146.1
#   steps                200      200          45        200
#
#   ORDER: pivot > still > fastfall > shuffle, and still − fastfall = +904 —
#   the margin the fix asks for (≥ 500) and nine times the -100 termination
#   marker, so the ranking cannot be flipped back by tuning that penalty alone.
#
# THE DECISIVE LINE IS `alive`. v4 correctly identified that a fall is paid for
# in forfeited income and then under-funded the income: its whole forfeitable
# stack was the +3.0/step `common` row, so a 0.9 s fall gave up ~465 against a
# progress term paying up to 6.0/step, and the trade was worth taking. v5's
# faller gives up 579 of `alive` on top of that — 984 in total, against the 28
# of progress and 21 of pin it banks before the 15° gate closes. The -100 marker
# is a tenth of the price, deliberately: a -900 spike on one terminal step is
# the jackpot AGENTS.md warns about, in reverse.
#
# What removes the faller's income at the top of the stack is still v4's 15°
# gate on the progress potential (`PV_PROGRESS_TILT_DEG`): the whip banks 7
# steps of progress before it passes 15° of lean and then nothing, so it never
# completes and never reaches the settle. v5 extends the same 15° cone to
# `alive`, `pivot_complete` and `settle`, so every large positive in the stack
# now means the same thing by "upright".
#
# Settle phase (after the one-shot completion): progress / pin / paddle /
# paddle_reach / stall / pin displacement / pin_broken / two_feet_still are all
# gated off by `_pv_done`, and settle 6.0 takes over → +6.0/step on top of the
# +3.0 alive and +3.0 common. Finishing early still wins: every step saved from
# the turn is a step of settle.
#
# The ranking holds at BOTH ends of the pin curriculum (loose, iteration 0:
# +2596 / +876 / -27.6 / +410) — the pin COSTS are made cheap for discovery, but
# the progress GATES (pin and upright) never are, and `alive` is never
# curriculum'd at all.

W_PROGRESS = 4.0
# v5: +5 -> +15, and re-gated from the 40 deg fallen line to 15 deg. The whole
# v5 substitution in one line: what the stack pays for is FINISHING UPRIGHT, not
# turning fast. A completion bonus paid at 39 deg of lean is a bonus for
# arriving in a state the robot is about to fall out of.
W_COMPLETE = 15.0
W_PIN = 3.0
# v5: +4 -> +6, same 15 deg re-gate. The settle is where a finished pivot lives
# for the back half of the episode, so this is the term a 2 s turn is actually
# buying: 100 steps of it at +6 is +600.
W_SETTLE = 6.0
W_PADDLE = 3.0
# THE v5 TERM. A constant paid every step the episode is alive with the trunk
# within 15 deg of vertical — no shaping, no pose to seek, nothing to farm.
# Sized so that surviving the episode (200 x 3.0 = 600) is worth more than the
# ENTIRE progress budget of a perfect turn (4.0 x 100 = 400): by construction,
# no turn can pay for the episode it ends. v4 made this same argument from
# `upright` + `height_stand` = 3.0/step and lost it on the numbers — that stack
# was half the size of the progress term it was competing with.
W_ALIVE = 3.0
# The reach bonus matches the paddle's weight but fires ONCE per plant, so on a
# 0.3 s cadence it is worth ~+0.2/step: a directional nudge toward strokes that
# actually travel 4–8 cm, not a term that can out-shout progress.
W_PADDLE_REACH = 3.0
W_HEIGHT = 1.0
# Opens loose (-0.5) so the first clumsy attempt at lifting the free foot is
# affordable, tightened to -1.5 by iteration 1600 and then held — phase-aligned
# with the pin std curriculum below.
W_PIN_DISPLACEMENT_START = -0.5
# v5: the ramp STOPS at -1.5 (never -2.0) and gets there by iteration 1600, then
# holds for the remaining 60% of the run. See PIN_TIGHTEN_ITER.
W_PIN_DISPLACEMENT = -1.5
W_PIN_SLIP = -0.5
W_RADIUS = -1.5
# v4: -0.5 → -2.0. `1 - cos(tilt)` is a gentle curve near vertical (0.03 at 15°,
# 0.13 at 30°), so at -0.5 the lean that precedes a fall cost almost nothing —
# v3's whole `tilt` line came to -0.0084 per episode. At -2.0 it is still under
# -0.07/step at 15° (nothing a lean-and-recover has to fear) and -2.0/step once
# the robot is on its side.
W_TILT = -2.0
W_STALL = -2.0
W_COUNTER_YAW = -1.0
# v4: the terminal state charged, one shot, exactly as the bow's v4 does — same
# weight, same reasoning. A fall must be the WORST outcome available, not a way
# to stop paying. The forfeited income is the larger half of the price (see the
# arithmetic above), but forfeiting is not being charged, and v3 proved a
# terminal state that merely stops the meter is an exit the policy will take.
# v5: -20 -> -100. v4's reasoning ("the forfeited income is the real price, this
# is only a marker") was right about the mechanism and wrong about the size: at
# iteration 3999 the v4 log reports `terminated` averaging -0.09/step against
# `pivot_progress` at +0.90/step, and the policy priced that correctly — falls
# per iteration 1.6 -> 18 -> 58 while progress ROSE. -100 is one step in 200 and
# is still the smaller half of the price next to the ~570 of alive + common
# income a 0.9 s fall throws away; it is simply visible now.
W_TERMINATED = -100.0
# v4: rocking on x/y is the precursor to the fall. The velocity recipe leaves
# this at -0.05, which for a task with a ~+10/step positive stack is noise; -0.2
# is still an order of magnitude below the task terms (AGENTS.md: motion-
# blockers stay LOW for dynamic tasks) and it touches only roll/pitch rate — the
# yaw axis, which is the trick, is untouched.
W_BODY_ANG_VEL = -0.2
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

# Curriculum boundary: the pin is loose until the turn exists, tighter after —
# and, from v5, IT STOPS THERE.
#
# v4 ran this ramp to iteration 2400 (60% of the run) and to the measured
# 1.5 cm / -2.0. The log says what that bought: falls per iteration 1.6 (iter
# 1000) -> 18 (2000) -> 58 (3000) and mean episode length 191 -> 133 -> 74,
# i.e. the collapse happened inside the tightening window and tracked it. A
# curriculum is supposed to consolidate a skill the policy already has; this one
# was pushing a policy that could just barely hold a pivot into a pin tolerance
# it could not hold at all, at the same time as the 0.75 turns/s pay cap was
# pulling it faster. v5 lands the pressures EARLIER (1600, 40% of the run) and
# LOWER (2.5 cm / -1.5), then holds them flat for the remaining 2400 iterations
# so the back half of the run is spent consolidating rather than chasing.
PIN_TIGHTEN_ITER = 1600


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
    # v5: the jackpot moved here from the rate cap. +15 (was +5), and paid only
    # for a turn finished with the trunk within 15 deg of vertical — the same
    # cone the progress potential advances in, not the 40 deg fallen line.
    cfg.rewards["pivot_complete"] = RewardTermCfg(
        func=microduck_mdp.pv_complete_bonus,
        weight=W_COMPLETE,
        params={**shared(),
                "gate_tilt_above_deg": microduck_mdp.PV_FINISH_TILT_DEG},
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
            # v5: 15 deg, as for the completion bonus. Settling is a state the
            # robot has to be OVER ITS FEET in; paid at 39 deg of lean it would
            # be paying a crouch that is on its way down.
            "gate_tilt_above_deg": microduck_mdp.PV_FINISH_TILT_DEG,
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
    # THE v4 TERM. v3 pivoted and fell over in 4 evaluation episodes out of 4,
    # and the log says it did so every episode of the run: mean episode length
    # 43 of 200, `fell_over` the only termination that ever fired. Under v3's
    # own arithmetic falling was not merely unpunished, it was the best exit
    # available to a policy that could not finish the turn — the stall bill a
    # standing robot pays for the full 4 s stops the moment the episode does.
    # Only `fell_over` is named: this cfg adds no behavioural termination of its
    # own, and a NaN guard is a sim blow-up, not something the policy chose.
    cfg.rewards["terminated"] = RewardTermCfg(
        func=microduck_mdp.pv_fall_penalty,
        weight=W_TERMINATED,
        params={"term_names": ("fell_over",)},
    )
    # THE v5 TERM, and the other half of the same fix. v4 charged the fall and
    # argued the real price was the FORFEITED income — correctly — but the
    # income it forfeited (upright 2.0 + height 1.0) was half the size of the
    # progress term the fall was buying. This is the forfeitable income, made
    # big enough to decide the question: +3.0 every step the episode is alive
    # and the trunk is within 15 deg of vertical, so 200 alive steps = 600 >
    # 400, the entire progress budget of a perfect turn. It is a CONSTANT — no
    # shaping, no pose to seek, nothing to farm — which is why it can be this
    # large without distorting anything: it changes the value of SURVIVING, not
    # the ranking of anything the policy does while alive.
    cfg.rewards["alive"] = RewardTermCfg(
        func=microduck_mdp.pv_alive_reward,
        weight=W_ALIVE,
        params={"gate_tilt_above_deg": microduck_mdp.PV_ALIVE_TILT_DEG},
    )
    # v4: rocking on roll/pitch is what a fall starts as. The velocity recipe's
    # -0.05 is noise against this task's ~+10/step positive stack; -0.2 is felt
    # and is still far below the task terms. x/y only — the yaw axis IS the
    # trick and this term never touches it.
    cfg.rewards["body_ang_vel"].weight = W_BODY_ANG_VEL

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
                {"step": 800 * 24, "std": 0.032},
                # v5 FLOOR: 2.5 cm, not the measured 1.5 cm, and reached at
                # iteration 1600 rather than 2400. Still well under a foot
                # width, so it is still a pin and not a step.
                {"step": PIN_TIGHTEN_ITER * 24,
                 "std": microduck_mdp.PV_PIN_STD_FLOOR_M},
            ],
        },
    )
    cfg.curriculum["pin_displacement_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "pin_displacement",
            "weight_stages": [
                {"step": 0, "weight": W_PIN_DISPLACEMENT_START},
                {"step": 800 * 24, "weight": -1.0},
                {"step": PIN_TIGHTEN_ITER * 24, "weight": W_PIN_DISPLACEMENT},
            ],
        },
    )
    # The third pin pressure, on the SAME iterations. The progress gate that
    # goes with it is NOT curriculum'd — it is on from step one, because a
    # policy that can earn progress off the pin will learn to earn progress off
    # the pin. Only the cost of scuffing is made cheap while the turn is being
    # discovered.
    #
    # v5 relaxes the two GRADED pin pressures (std, displacement) and does NOT
    # relax this one: it still ends at -3.0, just 800 iterations earlier. The
    # distinction is what each one charges. `pin_std` and `pin_displacement` are
    # continuous, and a genuine pivot pays them EVERY STEP for the few
    # millimetres a real planted foot inevitably travels — that is the pressure
    # that was squeezing a working policy, and it is the one that comes off.
    # `pin_broken` is a binary requirement (off the floor > 40 ms, or > 3 cm
    # from the anchor) that a correct pivot never trips at all; the only
    # strategy it charges is the two-footed shuffle. Relaxing it would buy the
    # shuffle back for nothing, and the arithmetic below says so directly — the
    # shuffle is the row that needs it to stay last.
    cfg.curriculum["pin_broken_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "pin_broken",
            "weight_stages": [
                {"step": 0, "weight": W_PIN_BROKEN_START},
                {"step": 800 * 24, "weight": -2.0},
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
                {"step": 1600 * 24, "weight": -0.10},
                {"step": 2880 * 24, "weight": -0.20},
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
    max_iterations=4_000,
)

assert math.isclose(PV_TARGET_YAW, 2.0 * math.pi)
