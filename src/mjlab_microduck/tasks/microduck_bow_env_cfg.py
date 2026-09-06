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
    is the heaviest PER-STEP cost in the stack.

── Reward design, in the playbook's terms ───────────────────────────────────
  * height / pitch / head track a slewed ramp (Gaussian): no jackpot, no
    waypoint camping, and the reward is zero-mean-safe at every phase.
  * rise_progress is the one dense gradient OUT of the crouch: potential-based
    Δ of a running-max trunk height, exactly like happy-spin's `hs_progress`.
    Every other term prices where the trunk IS; this one pays for the act of
    going up, so a rise abandoned half-way still banks half the money.
  * stand_pose is a TIME window (the stand phase), not a state gate — the
    policy cannot park in it early, and it is what makes the episode finish
    in a clean stand instead of leaving the robot folded.
  * upright prices LATERAL tilt only. The stock `upright` term prices total
    tilt and would fight the intended nose-down pitch, so it is dropped.
  * bow_head_still / head_joint_vel cover the OTHER two head joints. Between
    the four terms every neck/head servo is now priced by position, by speed,
    or by both — v4 priced two of the four and the policy spent the other two
    (see the v4 history below).
  * foot_lift is the anti-fall constraint that matters most: Microducks fall
    while RISING out of a crouch, and every one of those falls starts with a
    foot leaving the floor. Charged per foot, so a single lift already costs —
    but only once a foot has been airborne for 40 ms, so the heel flicker of a
    legitimate push-up is free.
  * risen: a one-shot milestone for getting back up, feet planted, having
    actually reached the crouch.
  * terminated: a flat cost on the two BEHAVIOURAL terminations. Ending the
    episode must never be the cheap way to stop paying (v3, below).
  * collapse termination: a trunk below BOW_COLLAPSE_Z is a collapse, not a
    bow — terminating stops the policy from farming the crouch reward's tail
    from a heap on the floor.

════════════════════════════════════════════════════════════════════════════
THE HISTORY. Four trained runs, four different failures, one lesson each.
════════════════════════════════════════════════════════════════════════════

── v1 PARKED IN THE CROUCH ─────────────────────────────────────────────────
Weights (3 height / 2 pitch / 2 head / 2 upright / 2 stand_pose / 0.5+0.5
track / -3 foot_lift / -1.5 drift / -0.2 action_rate), 2 cm height std.

The rendered rollout of checkpoint 1999 (BAM servos) got the whole first half
right: trunk 0.127 → 0.088 m by 1.3 s, 10° nose-down, head down to 0.202 m,
both feet planted, lateral tilt 10°, a clean hold. Then it simply stayed there
— 0.088 m from 1.3 s all the way to 4.0 s, straight through the rise and the
stand phase. Parking was cheap: the +2.0 stand-phase pose term and the height
ramp back up did not outweigh what rising RISKS (a foot-lift cost that fires
on the heel flicker every push-up produces, and a collapse termination that
forfeits the rest of the episode). Rising was the scary part, and nothing made
it the argmax. LESSON: v1 learned the dip. Whatever gets changed, do not break
the thing that worked.

── v2 PARKED HIGHER ────────────────────────────────────────────────────────
v2 raised the stand-phase stack (height 3→4, stand_pose 2→5), added the +5
one-shot `risen`, and added a `not_risen` ramp measured against a fixed
height. The rollout never bowed at all: 0.127 m spawn → 0.101 m by 0.5 s, then
frozen for 3.5 s. Pitch -4°, head 0.218 m, feet planted, no dip, no rise.
0.101 m is the COMPROMISE — close enough to the 0.085 crouch and to the 0.115
stand to collect partial credit from BOTH phases at a 2 cm std (0.53 and 0.61)
while never paying the action-rate cost of moving. The log agrees term by term
for a robot that did nothing: bow_height 0.71 of max, bow_pitch 0.70, bow_head
0.61, bow_upright 0.97, action_rate_l2 -0.71, `risen` fired in ~2% of
episodes, and `not_risen` logged -0.0003 — its dead band ended at 0.095 m and
the policy parked at 0.101, so the anti-parking term never touched the pose it
was written to stop. LESSON: a bigger prize for the goal state does not buy a
motion; and a tax on moving, while the motion is still being discovered, buys
stillness.

── v3 DIVERGED: IT LEARNED TO FALL OVER ────────────────────────────────────
v3 attacked the compromise pose directly. Tolerances went smaller than the
motions they measured (height std 6 mm, pitch 0.05, head 0.10); `phase_match`
paid +25 per phase for holding a 1 cm corridor around the ramp end-to-end;
`not_dipped` and `not_risen` charged -5.0 each for being on the wrong side of
the ramp; survival income (upright, track) was halved so it could not fund
parking; stand_pose became a multiplicative composite at 5.0.

On paper it worked: the arithmetic in the v3 docstring had a textbook bow at
2260 against 214 for the high park and 857 for the low park. In training it
DIVERGED. The final log:

    Episode_Termination/collapsed   31.8 per iteration
    Episode_Termination/time_out     4.6
    Mean episode length             97 of 200
    Mean action noise std           1.26
    Mean reward                     2.29

Terminating beat staying alive. A parked pose was charged about -3.5/step by
the two off-ramp costs against a thinned-out positive stack, and an episode
that ENDS collects nothing — so falling over was strictly the cheaper of the
two, and PPO found it in a few hundred iterations. The v3 docstring predicted
this failure by name ("WATCH FOR … the policy is buying its way out by falling
over") and set the tripwire at `Mean episode length` under 195; it ran at 97.

TWO LESSONS, and v4 is built on both:
  (i)  A COST THAT MAKES TERMINATION ATTRACTIVE IS NOT A FIX. Any anti-parking
       pressure that drives the per-step total near zero is competing with an
       exit that pays exactly zero. Price the RIGHT behaviour up; do not price
       the wrong one down past the value of quitting. And make quitting itself
       cost something.
  (ii) A SPARSE TARGET WITH NO GRADIENT ALONG THE PATH IS NOT A FIX EITHER.
       v3's corridor latch and its 6 mm Gaussians pay nothing at all until the
       trunk is already where it should be. A policy one centimetre into an
       unfinished rise collected the same as one that never tried. Narrowing a
       Gaussian below the error the CURRENT policy can hold does not sharpen
       the gradient, it deletes it (AGENTS.md: "≈ the error you still care
       about, not the max error" — and price only the ESCAPABLE part).

── v4: v1's recipe, plus a gradient up ─────────────────────────────────────
Start from v1 VERBATIM — it is the only version that learned the dip — and add
the smallest set of terms that make rising the argmax without making quitting
attractive. Nothing from v2 or v3 survives: no corridor credit, no
not_dipped/not_risen, no tightened tolerances, no multiplicative stand_pose,
and survival income back at v1's level.

  (a) `rise_progress` (+6.0), THE term. Potential-based, modelled on
      `hs_progress_reward`: over the rise window (2.0–3.2 s) it pays Δ of a
      running-max potential — the trunk height clamped to [0.085, 0.115] —
      normalised by a 0.06 m/s rate cap. The whole 3 cm is worth
      0.030/(0.06·0.02) = 25.0 credits = 150 points however fast it is
      climbed, a tenth of the climb is worth a tenth of that, and:
        · holding still pays 0 (Δ of a potential), so there is nothing to camp
          on and no version of "arrive early and farm it";
        · sinking and re-climbing pays nothing the second time (running max);
        · climbing past the standing height pays nothing (clamped high);
        · a robot that never crouched has no runway at all — the potential
          arms at whatever height it brings into the window — so standing tall
          for four seconds earns exactly zero;
        · payment is gated on both feet planted while the potential keeps
          updating, so a HOP forfeits the height it flies through instead of
          banking 150 points for four steps of flight.
  (b) `risen` (+5.0), one shot: the first step at/after 2 s with the trunk back
      within 1 cm of STAND_Z, both feet planted, the crouch actually reached.
      Kept from v2 — a milestone, not a jackpot.
  (c) `terminated` (-20.0) on the `collapsed` and `fell_over` terminations
      (mjlab computes terminations before rewards, and the term reads those two
      by name so a `nan_state` sim blow-up is not charged to the policy). This
      is lesson (i) as a single line of config.
  (d) `action_rate_l2` -0.05, fixed, no curriculum. v1 ran -0.2 and v2's log
      shows that costing a MOTIONLESS robot 0.71/step. Motion is the trick.
  (e) `stand_pose` +3.0, stand phase only, and ADDITIVE (v1's form). v3 made it
      a product with a height factor, which collapses to ~0 for a policy that
      finishes a centimetre short — deleting the gradient exactly where the
      last mile of the rise lives, which is lesson (ii) in miniature.
  (f) `foot_lift` -3.0 with v2's 40 ms flicker exemption, so the heel unweight
      of a legitimate push-up is not priced as a hop.

── v4 BOWED — AND NEARLY BROKE ITS OWN NECK ────────────────────────────────
v4 worked. The rendered rollout dips, holds, rises and finishes standing with
both feet planted — the trunk trajectory this file has been chasing since v1.
And the whole time, the HEAD ROTATES WILDLY: it swings through the dip, through
the hold and through the rise, hard enough to look like it is going to break
its own neck.

That is not a training accident, it is the reward stack. `bow_head` prices
servo joints (5, 6) — neck_pitch and head_pitch, the sagittal pair that carries
the head down and back up. Joints 7 and 8, head_yaw and head_roll, are named by
NOTHING in the v4 stack. head_yaw has ±2.97 rad of travel on robot_walk.xml and
head_roll ±0.44, on a head that is 38% of the body mass, and the only thing
charging any of that motion was `action_rate_l2` at -0.05.

An unpriced heavy joint is a free counterweight. Swinging the head shifts the
CoM, which buys exactly what `bow_upright` (+2.0), `bow_height` (+3.0) and
`rise_progress` (+6.0) are paying for, at a smoothness cost of a few hundredths
per step. PPO found the trade, and it should have: the stack asked for it.
LESSON: a joint you do not price is a joint the policy will spend. Enumerate
the whole actuator set when designing a pose reward, not only the joints that
carry the motion you had in mind.

── v5: NAME THE OTHER TWO JOINTS ───────────────────────────────────────────
v4 VERBATIM — every weight, every tolerance, every gate — plus two terms that
between them cover all four neck/head joints. Nothing else moves.

  (a) `bow_head_still` (+2.0), the whole episode: a tight Gaussian (std 0.08
      rad ≈ 4.6°) holding head_yaw and head_roll at HOME. HOME is 0.0 for both
      (microduck_constants.py), so this is literally "hold them at zero". No
      phase gate — there is no moment in a bow when the head is meant to be
      pointing sideways.

      Why a 0.08 std is legitimate here when v3's tight tolerances were the
      thing that broke it: v3 narrowed Gaussians below the tracking error a
      policy still learning the motion could hold, which deletes the gradient
      (AGENTS.md: price the ESCAPABLE part of an error). These two joints have
      no trajectory to track and no servo lag to fight. The target is the pose
      the robot already starts in and never has to leave, so ALL of the error
      is escapable and the std is simply the wobble worth tolerating.

      Weighted level with `bow_head` on purpose: where the head POINTS is worth
      as much as how low it is held.

  (b) `head_joint_vel` (-0.5), bounded, saturating at 4.0 rad/s: the mean of
      clamp((joint_vel / cap)², 0, 1) over ALL FOUR head joints. This is the
      half (a) cannot do — neck_pitch and head_pitch MUST move, they track the
      down/up ramp, so no position term can settle them. Pricing SPEED lets the
      ramp through and charges everything faster.

      The numbers: the ramp's steepest ask is the 0.60 rad of down→up travel
      across the 1 s rise, 0.6 rad/s — (0.6/4)² = 0.023 of the term on one
      joint, about 0.003/step after the weight and the mean over four joints. A
      head snapping at the cap costs 0.5/step, 150× more. Quadratic below the
      cap and FLAT above it, so it is a smoothness prior on the intended motion
      and a wall on the thrash.

      Bounded, and that is not decoration. An l2 on joint velocity has no
      ceiling and its worst case is set by the sim; a cost like that driving
      the per-step total toward zero is precisely the shape that taught v3 to
      fall over (lesson (i)). Here the whole term is worth at most 0.5/step
      against a still head's +2.0/step, so the v5 pair is NET POSITIVE income
      for a head that behaves and can never make quitting look good.

THE ARITHMETIC (`tests/test_bow_cfg.py::test_the_four_strategies` and
`::test_the_thrashing_head_now_loses`, real reward functions, all 200 steps,
sums of weighted values before mjlab's ×dt — divide by 200 for wandb's
`Episode_Reward` units). `thrash` is what v4 rendered: a textbook bow with the
head also swinging at 2 Hz through ±1.5 rad of yaw and ±0.4 of roll.

                   textbook     thrash  park@0.101  park@0.088  collapse@1.5s
      bow_height      600.0      600.0       431.9       400.1        222.6
      bow_pitch       400.0      400.0       288.8       269.7        150.0
      bow_head        400.0      400.0       227.8       247.7        150.0
      bow_head_still  400.0       30.9       400.0       400.0        150.0
      head_joint_vel   -0.3      -38.9         0.0        -0.1         -0.1
      bow_upright     400.0      400.0       400.0       400.0        150.0
      stand_pose      153.0      153.0        85.7        62.0          0.0
      rise_progress   150.0      150.0         0.0         0.0          0.0
      risen             5.0        5.0         0.0         0.0          0.0
      terminated        0.0        0.0         0.0         0.0        -20.0
      track           200.0      200.0       200.0       200.0         75.0
      ---------------------------------------------------------------------
      TOTAL          2707.7     2300.0      2034.2      1979.4        877.5
      per phase dip   587.9      486.2       534.6       587.9        587.9
               hold   600.0      498.1       407.0       600.0        289.6
               rise   751.7      649.8       534.5       443.3          0.0
               stand  768.0      665.8       558.1       348.1          0.0
      per step        13.54      11.50       10.17        9.90         4.39

Read the `thrash` column top to bottom: it is IDENTICAL to the textbook bow in
every v4 term, line for line. That is the bug, printed. The two v5 rows are the
only place the stack can tell them apart, and they are worth 2.04/step of
separation — 40× what `action_rate_l2` was charging the swing.

The v4 ordering survives untouched: textbook > park@0.101 > park@0.088 >
collapse, with collapse still under half the next-worst. Two things to note
about the numbers rather than the order:

  * `bow_head_still` pays every well-behaved strategy the same +400, parked
    ones included, so the RATIO of park to textbook rises (0.68 → 0.73) while
    the MARGIN is unchanged at 728 points / 3.64 per step. The margin is what
    a policy optimizes; the test asserts both, and says why.
  * The thrashing bow still beats parking (2300 > 1979). That is deliberate,
    and it is lesson (i) again — a bow with a bad head is a WORSE BOW, not a
    failure, and pricing it below "never bowed at all" would be exactly the
    inversion that made v3 quit. It loses 407 points to the clean bow, which is
    more than half the gap between a clean bow and one that never rises.

park@0.101 is v2's learned trajectory, park@0.088 is v1's (a textbook dip and
hold, then frozen in the crouch — credited in full for the half of the bow it
genuinely does), and collapse@1.5 s is a textbook bow that folds mid-hold and
terminates.

The ORDER is the point, and it is the first version to get all of it right:
textbook > park@0.101 > park@0.088 > collapse, with collapsing at less than
half of the next-worst. Note what is NOT claimed: collapse is +728, not
negative. An episode that ends simply stops earning, and no sane penalty makes
1.5 s of an honest bow worth less than nothing. The -20 is insurance; the
mechanism is the 125 steps of forfeited income (drop the -20 and collapse is
still last by 830 points).

Note also that the paper margin over the low park (1.46×) is THINNER than
v3's (2.6×). That is deliberate. v3 bought its margin with costs that made
quitting cheap, and the policy quit. What v4 has instead is a per-step
GRADIENT: from the parked crouch, a step of climbing is worth strictly more
than a step of staying, at every rate from a 0.005 m/s crawl to the 0.06 m/s
cap, and the best-paid rate is the one that tracks the ramp
(`test_rising_out_of_the_crouch_pays_at_every_millimetre`). v1's stack had no
such slope — that is why it sat at 0.088 m for 2.7 s.

WATCH FOR, in this order:
  1. `Episode_Termination/collapsed` per iteration, against `time_out`, and
     `Mean episode length` — the v3 tripwire. Below ~190 of 200 something is
     paying the policy to quit; W_TERMINATED goes more negative before
     anything else is touched.
  2. `Episode_Reward/rise_progress`. A textbook bow averages 150/200 = 0.75
     per step. If it stays under ~0.1 while `bow_height` is healthy, the
     policy is dipping and parking again (v1's failure) and the rise term is
     not reaching it — raise W_RISE_PROGRESS before adding any cost.
  3. `Episode_Reward/risen` — the fraction of episodes that finish standing.
  4. `Episode_Reward/foot_lift` climbing while `rise_progress` climbs: that is
     the hop shortcut being attempted. The planted gate should already make it
     unprofitable; if not, the gate is the thing to check, not the weight.
  5. `Episode_Reward/bow_head_still`, the v5 tripwire. A well-behaved head sits
     at ~2.0 (its weight); the thrashing v4 policy would log ~0.15. Anything
     under ~1.5 while the trunk terms are healthy means the head is being spent
     as a counterweight again — raise W_HEAD_STILL before touching W_HEAD_VEL,
     since the position term is the one that says where the head should BE.
  6. `Episode_Reward/head_joint_vel` should idle near -0.003 (the head ramp's
     own cost) and never approach -0.5. If it parks near that floor the head is
     saturating the cap every step, and the thing to check is whether some
     other term is paying MORE for the swing than this one charges.

Reward MASS check (AGENTS.md: compare mass, not weights, when regularizers are
shared): the episode-average positive stack for a textbook bow is 13.5 (v4's
11.5 plus the +2.0 of `bow_head_still`), still the same order as the velocity
recipe's ~11 that the inherited regularizers were tuned against — with
action_rate_l2 deliberately well below that scaling.
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

# ── Rising out of the crouch (see the version history above) ────────────────
RISEN_TOL = microduck_mdp.BOW_RISEN_TOL          # 0.01 — "back within 1 cm"
FLICKER_S = microduck_mdp.BOW_FLICKER_S          # 0.04 — 2 steps of contact loss

# ── The rise progress potential (v4) ────────────────────────────────────────
RISE_RATE_CAP = microduck_mdp.BOW_RISE_RATE_CAP        # 0.06 m/s
RISE_PAY_START_S = microduck_mdp.BOW_RISE_PAY_START_S  # 2.0 — the rise opens …
RISE_PAY_END_S = microduck_mdp.BOW_RISE_PAY_END_S      # 3.2 — … 0.2 s of grace
# The whole 3 cm climb is worth this many unweighted credits, whatever speed it
# is done at (rate-capped Δ of a potential: 0.030 / (0.06 * 0.02)).
RISE_CREDITS = BOW_DIP_M / (RISE_RATE_CAP * 0.02)
assert abs(RISE_CREDITS - 25.0) < 1e-9

# ── Tolerances: v1's, the run that actually learned the dip ─────────────────
# v3 made every one of these smaller than the motion it measures and the policy
# stopped moving at all (see the history above): a std tighter than the error
# the CURRENT policy can hold is a flat term, not a sharp one.
HEIGHT_STD = 0.02           # m,   ≈ the height error worth pricing on a 3 cm dip
PITCH_STD = 0.12            # rad, ~7°, comfortably under the 10° target
HEAD_STD = 0.30             # rad, the head lags; price the escapable part only
STAND_POSE_STD = 0.12       # rad, leg joints at HOME

# ── v5: the head joints v4 left free ────────────────────────────────────────
# Servo indices in the canonical 14-joint order (AGENTS.md / README / the map
# in tasks/symmetry.py): 5 neck_pitch, 6 head_pitch, 7 head_yaw, 8 head_roll.
# Ranges on robot_walk.xml: neck_pitch [-1.571, 1.047], head_pitch [-1.571,
# 1.571], head_yaw [-2.967, 2.967], head_roll [-0.436, 0.436]; HOME (from
# microduck_constants.py) is neck_pitch 0.3491, head_pitch 0.3491, head_yaw
# 0.0, head_roll 0.0 — so "hold yaw and roll at 0 rad" IS "hold them at HOME".
# head_yaw alone has +/-170 deg of travel and v4 priced none of it.
HEAD_STILL_JOINTS = microduck_mdp.BOW_HEAD_STILL_JOINTS  # (7, 8) yaw, roll
HEAD_STILL_STD = microduck_mdp.BOW_HEAD_STILL_STD        # 0.08 rad ~ 4.6 deg
HEAD_VEL_JOINTS = microduck_mdp.BOW_HEAD_ALL_JOINTS      # (5, 6, 7, 8)
HEAD_VEL_CAP = microduck_mdp.BOW_HEAD_VEL_CAP            # 4.0 rad/s, saturating
# The steepest thing the head ramp ever asks for: down -> up across the 1 s
# rise, |BOW_HEAD_UP - BOW_HEAD_DOWN| on the pitch pair. The cap must sit well
# above it or the velocity cost would tax the motion it is meant to smooth.
HEAD_RAMP_RATE = max(
    abs(u - d)
    for u, d in zip(microduck_mdp.BOW_HEAD_UP, microduck_mdp.BOW_HEAD_DOWN)
) / (RISE_END_S - HOLD_END_S)
assert HEAD_VEL_CAP > 5.0 * HEAD_RAMP_RATE

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# ── Reward weights ──────────────────────────────────────────────────────────
# v1's stack verbatim, plus exactly three new terms (rise_progress, risen,
# terminated) and two adjustments (stand_pose 2 -> 3, action_rate -0.2 ->
# -0.05). Episode-average positive task mass ~= 11.5 for a textbook bow,
# comparable to velocity's ~11, so the shared regularizers keep their relative
# strength (AGENTS.md: compare reward MASS, not weights, when copying
# regularizers between envs) — with the deliberate exception of action_rate_l2.
# The full four-strategy arithmetic is in the module docstring and in
# `tests/test_bow_cfg.py::test_the_four_strategies`.
W_HEIGHT = 3.0        # the shape of the bow — and the ramp back UP  (v1)
W_PITCH = 2.0         # nose-down: what makes the crouch read as a bow  (v1)
W_HEAD = 2.0          # head down, then lifted  (v1)
# v6 (2026-09-06): v5 pinned the head (yaw/roll within a degree) but the bow
# regressed to a half-crouch (dip 0.094 m, final 0.099): the always-on
# head-still income diluted the rise. Weights only: rise progress 6 -> 12,
# risen 5 -> 10, stand pose 3 -> 4, head still 2 -> 1.
W_HEAD_STILL = 1.0    # THE v5 TERM. head_yaw and head_roll pinned to HOME for
                      # the whole episode. v4 bowed correctly and swung its head
                      # around violently while doing it — those two joints were
                      # priced by nothing at all, so the policy used the heaviest
                      # link on the robot as a free counterweight. Sized to match
                      # W_HEAD: the direction the head POINTS is worth as much as
                      # the height it is held at.
W_HEAD_VEL = -0.5     # bounded, saturating at HEAD_VEL_CAP. Prices the SPEED of
                      # all four head joints, which is the only way to reach
                      # neck_pitch and head_pitch — they have to move, so no
                      # position term can settle them. At the ramp's own rate it
                      # costs ~0.01/step; at the cap, 0.5/step. Bounded on
                      # purpose: an unbounded velocity cost is exactly the shape
                      # that made quitting cheap in v3.
W_UPRIGHT = 2.0       # lateral tilt only — the pitch is intended  (v1)
W_TRACK = 0.5         # zero twist = don't travel, don't turn  (v1)
W_STAND_POSE = 4.0    # the finish, stand phase only. v1's 2.0 was too cheap to
                      # outbid the crouch; v2's 5.0 was raised without a path to
                      # the pose it pays for, so nothing collected it. 3.0 plus
                      # a gradient up (below) is the version that has both.
W_RISE_PROGRESS = 12.0 # THE v4 TERM. Potential-based Δ of a running-max trunk
                       # height over the rise window, rate-capped: the full 3 cm
                       # is worth 25 credits = 150 points however it is climbed,
                       # and a partial rise is worth exactly its fraction. Every
                       # earlier version priced where the trunk IS, so a policy
                       # one centimetre up an unfinished rise collected nothing
                       # for that centimetre and the crouch was a flat optimum.
W_RISEN = 10.0        # ONE step per episode: back up, feet planted. Same size
                      # and same one-shot shape as happy-spin's completion.
W_TERMINATED = -20.0  # v3 DIVERGED by learning to fall over: its parked-pose
                      # costs meant an episode that ENDS stops paying them
                      # (collapsed 31.8/iteration vs 4.6 time-outs, mean length
                      # 97 of 200). A terminal state must be the worst outcome
                      # available. This is INSURANCE, not the mechanism — what
                      # really prices a collapse is the hundreds of steps of
                      # forfeited income (see the arithmetic in the docstring:
                      # remove the -20 and collapsing is still last by 830).
                      # 20 points is ~1.7 steps of a textbook bow's take, and
                      # reads as -0.1 in wandb's Episode_Reward units.
W_FOOT_LIFT = -3.0    # the hard constraint: both feet planted, always  (v1),
                      # with v2's 40 ms flicker exemption so the heel unweight
                      # of a legitimate push-up is not priced as a hop.
W_DRIFT = -1.5        # a bow happens on the spot  (v1)
W_SLIP = -0.2         # (v1)
W_ACTION_RATE = -0.05  # FIXED, no curriculum. v1's -0.2 taxed the dip and the
                       # rise; the whole trick is motion, so make it cheap. Not
                       # zero: the servos still have to be protected from thrash.


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

    # Smoothness: FIXED and LIGHT. The velocity curriculum ramps this to -1.0
    # by iter 1500, which over a 2000-iter episodic run would spend most of
    # training taxing a slow, large, deliberate motion. v1 ran it at -0.2; the
    # v2 log shows that alone costing a MOTIONLESS robot 0.71/step, more than
    # the whole dip was worth. Motion is the trick: make it cheap.
    cfg.rewards["action_rate_l2"].weight = W_ACTION_RATE

    sensor_name = "feet_ground_contact"
    trunk = SceneEntityCfg("robot", body_names=("trunk_base",))

    cfg.rewards["bow_height"] = RewardTermCfg(
        func=microduck_mdp.bow_height_reward,
        weight=W_HEIGHT,
        params={
            "sensor_name": sensor_name,
            # The ramp itself lives in `_bow_update` (STAND_Z → BOW_Z → STAND_Z):
            # one source of truth for every height-sensitive term.
            "std": HEIGHT_STD,
        },
    )
    cfg.rewards["bow_pitch"] = RewardTermCfg(
        func=microduck_mdp.bow_pitch_reward,
        weight=W_PITCH,
        params={
            "sensor_name": sensor_name,
            "pitch_rad": BOW_PITCH_RAD,
            "std": PITCH_STD,   # ~7°, comfortably under the 10° target
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
    # THE v5 TERMS. bow_head above prices neck_pitch and head_pitch only; these
    # two close the gap that let the v4 policy thrash its head through the bow.
    cfg.rewards["bow_head_still"] = RewardTermCfg(
        func=microduck_mdp.bow_head_still_reward,
        weight=W_HEAD_STILL,
        params={
            "sensor_name": sensor_name,
            "joint_indices": HEAD_STILL_JOINTS,  # (7, 8) = head_yaw, head_roll
            "std": HEAD_STILL_STD,               # HOME is 0.0 for both
        },
    )
    cfg.rewards["head_joint_vel"] = RewardTermCfg(
        func=microduck_mdp.bow_head_vel_penalty,
        weight=W_HEAD_VEL,
        params={
            "sensor_name": sensor_name,
            "joint_indices": HEAD_VEL_JOINTS,    # all four neck/head joints
            "vel_cap": HEAD_VEL_CAP,
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
        },
    )
    # THE v4 TERM: a dense, potential-based gradient all the way up. Everything
    # else in this stack prices where the trunk IS; this is the only term that
    # pays for the act of going up, so a rise abandoned half-way still banks
    # half the money and there is a slope to follow from the crouch.
    cfg.rewards["rise_progress"] = RewardTermCfg(
        func=microduck_mdp.bow_rise_progress_reward,
        weight=W_RISE_PROGRESS,
        params={"sensor_name": sensor_name, "rate_cap": RISE_RATE_CAP},
    )
    # … and the milestone that closes it: back up, feet planted, having bowed.
    cfg.rewards["risen"] = RewardTermCfg(
        func=microduck_mdp.bow_risen_bonus,
        weight=W_RISEN,
        params={"sensor_name": sensor_name},
    )
    # Collapsing must never be the cheap exit (v3 learned exactly that). Only
    # the two BEHAVIOURAL terminations are charged — a nan_state termination is
    # a sim blow-up, not something the policy chose.
    cfg.rewards["terminated"] = RewardTermCfg(
        func=microduck_mdp.bow_termination_penalty,
        weight=W_TERMINATED,
        params={"term_names": ("collapsed", "fell_over")},
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
