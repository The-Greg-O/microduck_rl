"""Microduck JUMP task — a hop in place: both feet off the floor, land standing.

Episodic skill: from a standing start, load both legs, extend them together so
BOTH FEET LEAVE THE FLOOR WITHIN ONE CONTROL STEP OF EACH OTHER, reach an apex
of >= 13 mm above the settled standing height, come down on both feet and settle
back to standing — all inside a 2 s episode. Deploys as a constant-command
`episodic` skill (`robotctl policy add jump <repo>`, `--duration-s 2.0`), like
tippy-taps and the bow. NO DIRECTION SIGN (unlike the pivot), so the twist slot
stays pinned near zero and one ONNX installs once.

Built on the velocity recipe (AGENTS.md workflow step 1) so DR, observation
noise, delays, the BAM actuator, the 61D obs contract and the `_safe` critic
terms come for free, and structured exactly like bow and pivot: a pinned
command, a short episode, one per-env memory ticked once per step by every
reward term, potential-based progress, a one-shot completion bonus, an `alive`
income and a termination penalty.

Recipe source: `go-grgs/docs/research/jump-design.md` (ticket The-Greg-O/go-grgs
#21, docket row 5). Everything below that reads as a number was measured there
before this file existed — AGENTS.md workflow step 2 and the docket's "Skill
iteration protocol" step 1, feasibility for free before anything is paid for.

── What the body can actually do (Part A of the design note) ────────────────
2430 scripted open-loop schedules swept on the walk model under BAM, plus
rollouts of the trained runner for comparison:

  * 3 cm IS NOT REACHABLE, and the ceiling is the servo's SPEED, not its
    torque. The XL330 m6's free-running speed is 16.2 rad/s and the leg's
    extension gain at the moment of take-off is 0.024 m/rad, so the fastest the
    body can leave the floor is 0.39 m/s — a ballistic rise of 7.7 mm and a
    flight of 79 ms. A 3 cm apex needs 0.767 m/s, twice that. Torque has 8x of
    spare (the push needs 12.6 N; the legs make 99 N), and raising the current
    limit 10x changes the hop not at all. The ticket's 3 cm should be re-cut to
    13-16 mm; nothing in any reward can buy the difference.
  * A HOP DOES EXIST. Best scripted schedule that lands and stays up: both feet
    off for 0.06 s, trunk apex 0.1284 m (+13.5 mm over the settled standing
    height of 0.1149 m), foot clearance 8-10 mm flat-footed, peak tilt 32.4 deg
    on landing. The best flight of any configuration is 0.08 s at +15.8 mm —
    that one falls.
  * THE LANDING IS THE REWARD'S JOB. The schedule that lands upright at seed 0
    falls in 5 of 6 other seeds: the flight survives every seed (0.02-0.06 s,
    +12.8 to +14.9 mm), the landing does not, because the per-env battery
    voltage (6.5-8.2 V, sampled even with `domain_rand=False`) moves the launch
    by ~10%. A feed-forward hop is repeatable; a feed-forward landing is not.
  * THE FARM TO DESIGN AGAINST IS A HEIGHT REWARD WITHOUT A CONTACT GATE. The
    trained runner `run-v2` reaches a trunk z of 0.1277 m mid-stride — 81% of
    the best hop's apex — with both feet off the floor for 0.5% of its steps.
    `alpha_walking` is the same. NEITHER POLICY HAS EVER LEFT THE FLOOR. And a
    tall STATIC park is worth only +2.9 mm, so the gate cannot be height and
    cannot be stillness: it has to be CONTACT.

Success metric this recipe trains for (the design note's restatement of #21):
both feet off the floor for >= 60 ms with the trunk apex >= 13 mm above 0.115 m,
landing on both feet with tilt under 15 deg and settled within 0.5 s. Stretch:
16 mm / 80 ms.

── The two knobs that are not the house default, and why ────────────────────
`ENABLE_SYMMETRY = True`. A hop is genuinely left/right symmetric — the bow's
argument ("the mirror loss is a valid prior and buys sample efficiency, unlike
the one-footed tricks where it must be off") — and here it is also a prior for
the very thing the reward is trying to buy: the mirror loss pushes the two legs
to do the same thing at the same time, which is the ticket's "both feet
together".

`EPISODE_LENGTH_S = 2.0` (100 control steps at 50 Hz) against the house
standard of 4.0. The manoeuvre takes ~0.65 s end to end (0.20 s load, 0.06 s
push, 0.06 s flight, 0.30 s settle), so a 4 s episode is 3.3 s of standing
still. That matters twice: it doubles the `alive` income the hop has to compete
against (600 instead of 300, against a hop's ~130) and it doubles the
wall-clock per rehearsal. If v1 needs more settling room, 3.0 s is the fallback
— not 4.0.

── The observation situation: the crouch is the clock ───────────────────────
NO OBSERVATION CHANGE. Same 61D contract, same actor. A hop needs a launch
INSTANT and the actor has no clock (none of bow, pivot or happy-spin gives it
one). Three places to get one, and this recipe uses the second:

  1. a phase signal in the observation — rejected, it breaks the 61D hot-swap
     contract and the daemon-side phase signal is not built;
  2. A PROPRIOCEPTIVE CUE, which is free: the launch instant is THE BOTTOM OF
     THE HOP'S OWN CROUCH, and crouch depth is fully visible in `joint_pos`
     (14 slots) with its rate in `joint_vel` (14 more). "Extend hard when the
     knees are folded and the fold has stopped deepening" is a state-feedback
     rule, not a timed one. The state machine is in the legs — which is the
     structural reason a hop should be EASIER than happy-spin's stop, which has
     no such cue and is therefore latency-sensitive;
  3. the episode window backing it up: `jp_load`'s 0.8 s window fixes the launch
     to the front of the episode FROM THE REWARD SIDE, without putting a clock
     in the observation.

The half that IS delay-sensitive is the landing — a reflex on gyro and projected
gravity, the two delayed channels. That is why the lab eval is read under the
lab's 3-6 step observation delay as well as without it, and why a
delay-widening knob is v3's variable, never v1's.

── The reward stack ─────────────────────────────────────────────────────────
Every `jp_*_penalty` returns a NON-NEGATIVE 0..1 cost and carries a NEGATIVE
weight, which is what the bow / pivot / happy-spin stacks all actually do
despite AGENTS.md's naming rule. `test_gait_terms_gone_and_every_weight_is_signed_the_right_way`
asserts it; the run check is that every `Episode_Reward/<penalty>` is <= 0.

  * `alive` +3.0/step, gated at 25 DEG of tilt. Pivot v5's decisive term, at
    pivot v5's weight: v3 ended every episode in `fell_over` and had never
    priced falling at all; v4 priced it at -20 and the curve went the WRONG WAY
    monotonically (length 191 -> 77, falls 1.6 -> 47 per iteration, progress
    rising the whole way) because "a turn which falls at ~0.9 s out-earns a
    careful one". Over 100 steps this is 300 points a fall forfeits, against
    everything a hop can earn (~130). The gate is 25 deg and not the pivot's 15
    because the measured hop peaks at 32.4 deg on its own landing; 25 deg still
    costs it ~9 of the 300, which is the right size for a nudge.
  * `jp_flight` +25.0/step, GATED ON BOTH FEET OFF, paid by
    `clip(min(foot site z) / 0.012, 0, 1)`. The hop itself. The gate is the
    whole term (see the ungated table below); the 0.012 m normaliser is the
    measured 8-10 mm flat-foot clearance, where the lab's `_af_airborne` uses
    0.06 m that this body can never reach. One thing the design note did not
    spell out and `_jp_update` has to: THE SPAWN DROP IS NOT A HOP. The reset
    places the trunk at the keyframe height, where the soles sit ~13 mm above
    the floor, so the first steps of every episode have both feet off —
    ungated that free fall would pay this term at a saturating clearance,
    advance the apex potential, arm the landing bonus without a hop and switch
    `jp_load` off permanently at t = 0. Every airborne test is ANDed with a
    touched-down latch, exactly as the bow gates its foot-lift cost.
  * `jp_apex` +15.0, POTENTIAL-BASED on `clip((z - 0.115)/0.015, 0, 1)`,
    advancing only while airborne. Pays each new maximum once — holding pays
    zero, re-climbing pays zero, overshoot pays zero. bow v4's lesson stated
    plainly ("the potential-based rise-progress term was the missing piece");
    bow v2 is the counter-example, a per-step height Gaussian that produced a
    compromise pose collecting partial credit from both phases while never
    paying the action-rate cost of moving. At this weight and normaliser the
    term pays exactly ONE POINT PER MILLIMETRE of apex, capped at 15.
  * `jp_land` +15.0, ONE SHOT, >= 0.30 s after the first genuine flight ends,
    both feet down, tilt < 15 deg, |z - 0.115| < 20 mm, |vz| < 0.15 m/s.
    Pivot v5's finish-upright jackpot and the bow's `risen` in the same shape.
    Conditioned on a prior flight so a duck that never leaves the floor cannot
    collect it; one-shot so it cannot be farmed by bouncing.
  * `jp_load` +0.5/step, only BEFORE the first flight and only in the FIRST
    0.8 s, `clip((0.115 - z)/0.030, 0, 1)`. The gradient's first rung: from
    "stand still" there is otherwise nothing at all pointing at a hop. Bow v1,
    v2 and v6 all PARKED IN THE CROUCH, and two things stop that here — the
    0.8 s window caps the term at 20 POINTS AN EPISODE (against the flight's
    100+), and it switches off permanently at the first flight, so a crouch is
    only ever worth anything as a prelude.
  * `jp_stagger` -2.0, `clip((|Δ last-contact step| - 1)/4, 0, 1)`, latched at
    the launch and charged for the rest of the episode. The ticket's own words.
    Without it the cheapest route to both-feet-off is one foot then the other —
    which is what tippy-taps already trains and what the runner does 95% of the
    time. Tolerance ONE control step, deliberately loose (bow v3's 6 mm
    tolerances diverged: "narrowing a Gaussian below the error the CURRENT
    policy can hold does not sharpen the gradient, it deletes it").
  * `jp_drift` -1.0, `clip(||xy - home||^2 / 0.0025, 0, 1)` — saturating at
    5 cm. A hop in place, not a leap. The bow's shape at -1.5/10 cm; 5 cm here
    because a hop travels less than a bow does.
  * `upright` +2.0 (inherited, unchanged) — common survival income at the
    velocity/pivot value so the arithmetic is directly comparable to the
    pivot's tables. NOT the bow's lateral-only variant: a hop has no intended
    pitch, so the full tilt is the right quantity.
  * `head_still` +1.0 at std 0.5 RAD on joints (7, 8). bow v4: "a joint you do
    not price is a joint the policy will spend" — head yaw and roll were
    unpriced and became a free counterweight on a head carrying 38% of the
    mass. bow v5 pinned them at 0.08 rad and the always-on head-still income
    diluted the rise; v7's single variable was exactly this tolerance,
    0.08 -> 0.5. Start at v7's number. If the launch turns out to need a head
    throw this is the first term to drop — but it is not v1's variable.
  * `head_joint_vel` -0.5, the bow's BOUNDED form over joints (5, 6, 7, 8):
    bounded because "an l2 on joint velocity has no ceiling ... a cost like
    that driving the per-step total toward zero is precisely the shape that
    taught bow v3 to fall over".
  * `terminated` -100.0 on `fell_over` only. Pivot v5's number and pivot v5's
    scoping: a `nan_state` is a sim blow-up rather than something the policy
    chose, and `time_out` is the normal end of a 2 s episode.
  * inherited regularisers: `dof_pos_limits` -1.0, `body_ang_vel` -0.05,
    `angular_momentum` -0.02, `self_collisions` -1.0, `foot_slip` -0.2 WITH
    `command_threshold = -1.0` (the bow's fix — otherwise it is silently inert
    at a pinned command), `track_linear_velocity` +0.5 and
    `track_angular_velocity` +0.5 (zero-twist tracking = "don't travel, don't
    turn"), `head_pose_tracking` and `body_pose_tracking` at 0.0 so the obs
    slots' roles stay visible in the log.
  * `action_rate_l2` -0.05, FIXED, no curriculum — the bow's choice, not the
    velocity ramp to -1.0. AGENTS.md: "any attempt-tax active while a hard
    skill is being explored makes 'do nothing' win", and a hop is the most
    action-rate-expensive thing in the whole docket (the push is a full-stroke
    joint slam in three control steps).

DELETED REWARDS: `air_time`, `foot_clearance`, `foot_swing_height`, `pose`,
`head_pose_bias`, and the curricula `standing_envs`, `head_pose_range`,
`body_pose_range`, `head_pose_bias_weight`, `action_rate_weight`. Note on
`air_time` specifically, because it LOOKS like it should pay a hop: its window
is 0.125-0.300 s of PER-FOOT flight and the whole hop's flight is 0.06-0.08 s,
so it would pay zero even if it were live — and it is not live, because it gates
on command magnitude and this command is pinned near zero.

NO `height_stand` TERM. Pivot and happy-spin both carry a Gaussian on trunk z at
0.115; a hop must go 3 cm BELOW it and 1.5 cm above it, so that term would price
both halves of the manoeuvre as error. The height that matters is priced by
`jp_apex` (potential, airborne-only) and `jp_land` (one-shot, at the end).

NO HEIGHT TERMINATION, and this is deliberate: the crouch bottoms at 0.088 m and
the reachable crouch floor is 0.061 m, both at or under the lab's `FALL_HEIGHT`
0.07 and both under the bow's `BOW_COLLAPSE_Z` 0.060. The lab's `deep_squat`
turns the z-kill off for exactly this reason, and the A/B recorded there (park at
0.076 with the kill on, 0.070 with it off and 22% more reward) says the kill
MOVES THE POLICY OFF THE GOAL POSE. A hop's crouch is deeper than the bow's, so
the bow's `collapsed` termination must NOT be copied across.

── Two arithmetic rules the stack obeys ─────────────────────────────────────
  * A FALL IS NEVER PROFITABLE. Everything a hop can earn (flight ~100 + apex 15
    + land 15 = 130) is worth less than the `alive` income a fall forfeits (up
    to 300) plus the -100. Pivot v5: "the decisive line is the alive income a
    fall forfeits, not the penalty".
  * ENDING THE EPISODE IS NEVER AN ESCAPE. Total per-step penalties are bounded
    at -3/step (`jp_stagger` -2 saturated + `jp_drift` -1 saturated, and those
    two cannot both saturate while standing still), so a surviving non-hopping
    episode is billed at most ~300 against a fall's 400. Read the two failures
    this inverts: bow v3 DIVERGED because "terminating beat staying alive", and
    the pivot audit measured a surviving non-pivot billed at -1060 per episode
    against a fall's -100.

── The arithmetic: the stack scored over whole 2 s episodes ─────────────────
Per-episode totals (mjlab's `Episode_Reward/x` x 100). The design note scored
these on RECORDED trajectories (protocol step 2); the numbers below are what
`tests/test_jump_cfg.py::test_the_arithmetic_hopping_wins` measures by running
the REAL reward functions over synthetic reconstructions of those same
trajectories for all 100 steps (protocol step 4: synthetic arithmetic is a
sanity check on the ORDERING, never the reason to launch).

  strategy                        steps  alive  flight  apex  land  load   term    total
  textbook hop (best, lands)        100    291   100.0  13.5  15.0   2.8      0    422.3
  crouch-and-park that holds        100    300       0     0     0  20.0      0    320.0
  stand still                       100    300       0     0     0     0      0    300.0
  trained runner (stride bounce)    100    300       0     0     0     0      0    300.0
  tallest stable park (+2.9 mm)     100    288       0     0     0     0      0    288.0
  hop that lands and falls           50    120    52.4  15.0     0   2.1   -100     89.4
  crouch-and-fall                    47    102       0     0     0   6.3   -100      8.3

ORDERING: hop (422) > crouch-park (320) > still ~= runner (300) > tall park
(288) >> hop-and-fall (89) > crouch-and-fall (8). The margin from the best
non-hop to the hop is +102 (32%); a fall costs -333 relative to the hop.

The test's synthetic reconstruction reproduces every row of that table within a
point (423.2 / 320.0 / 300.0 / 300.0 / 300.0 / 89.6 / 8.7) with ONE exception:
the tallest stable park scores 300 here rather than the recorded 288, because
the 12 points it gives up in the recorded rollout are a spawn transient outside
the 25 deg `alive` gate that a synthetic constant-height park has no way to
reproduce. It is worth nothing either way — which is the row's whole point.

THE SAME STACK WITH THE BOTH-FEET-OFF GATE REMOVED — the naive version, and the
one a first draft would have written:

  strategy                        flight   apex    total
  trained runner (stride bounce)  1352.4   12.5   1664.9
  tallest stable park              456.4    4.3    748.7
  textbook hop                     305.9   13.5    615.9
  stand still                       27.3    0.9    328.3

Ungated, a WALKING policy out-earns the hop 2.7 to 1 and a static tall pose
beats it too. This is the bow v2 failure in a new costume, and it is why the
gate is not negotiable. `test_removing_the_flight_gate_hands_it_to_the_runner`
reproduces the mechanism on the synthetic trajectories — 1976 for the runner
against 113 for the hop, a wider gap than the recorded 1665 / 616 because the
synthetic runner's swing foot saturates the 12 mm normaliser for most of its
stride — and then shows the same two terms paying the runner EXACTLY ZERO once
the contact gate is put back.

THE EXPLORATION BARRIER, and why it is survivable: an attempt that FALLS costs
333 relative to the hop, so on those numbers alone a policy should only attempt
when its odds beat (300 - 89)/(422 - 89) = 63%. But the relevant comparison is
not hop-vs-fall, it is ATTEMPT-VS-STAND: an attempt that leaves the floor and
lands badly WITHOUT falling scores 300 plus whatever flight it earned, i.e.
never worse than standing still. Only `fell_over` is punished, and a 6 cm-high
hop has a very short way to fall. `jp_load` is the rung into that region.

── PREDICTION FOR v1 (written before launch, per the protocol) ──────────────
`jump-v1`, 2000 iterations, 4096 envs, `num_steps_per_env` 24,
`experiment_name="jump"`. To be read against the curve afterwards:

  * MEAN EPISODE LENGTH 95-100 of 100 by iteration 400 and stable;
    `Episode_Termination/fell_over` under 3 per iteration by iteration 1000.
    Pivot v5's `alive` did exactly this job (length 198 by iteration 200, falls
    0.3-1.6, held for 1200 iterations) and it is the part of that recipe I
    trust most.
  * `Episode_Reward/jp_load` RISES FIRST (iteration 200-400) and then FALLS
    BACK as `jp_flight` takes over. If it rises and STAYS at its 0.20 cap with
    `jp_flight` at 0.00, the policy has parked in the crouch — bow v1's failure
    — and v2's single variable is the 0.8 s window, not a weight.
  * `Episode_Reward/jp_flight` 0.3-0.8 (of weight 25.0, i.e. 30-80 points an
    episode) by iteration 1500. IT WILL NOT REACH ITS CAP: the cap is 12 mm of
    clearance held for many steps, and the physics allows 8-10 mm for three.
  * `Episode_Reward/jp_apex` 0.08-0.14 (8-14 of the 15 available). The term
    with the most headroom and the one most likely to look solved while the
    flight is marginal. Read it AGAINST `jp_flight`, never alone.
  * `jp_land` 0.06-0.12 — the hop lands cleanly in roughly half to
    three-quarters of episodes. The term I expect to be v2's variable.
  * LAB EVAL, 6 episodes, `render-rollout`: both feet off in 4-6 of 6, flight
    40-80 ms, trunk apex +10 to +16 mm, 0-2 falls. Under the lab's 3-6 step
    observation delay I expect ONE OR TWO MORE FALLS than without it — the
    landing is the delay-sensitive half, in exactly the way happy-spin's stop
    is.
  * THE MOST LIKELY FAILURE MODE IS A ONE-FOOT HOP WITH A TWO- OR THREE-STEP
    STAGGER — cheap, nearly as tall, and only `jp_stagger` stands against it.
    If the contact sheet shows that, v2's single variable is `jp_stagger`'s
    TOLERANCE (1 step -> 0), not its weight.

VERDICT RULE. v1 works if the lab eval shows both feet off in >= 4 of 6 episodes
with <= 1 fall. If `jp_flight` is below 0.1 at iteration 1500 the run is dead and
v2's single variable is the `jp_load` window; if `jp_flight` is healthy and the
falls are on landing, the variable is `jp_land`'s tolerance band.

Open question carried from the design note: the head throw. 38% of the mass is
in the head (`airflip._af_head_throw` exists for that reason), but a scripted
+/-0.5 rad neck/head swing through the launch changed the apex by -2 to +1 mm —
nothing. A policy could time it against the push in a way a script cannot; if v1
stalls at a marginal flight that deserves a feasibility pass of its own before
it becomes a reward term, and `head_still` is the term it would displace.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

# A hop is left/right symmetric, and here the mirror loss is also a PRIOR FOR
# THE THING THE REWARD IS BUYING: it pushes the two legs to do the same thing at
# the same time, which is the ticket's "both feet leave the floor together".
# (Off for the one-footed tricks — pivot, flamingo — where it is a false prior.)
ENABLE_SYMMETRY = True

# 100 control steps at 50 Hz. The manoeuvre is ~0.65 s end to end; a 4 s episode
# would be 3.3 s of standing still, doubling both the `alive` income a hop has
# to compete against and the wall-clock per rehearsal. Fallback if v1 needs more
# settling room: 3.0, never 4.0.
EPISODE_LENGTH_S = 2.0

# ── Geometry (measured; see the module docstring and the design note) ────────
STAND_Z = microduck_mdp.JP_STAND_Z            # 0.115 settled standing trunk z
APEX_M = microduck_mdp.JP_APEX_M              # 0.015 — the apex normaliser
CLEARANCE_M = microduck_mdp.JP_CLEARANCE_M    # 0.012 — measured flat-foot clearance
LOAD_DEPTH_M = microduck_mdp.JP_LOAD_DEPTH_M  # 0.030 — the crouch the load prices
LOAD_WINDOW_S = microduck_mdp.JP_LOAD_WINDOW_S  # 0.8 s — and the cap that gives it
ALIVE_TILT_DEG = microduck_mdp.JP_ALIVE_TILT_DEG  # 25, not the pivot's 15
LAND_TILT_DEG = microduck_mdp.JP_LAND_TILT_DEG    # 15 — the finish cone
LAND_SETTLE_S = microduck_mdp.JP_LAND_SETTLE_S    # 0.30 s after the flight ends
FLIGHT_MIN_S = microduck_mdp.JP_FLIGHT_MIN_S      # 0.04 = "a genuine flight"
DRIFT_SAT_M = microduck_mdp.JP_DRIFT_SAT_M        # 0.05 — a hop in place

HEAD_STILL_JOINTS = microduck_mdp.JP_HEAD_STILL_JOINTS  # (7, 8) yaw, roll
HEAD_STILL_STD = microduck_mdp.JP_HEAD_STILL_STD        # 0.5 rad — bow v7's number
HEAD_VEL_JOINTS = microduck_mdp.JP_HEAD_ALL_JOINTS      # (5, 6, 7, 8)
HEAD_VEL_CAP = microduck_mdp.JP_HEAD_VEL_CAP            # 4.0 rad/s, saturating

# The load term is capped by its window, not by a clip: 0.8 s x 50 Hz x 0.5.
LOAD_CAP_POINTS = 0.5 * LOAD_WINDOW_S / 0.02
assert abs(LOAD_CAP_POINTS - 20.0) < 1e-9

# ── Reward weights ──────────────────────────────────────────────────────────
# The full table, the strategy arithmetic and the ungated counter-example are in
# the module docstring; `tests/test_jump_cfg.py` runs both tables through the
# real reward functions rather than asserting them in prose.
W_ALIVE = 3.0        # pivot v5's term at pivot v5's weight, gated at 25 deg (not
                     # 15): the measured hop peaks at 32.4 deg on its own
                     # landing. 100 steps = 300 points a fall forfeits.
W_FLIGHT = 25.0      # THE TERM. Both feet off, paid by clearance / 0.012 m.
                     # Ungated, the trained runner takes 1352 of it for a gait
                     # that never leaves the floor and wins the stack 1665 to
                     # 616. The gate is not negotiable.
W_APEX = 15.0        # potential-based, airborne-only. With APEX_M = 0.015 this
                     # is exactly one point per millimetre of apex, capped at 15.
W_LAND = 15.0        # one shot, conditioned on a prior flight. Pivot v5's
                     # `pivot_complete` in the same shape and at the same size.
W_LOAD = 0.5         # the first rung, capped at 20 points an episode by the
                     # 0.8 s window and killed at the first flight.
W_STAGGER = -2.0     # "both feet together", one control step of tolerance.
W_DRIFT = -1.0       # a hop in place: 5 cm saturates.
W_UPRIGHT = 2.0      # inherited, unchanged — the common survival income, at the
                     # velocity/pivot value so the tables stay comparable.
W_HEAD_STILL = 1.0   # bow v7's tolerance (0.5 rad): priced, but loose enough
                     # that the head can still be the policy's clock.
W_HEAD_VEL = -0.5    # bow's BOUNDED speed cost over all four head joints.
W_TERMINATED = -100.0  # pivot v5's marker on `fell_over` only. The larger half
                       # of a fall's price is still the forfeited `alive`.
W_TRACK = 0.5        # zero twist = "don't travel, don't turn".
W_SLIP = -0.2        # with command_threshold = -1.0, or it is inert at a pinned
                     # command (the bow's fix).
W_ACTION_RATE = -0.05  # FIXED, no curriculum. The push is a full-stroke joint
                       # slam in three control steps; an attempt-tax during
                       # discovery makes "do nothing" win.


def make_microduck_jump_env_cfg(
    play: bool = False, rough: bool = False
) -> ManagerBasedRlEnvCfg:
    """The jump env: a 2 s hop in place, both feet off together, land standing."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Command: pinned near zero for the whole episode (obs-shape parity, the
    # 61D contract's slots kept alive; being SELECTED is the trigger, as for
    # tippy-taps and the bow). No direction sign, so one ONNX installs once. ──
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
    # their widening curricula and their reward go. com_range / head_com_range
    # (DR) are kept.
    for gone in (
        "standing_envs",         # drives a command we no longer vary
        "head_pose_range",       # the head is priced here, not commanded
        "body_pose_range",       # body_pose tracking is weight 0 (as in velocity)
        "head_pose_bias_weight", # ramps a term this env deletes
        "action_rate_weight",    # would ramp -0.1 → -1.0; see W_ACTION_RATE
    ):
        if gone in cfg.curriculum:
            del cfg.curriculum[gone]

    # ── Rewards: drop the gait terms ────────────────────────────────────────
    # air_time / foot_clearance / foot_swing_height: gait shaping, and SILENTLY
    #   INERT at this pinned command (they gate on command magnitude) — deleting
    #   them is hygiene. Note `air_time` in particular LOOKS like it should pay
    #   a hop and does not: its window is 0.125-0.300 s of PER-FOOT flight and
    #   the whole hop's flight is 0.06-0.08 s, so it would pay zero even if it
    #   were live. (The pivot's warning applies in reverse: there the yaw slot
    #   carries ±1, so these terms WOULD have been live and would have demanded
    #   the pin foot swing.)
    # pose: a fixed HOME-pose Gaussian prices the crouch and the tuck as error.
    # head_pose_bias: the head is priced by head_still / head_joint_vel here.
    for name in ("air_time", "foot_clearance", "foot_swing_height", "pose",
                 "head_pose_bias"):
        if name in cfg.rewards:
            del cfg.rewards[name]

    # The head is priced by the two terms below, not by the head_pose command.
    # The tracking terms stay at weight 0 so the obs slots' roles remain visible
    # in the log (the pattern velocity uses for body_pose_tracking).
    cfg.rewards["head_pose_tracking"].weight = 0.0
    cfg.rewards["body_pose_tracking"].weight = 0.0

    # Tracking a ~zero twist = "don't travel, don't turn", kept light.
    cfg.rewards["track_linear_velocity"].weight = W_TRACK
    cfg.rewards["track_angular_velocity"].weight = W_TRACK

    # foot_slip is command-gated in the velocity recipe and would be dead at a
    # pinned ~zero command; -1.0 turns the gate permanently on (the bow's fix).
    cfg.rewards["foot_slip"].weight = W_SLIP
    cfg.rewards["foot_slip"].params["command_threshold"] = -1.0

    # Smoothness: FIXED and LIGHT. A hop is the most action-rate-expensive thing
    # in the docket — the push is a full-stroke joint slam in three control
    # steps — and an attempt-tax active during discovery makes "do nothing" win.
    cfg.rewards["action_rate_l2"].weight = W_ACTION_RATE

    # NOTE what is NOT added: no `height_stand`. Pivot and happy-spin both carry
    # a Gaussian on trunk z at 0.115; a hop must go 3 cm below it and 1.5 cm
    # above it, so that term would price both halves of the manoeuvre as error.

    sensor_name = "feet_ground_contact"

    def feet() -> SceneEntityCfg:
        # A FRESH SceneEntityCfg per term: the managers resolve these in place
        # (site_names → site_ids), and a module-level instance shared across
        # terms — and across the train / play / backlash copies of this cfg —
        # would bind to whichever scene got there first.
        return SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))

    # THE TERM THE WHOLE RECIPE TURNS ON. Both feet off the floor, paid by
    # clearance so a step of contact chatter earns almost nothing. The trained
    # runner reaches 81% of the hop's apex with a foot planted and takes ZERO of
    # this; ungated it takes 1352 and wins the stack outright.
    cfg.rewards["jp_flight"] = RewardTermCfg(
        func=microduck_mdp.jp_flight_reward,
        weight=W_FLIGHT,
        params={"sensor_name": sensor_name, "feet_cfg": feet(),
                "clearance_m": CLEARANCE_M},
    )
    # Potential-based height, advancing only while airborne: rising pays,
    # holding pays zero, a stride bounce pays zero.
    cfg.rewards["jp_apex"] = RewardTermCfg(
        func=microduck_mdp.jp_apex_reward,
        weight=W_APEX,
        params={"sensor_name": sensor_name, "feet_cfg": feet()},
    )
    # The finish: one shot, 0.30 s after the first genuine flight ends, on both
    # feet, upright, at standing height, stopped.
    cfg.rewards["jp_land"] = RewardTermCfg(
        func=microduck_mdp.jp_land_bonus,
        weight=W_LAND,
        params={"sensor_name": sensor_name, "feet_cfg": feet()},
    )
    # The first rung of the gradient — and the term most likely to be v2's
    # variable if the policy parks in the crouch instead of leaving the floor.
    cfg.rewards["jp_load"] = RewardTermCfg(
        func=microduck_mdp.jp_load_reward,
        weight=W_LOAD,
        params={"sensor_name": sensor_name, "feet_cfg": feet(),
                "depth_m": LOAD_DEPTH_M, "window_s": LOAD_WINDOW_S},
    )
    # "Both feet leave the floor together", with one control step of tolerance.
    cfg.rewards["jp_stagger"] = RewardTermCfg(
        func=microduck_mdp.jp_stagger_penalty,
        weight=W_STAGGER,
        params={"sensor_name": sensor_name, "feet_cfg": feet()},
    )
    cfg.rewards["jp_drift"] = RewardTermCfg(
        func=microduck_mdp.jp_drift_penalty,
        weight=W_DRIFT,
        params={"sensor_name": sensor_name, "feet_cfg": feet(),
                "saturate_m": DRIFT_SAT_M},
    )
    # THE DECISIVE TERM, from pivot v5: a constant paid every step the episode
    # is alive with the trunk within 25 deg of vertical. Nothing to farm — no
    # shaping, no pose to seek — so it can be this large without distorting the
    # ranking of anything done WHILE alive; it changes only the value of
    # surviving to do it. 300 points over the episode, against a hop's ~130.
    cfg.rewards["alive"] = RewardTermCfg(
        func=microduck_mdp.jp_alive_reward,
        weight=W_ALIVE,
        params={"gate_tilt_above_deg": ALIVE_TILT_DEG},
    )
    # ...and the marker on the outcome it is protecting against. Only
    # `fell_over`: a `nan_state` is a sim blow-up, `time_out` is the normal end.
    cfg.rewards["terminated"] = RewardTermCfg(
        func=microduck_mdp.jp_fall_penalty,
        weight=W_TERMINATED,
        params={"term_names": ("fell_over",)},
    )
    # bow v5's pair, at bow v7's tolerance: an unpriced joint on a head carrying
    # 38% of the mass is a free counterweight, but pinned too tightly the
    # always-on income dilutes the trick.
    cfg.rewards["head_still"] = RewardTermCfg(
        func=microduck_mdp.jp_head_still_reward,
        weight=W_HEAD_STILL,
        params={"joint_indices": HEAD_STILL_JOINTS, "std": HEAD_STILL_STD},
    )
    cfg.rewards["head_joint_vel"] = RewardTermCfg(
        func=microduck_mdp.jp_head_vel_penalty,
        weight=W_HEAD_VEL,
        params={"joint_indices": HEAD_VEL_JOINTS, "vel_cap": HEAD_VEL_CAP},
    )

    # ── Terminations: INHERITED ONLY ────────────────────────────────────────
    # `fell_over` (bad_orientation at 70 deg) is the whole behavioural list, and
    # in particular there is NO HEIGHT TERMINATION: the hop's crouch bottoms at
    # 0.088 m and the reachable floor is 0.061 m, both under the bow's 0.060
    # collapse line and the lab's 0.07 fall height. `deep_squat`'s A/B says the
    # z-kill MOVES THE POLICY OFF THE GOAL POSE, so the bow's `collapsed` must
    # not be copied across.
    return cfg


MicroduckJumpRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="jump",
    run_name="jump",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=2_000,
)

# The episode is exactly 100 control steps, which every number in the tables
# above is quoted against.
assert math.isclose(EPISODE_LENGTH_S / 0.02, 100.0)
