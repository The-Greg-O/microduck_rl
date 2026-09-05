"""Microduck HAPPY SPIN task — the excited-dog twirl.

Episodic skill: from a standing start, ONE fast full turn (360°) on the spot,
then stand still, all inside a 4 s episode. Deploys as a constant-command
`episodic` skill (`robotctl policy add spin <repo>`), duration ≈
EPISODE_LENGTH_S, bindable to a gamepad button — the daemon holds the twist
command at zero for the whole skill window, so the skill takes no command:
being selected IS the trigger (as for the kicks and tippy-taps).

Built on the velocity recipe so DR, observation noise, delays, the BAM
actuator and the 61D obs contract come for free (AGENTS.md: build on the
velocity template, don't start standalone). Same shape as tippy-taps: pinned
command, short episode, one per-env memory ticked by every reward term.

DIRECTION IS FIXED: counter-clockwise (+z yaw rate). Both directions would be
the nicer skill, but the 61D observation contract has no free slot to tell the
policy which way to turn this episode, so a per-env random sign would be
unobservable and unlearnable. A clockwise twin is a separate policy (flip
`HS_DIRECTION`), or a future contract revision.

Why the walk model and not the roller `Mjlab-Spin-Flat-MicroDuck`: that task is
a different animal — rollers, a cyclic phase command, 20 s episodes, driven by
the ground-pick slot. This one is legged, episodic, and command-free.

Reward design, in the playbook's terms:
  - `spin_progress` pays Δ(accumulated yaw) — potential-based, so holding
    still pays zero and there is nothing to farm; capped per step (no pay for
    violence past ~1 turn/s) and capped in total at one turn (no pay for
    over-spinning — but see `overshoot`: no pay is not the same as a cost).
  - `spin_complete` is ONE step, not a per-step "you are done" — that would be
    the jackpot the playbook warns about and would buy a ballistic whip. It
    fires a few degrees SHORT of the turn rather than on crossing it, so the
    settle stack is already paying while the last degrees are turned and a
    controlled arrival beats a fast crossing.
  - `heading` is the term v1 was missing: a Gaussian (std 10°) on the wrapped
    heading error against the STARTING heading, settle phase only. v1 spun a
    clean 360° and then coasted to 415–457°, because nothing in the stack ever
    said where to stop — settle pays for stillness facing any direction at all.
  - `overshoot` is its coarse partner: a bounded ramp on the degrees turned
    past 2π, saturating at 90°, reading the RAW yaw integral (the progress
    potential clamps at 2π, which is exactly what hid the over-rotation). The
    Gaussian supplies the peak, the ramp supplies the gradient a 55°-over
    policy can actually feel.
  - Speed is bought by the SETTLE phase instead: finishing early leaves more of
    the 4 s episode collecting settle + heading pay, and both only pay while
    upright, stopped and back in the STAND pose. Flopping after a fast whip
    scores ~0 — and so, now, does whipping past the mark.
  - `settle` is a PRODUCT of Gaussians (stopped × pose), not a sum: an additive
    pair has a compromise basin where a still-turning crouch keeps most of it.
  - `angular_momentum` is DELETED (its 3D norm fights the trick directly) and
    so is `track_angular_velocity` (the command is pinned to ~0 yaw rate, so it
    would pay the policy for NOT spinning). `body_ang_vel` stays: it is x/y
    only, so it mates roll/pitch thrash without touching yaw.
  - `foot_slip` is DELETED: a spin slides. That is the maneuver, not a fault.
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

# A spin has a HANDEDNESS: the left/right mirror of a counter-clockwise turn is
# a clockwise turn, so the mirror loss would be a false prior here. OFF.
ENABLE_SYMMETRY = False

EPISODE_LENGTH_S = 4.0
STAND_Z = 0.115  # measured standing trunk z at HOME (standup env)

HS_DIRECTION = microduck_mdp.HS_DIRECTION      # +1 = counter-clockwise
HS_TARGET_YAW = microduck_mdp.HS_TARGET_YAW    # 2π, one full turn
HS_COMPLETE_TOL = microduck_mdp.HS_COMPLETE_TOL  # bonus fires within 5° of the turn
HS_HEADING_STD = microduck_mdp.HS_HEADING_STD    # 10° heading Gaussian
HS_OVERSHOOT_SAT = microduck_mdp.HS_OVERSHOOT_SAT  # over-rotation cost saturates at 90°

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# Reward weights. The arithmetic that makes "spin fast" the argmax (dt = 0.02,
# 200 steps): `spin_progress` integrates to a FIXED W_PROGRESS/dt over a
# completed turn however fast it is done, so the tie is broken by the per-step
# stacks — ~5/step of context pay while spinning against ~9/step once settled.
# Finishing at t=1 s scores ~1800, at t=2 s ~1600, never spinning ~1000, and
# whipping round then falling ~450 (the settle/upright/height stack all gate on
# upright, and `fell_over` terminates the rest of the episode).
#
# v1 MEASURED (happy-spin-v1, ckpt 1999, BAM, noise + DR on, 3 seeds): the spin
# works — 360° at 0.74–0.88 s, no falls, drift < 8 cm — but TOTAL yaw came out
# 415 / 457 / 415°, i.e. it settles 55–100° PAST its starting heading. Every
# term above is blind to that: the progress potential clamps at 2π (so the 55th
# degree past pays zero, but costs zero too), the completion bonus is one-shot
# and fires on CROSSING (indifferent to the yaw rate you cross at, and the
# cheapest crossing is a fast one), and `settle` pays for a zero yaw rate in the
# STAND pose FACING ANYWHERE. Over-rotation was free. Two terms price it:
#
#   `heading`   W_HEADING × exp(-(wrapped heading error / 10°)²), settle phase,
#               the peak: stopping ON the starting heading.
#   `overshoot` W_OVERSHOOT × min(1, raw yaw past 2π / 90°), the coarse ramp the
#               current policy can actually feel, reading the RAW integral (the
#               clamped potential is exactly what hid the overshoot).
#
# The arithmetic, summed over a real 4 s episode at the measured completion time
# t ≈ 0.8 s (40 steps of turn, 160 settle steps; only the two NEW terms — they
# are the whole delta from v1's stack, and are what
# `test_stopping_on_the_mark_beats_crossing_fast_and_drifting` computes):
#
#   final heading   heading pay   overshoot cost      net
#     360° (mark)        +640            −0          +640
#     370° (+10°)        +249           −35          +215
#     415° (+55°)         +11          −184          −173   ← v1's best seed
#     457° (+97°)          +5          −302          −297   ← v1's worst seed
#
# So "stop at 360 ± 10°" beats "cross fast and drift to 420" by 390 (the
# tightest pairing) to 940 (dead-on vs the worst seed), on an episode that
# totals ~1800: over-rotation goes from free to a fifth or a half of the
# episode return. Both terms stay bounded, so completing the turn (worth
# W_PROGRESS/dt = 200 in progress alone, plus the whole settle stack) is never
# in question — "don't spin" still scores ~1000 against ~2400.
#
# And the fix does not simply buy slowness: the settle stack is now 7/step
# (settle 3 + heading 4), so each 0.1 s shaved off the turn is still worth
# 5 × 7 = 35 — but crossing fast and landing 55° over costs ~400. Spending
# 0.2 s decelerating into the mark wins by an order of magnitude. The
# completion bonus firing 5° EARLY (HS_COMPLETE_TOL) is what opens that
# window: heading and overshoot are live while the last degrees are still
# being turned, and `spin_progress` keeps paying right through 360°.
W_PROGRESS = 4.0
W_COMPLETE = 5.0
W_SETTLE = 3.0
W_HEADING = 4.0
W_HEIGHT = 1.0
W_TRACK_LIN = 1.0      # velocity's 2.0, halved: stillness is context, not the task
W_DRIFT = -1.5
W_OVERSHOOT = -2.0
W_TILT = -0.5


def make_microduck_happy_spin_env_cfg(
    play: bool = False, rough: bool = False
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Command: pinned near zero (obs-shape parity; the daemon sends twist
    # zero for the whole skill window, so the skill takes no command) ─────────
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

    # The standing-fraction curriculum drives a command we no longer vary.
    if "standing_envs" in cfg.curriculum:
        del cfg.curriculum["standing_envs"]

    # ── Rewards: drop the gait terms and everything that fights a spin ────────
    # foot_slip: a spin slides, by construction.
    # angular_momentum: 3D norm, fights the trick head-on.
    # track_angular_velocity: the command is pinned to ~0 yaw rate, so keeping
    #   it would pay the policy for standing still — the exact opposite term.
    for name in (
        "air_time", "foot_clearance", "foot_swing_height", "foot_slip", "pose",
        "angular_momentum", "track_angular_velocity",
    ):
        if name in cfg.rewards:
            del cfg.rewards[name]
    # A zero LINEAR command = "the trunk does not travel"; kept, reduced.
    cfg.rewards["track_linear_velocity"].weight = W_TRACK_LIN

    cfg.rewards["spin_progress"] = RewardTermCfg(
        func=microduck_mdp.hs_progress_reward,
        weight=W_PROGRESS,
        params={"direction": HS_DIRECTION, "target_yaw": HS_TARGET_YAW},
    )
    cfg.rewards["spin_complete"] = RewardTermCfg(
        func=microduck_mdp.hs_complete_bonus,
        weight=W_COMPLETE,
        params={"direction": HS_DIRECTION, "target_yaw": HS_TARGET_YAW},
    )
    cfg.rewards["settle"] = RewardTermCfg(
        func=microduck_mdp.hs_settle_reward,
        weight=W_SETTLE,
        params={
            "direction": HS_DIRECTION,
            "target_yaw": HS_TARGET_YAW,
            # rate_std 1.0 rad/s: "stopped" for a robot that was just doing
            # ~2π rad/s, wide enough that the first policy to finish a turn
            # scores visibly (invisible gradients change nothing).
            "rate_std": 1.0,
            # pose_std 0.4: the legs come back near HOME, loosely — recovering
            # from a spin is a big transient and must stay affordable.
            "pose_std": 0.4,
            "joint_indices": _LEG_JOINTS,
        },
    )
    # WHERE it stops, not just that it stops. Gated on the completion latch,
    # so the wrapped error is unambiguous: "one turn short" is unreachable
    # without having completed the turn first.
    cfg.rewards["heading"] = RewardTermCfg(
        func=microduck_mdp.hs_heading_reward,
        weight=W_HEADING,
        params={
            "direction": HS_DIRECTION,
            "target_yaw": HS_TARGET_YAW,
            "std": HS_HEADING_STD,
        },
    )
    # The cost of the degrees past the turn, read off the RAW integral: the
    # progress potential is clamped at 2π, which is precisely what made
    # over-rotation free. Bounded, so it cannot outweigh completing the turn.
    cfg.rewards["overshoot"] = RewardTermCfg(
        func=microduck_mdp.hs_overshoot_penalty,
        weight=W_OVERSHOOT,
        params={
            "direction": HS_DIRECTION,
            "target_yaw": HS_TARGET_YAW,
            "saturate_rad": HS_OVERSHOOT_SAT,
        },
    )
    # Trunk at standing height, both phases: a twirl, not a squat-and-scoot.
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=W_HEIGHT,
        params={
            "std": 0.04,
            "target_height": STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["drift"] = RewardTermCfg(
        func=microduck_mdp.hs_drift_penalty,
        weight=W_DRIFT,
        params={"direction": HS_DIRECTION, "target_yaw": HS_TARGET_YAW},
    )
    cfg.rewards["tilt"] = RewardTermCfg(
        func=microduck_mdp.hs_tilt_penalty,
        weight=W_TILT,
        params={"direction": HS_DIRECTION, "target_yaw": HS_TARGET_YAW},
    )

    # Action smoothness stays LIGHT: the velocity recipe ramps this to -1.0 by
    # iter 1500, which is a motion-blocker for a 1 s whole-body rotation, and
    # an attempt-tax during skill discovery makes "do nothing" win (AGENTS.md).
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.05},
                {"step": 750 * 24, "weight": -0.10},
                {"step": 1500 * 24, "weight": -0.20},
            ],
        },
    )
    return cfg


MicroduckHappySpinRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="happy_spin",
    run_name="happy_spin",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=2_000,
)
