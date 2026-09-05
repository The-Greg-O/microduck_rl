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
    violence past ~1 turn/s) and capped in total at one turn (no over-spinning).
  - `spin_complete` is ONE step, not a per-step "you are done" — that would be
    the jackpot the playbook warns about and would buy a ballistic whip.
  - Speed is bought by the SETTLE phase instead: finishing early leaves more of
    the 4 s episode collecting settle pay, and settle only pays while upright,
    stopped and back in the STAND pose. Flopping after a fast whip scores ~0.
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

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# Reward weights. The arithmetic that makes "spin fast" the argmax (dt = 0.02,
# 200 steps): `spin_progress` integrates to a FIXED W_PROGRESS/dt over a
# completed turn however fast it is done, so the tie is broken by the per-step
# stacks — ~5/step of context pay while spinning against ~9/step once settled.
# Finishing at t=1 s scores ~1800, at t=2 s ~1600, never spinning ~1000, and
# whipping round then falling ~450 (the settle/upright/height stack all gate on
# upright, and `fell_over` terminates the rest of the episode).
W_PROGRESS = 4.0
W_COMPLETE = 5.0
W_SETTLE = 3.0
W_HEIGHT = 1.0
W_TRACK_LIN = 1.0      # velocity's 2.0, halved: stillness is context, not the task
W_DRIFT = -1.5
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
