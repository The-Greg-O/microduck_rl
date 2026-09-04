"""Microduck TIPPY TAPS task — the excited-dog dance.

Episodic skill: from a standing start, quick alternating foot lifts on the
spot, head up (and commandable), no travel, back to a stand. Deploys as a
constant-command `episodic` skill (`robotctl policy add tippy-taps <repo>`),
duration ≈ EPISODE_LENGTH_S, bindable to a gamepad button and — the point —
triggerable by a future social layer when a friend duck or a person shows up.

Prototyped first in microduck-lab (`behaviors/tippy_taps.py`, branch `go`);
this is the sim2real port: same terms, on top of the velocity recipe so DR,
observation noise, delays, the BAM actuator and the 61D obs contract come
for free (AGENTS.md: build on the velocity template, don't start standalone).

Reward design, in the playbook's terms:
  - tap / switch_feet are per-step, bounded, and gated on a contact-sensor
    air-time WINDOW (0.04–0.25 s): a lift must break contact for real, and a
    hold is not a tap. No jackpots.
  - idle is a COST on time since the last real lift (grace 0.3 s): the
    anti-"learn to stand" pressure the plain walk lacks at zero command.
  - hop is a COST on both feet airborne, not charged before first touchdown.
  - velocity tracking at a near-zero command prices drift SPEED; drift prices
    POSITION (a slow shuffle beats a speed cost — lab lesson).
  - leg pose std is loose (0.5): lifting a leg is the task.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

# Alternating taps are left/right symmetric: the mirror loss is a valid prior
# and buys sample efficiency (unlike one-footed tricks, where it must be off).
ENABLE_SYMMETRY = True

EPISODE_LENGTH_S = 4.0
STAND_Z = 0.115  # measured standing trunk z at HOME (standup env)

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# Reward weights (task mass ≈ 5 on the positive side, comparable to velocity's
# tracking + air_time stack, so the shared regularizers keep their relative
# strength).
W_TAP = 3.0
W_SWITCH = 2.0
W_IDLE = -1.0
W_HOP = -2.0
W_DRIFT = -1.5
W_TRACK = 1.0          # velocity's 2.0, reduced: stillness is context, not the task
W_POSE_LEGS = 1.0
W_HEIGHT = 1.0


def make_microduck_tippy_taps_env_cfg(
    play: bool = False, rough: bool = False
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Command: pinned near zero (obs-shape parity; the skill is untriggered
    # by command, being selected IS the trigger, like the kicks) ─────────────
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

    # ── Rewards: drop the gait terms ──────────────────────────────────────────
    for name in ("air_time", "foot_clearance", "foot_swing_height", "foot_slip", "pose"):
        if name in cfg.rewards:
            del cfg.rewards[name]
    # Tracking a ~zero twist = "trunk stays still", kept at reduced weight.
    cfg.rewards["track_linear_velocity"].weight = W_TRACK
    cfg.rewards["track_angular_velocity"].weight = W_TRACK

    sensor_name = "feet_ground_contact"
    cfg.rewards["tap"] = RewardTermCfg(
        func=microduck_mdp.tt_tap_reward,
        weight=W_TAP,
        params={"sensor_name": sensor_name},
    )
    cfg.rewards["switch_feet"] = RewardTermCfg(
        func=microduck_mdp.tt_switch_reward,
        weight=W_SWITCH,
        params={"sensor_name": sensor_name},
    )
    cfg.rewards["idle"] = RewardTermCfg(
        func=microduck_mdp.tt_idle_penalty,
        weight=W_IDLE,
        params={"sensor_name": sensor_name},
    )
    cfg.rewards["hop"] = RewardTermCfg(
        func=microduck_mdp.tt_hop_penalty,
        weight=W_HOP,
        params={"sensor_name": sensor_name},
    )
    cfg.rewards["drift"] = RewardTermCfg(
        func=microduck_mdp.tt_drift_penalty,
        weight=W_DRIFT,
        params={"sensor_name": sensor_name},
    )
    # Legs near HOME between taps, loosely: a lift is a big transient deviation
    # and must stay affordable (ball-kick precedent, std 0.5).
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=W_POSE_LEGS,
        params={"std": 0.5, "joint_indices": _LEG_JOINTS, "target_overrides": None},
    )
    # Trunk at standing height: taps, not a squat-and-shuffle.
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=W_HEIGHT,
        params={
            "std": 0.04,
            "target_height": STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    return cfg


MicroduckTippyTapsRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="tippy_taps",
    run_name="tippy_taps",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=4_000,
)
