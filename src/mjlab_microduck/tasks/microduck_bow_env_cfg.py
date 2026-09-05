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
    foot leaving the floor. Charged per foot, so a single lift already costs.
  * collapse termination: a trunk below BOW_COLLAPSE_Z is a collapse, not a
    bow — terminating stops the policy from farming the crouch reward's tail
    from a heap on the floor.
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

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# ── Reward weights ───────────────────────────────────────────────────────────
# Positive task mass ≈ 12, comparable to velocity's ~11, so the shared
# regularizers keep their relative strength (AGENTS.md: compare reward MASS,
# not weights, when copying regularizers between envs).
W_HEIGHT = 3.0        # the shape of the bow
W_PITCH = 2.0         # nose-down: what makes the crouch read as a bow
W_HEAD = 2.0          # head down, then lifted
W_STAND_POSE = 2.0    # the finish, stand phase only
W_UPRIGHT = 2.0       # lateral tilt only — the pitch is intended
W_TRACK = 0.5         # zero twist = don't travel, don't turn (light: the dip
                      # itself makes a legitimate body-frame velocity transient)
W_FOOT_LIFT = -3.0    # the hard constraint: both feet planted, always
W_DRIFT = -1.5        # a bow happens on the spot
W_SLIP = -0.2
W_ACTION_RATE = -0.2  # light and FIXED (see below)


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
            "stand_z": STAND_Z,
            "crouch_z": BOW_Z,
            "std": 0.02,   # ≈ the height error still worth pricing on a 3 cm dip
        },
    )
    cfg.rewards["bow_pitch"] = RewardTermCfg(
        func=microduck_mdp.bow_pitch_reward,
        weight=W_PITCH,
        params={
            "sensor_name": sensor_name,
            "pitch_rad": BOW_PITCH_RAD,
            "std": 0.12,   # ~7°, comfortably under the 10° target
        },
    )
    cfg.rewards["bow_head"] = RewardTermCfg(
        func=microduck_mdp.bow_head_reward,
        weight=W_HEAD,
        params={
            "sensor_name": sensor_name,
            "down_deltas": microduck_mdp.BOW_HEAD_DOWN,
            "up_deltas": microduck_mdp.BOW_HEAD_UP,
            "std": 0.30,
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
            "std": 0.12,                    # tight: this IS the standing pose
        },
    )
    cfg.rewards["foot_lift"] = RewardTermCfg(
        func=microduck_mdp.bow_foot_lift_penalty,
        weight=W_FOOT_LIFT,
        params={"sensor_name": sensor_name},
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
