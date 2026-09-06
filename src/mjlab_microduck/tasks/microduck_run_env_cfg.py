"""Microduck RUN task — go fast in a straight line, on the velocity template.

Why a separate task instead of "walk, but with a bigger vx range": the walking
recipe prices running out of its own optimum. Its `pose` reward reuses
`std_walking` for `std_running` behind a running_threshold of 1.5 m/s that this
robot can never reach (so the running regime is dead code); `upright` is tuned
at std²=0.05 specifically to erase the 2–4° forward lean that a run needs;
`air_time` (3.0) outweighs `track_linear_velocity` (2.0), so a slow, high-
stepping gait beats a fast one; and `action_rate_l2` ramps to -1.0, which is a
smoothness tax heavy enough to cap stride rate.

This cfg is a port of Pollen's unreleased GPU `Mjlab-Run` recipe (transcribed
from the community port in microduck-lab `behaviors/locomotion.py`), applied on
top of `make_microduck_velocity_env_cfg` so DR, the 61D obs contract, obs noise,
the delay stack, the NaN guard and the BAM actuator all stay in sync for free
(AGENTS.md: build on the velocity template, never standalone).

What the recipe inverts, term by term:
  - speed is the LARGEST term: track_linear_velocity 2.0 → 4.0 (std² 0.1).
  - the pose running regime actually fires: running_threshold 1.5 → 0.40 m/s
    with a genuinely looser `std_running` (sagittal joints 0.4 → 0.6).
  - lean is cheap: upright std² 0.05 → 0.12 (weight stays 2.0).
  - smoothness caps at -0.5 instead of -1.0, and ramps later (stages at
    iters 0/1000/1500/2250/3000, not 0/500/750/1000/1250/1500) — an
    attempt-tax during skill discovery makes "do nothing" win (AGENTS.md).
  - air_time window shifts up to [0.15, 0.35] s: a longer flight phase.
  - foot_clearance targets 3 cm of lift (was 2 cm) at the template's -2.0.
  - body_ang_vel -0.05 → -0.025 and angular_momentum -0.02 → -0.01: both are
    motion-BLOCKERS, and a run physically requires trunk rotation.
  - head_pose_tracking 2.0 → 3.5: the 280 g head is 38% of body mass, and at
    speed an untracked head is the difference between a run and a faceplant.

Command mix (the other half of the recipe):
  - RUN_FORWARD_FRAC (55%) of envs get a forward-only command each resample —
    vx ∈ [0.3, ceiling], vy = wz = 0 — via the base template's
    `rel_forward_envs` bucket. Straight-line running is rare under independent
    uniform sampling (the same reason turn-in-place needed its own bucket), and
    it is exactly the region this task exists to train.
  - the remaining envs stay omni (vy ±0.3, wz ±1.0), so the policy is still a
    drop-in replacement for the walk policy at the runtime's command contract.
  - turn-in-place is OFF here (the walk policy owns spinning).
  - standing envs ramp 0.02 → 0.10 by iter 1500 — lower than velocity's 0.25,
    because at this sample budget standing envs are experience not spent on
    running, but non-zero so the deployment idle state stays trained.
  - a SPEED CURRICULUM raises the forward ceiling 0.4 → 1.1 m/s over 6000
    iterations (see RUN_SPEED_STAGES). Only the upper bound of lin_vel_x moves;
    backward stays at -0.4.

Experiment knobs (module constants, each overridable by an env var so a job can
sweep them without a code change — HF Jobs pass env, not patches):
  MICRODUCK_RUN_SPEED_CEILING  final forward ceiling, default 1.1 m/s. Every
                               stage value is clipped to it, so lowering it
                               truncates the curriculum instead of breaking it.
  MICRODUCK_RUN_STAGE_SCALE    multiplier on EVERY stage boundary (speed,
                               standing, action_rate), default 1.0. >1 stretches
                               the whole schedule for a longer run.
  MICRODUCK_RUN_FORWARD_FRAC   forward-only env fraction, default 0.55.
  MICRODUCK_RUN_TURN_FRAC      turn-in-place env fraction, default 0.0.
  MICRODUCK_RUN_TRACK_ANG_W    yaw-tracking weight, default 2.0 (4.0 = 1:1 with speed).
  MICRODUCK_RUN_AIR_MIN/_MAX   air_time window in seconds, default 0.15/0.35.

Deliberate deviations from the transcribed recipe, and why:
  - `head_pose_bias` (+ its curriculum) is kept from the fork's velocity
    template. It is not in the GPU recipe, but it prices only the ESCAPABLE DC
    head droop (L1 on a 1 s EMA), costs a correct policy nothing, and dropping
    it reintroduces the measured 15° droop on a task where the head is the
    largest lever on the robot.
  - `body_pose_tracking` stays at weight 0 with its tiny command ranges: the
    61D obs contract requires the slot to stay alive (AGENTS.md invariant).
  - `dof_pos_limits` (-1.0) is kept: joint-limit safety, not a gait shaper.
"""

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

# Running is left/right symmetric, so the mirror loss is a valid prior and buys
# sample efficiency (unlike the one-footed tricks, where it must be off).
ENABLE_SYMMETRY = True


def _env_float(name: str, default: float) -> float:
    """Read a float experiment knob from the environment.

    Empty/unset falls back to `default`; a non-numeric value is a hard error
    (a typo'd knob on a paid GPU job must not silently train the default).
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


# ── Experiment knobs: DEFAULT_* is the code default, the bare name is the
# import-time resolved value (what `uv run list-envs` and tests inspect), and
# the factory re-reads the env var at call time so a job's environment always
# wins even if this module was imported earlier. ──────────────────────────────
DEFAULT_SPEED_CEILING = 1.1
DEFAULT_STAGE_SCALE = 1.0
DEFAULT_FORWARD_FRAC = 0.55
DEFAULT_AIR_MIN = 0.15
DEFAULT_AIR_MAX = 0.35

SPEED_CEILING = _env_float("MICRODUCK_RUN_SPEED_CEILING", DEFAULT_SPEED_CEILING)
STAGE_SCALE = _env_float("MICRODUCK_RUN_STAGE_SCALE", DEFAULT_STAGE_SCALE)
FORWARD_FRAC = _env_float("MICRODUCK_RUN_FORWARD_FRAC", DEFAULT_FORWARD_FRAC)
DEFAULT_TURN_FRAC = 0.0
TURN_FRAC = _env_float("MICRODUCK_RUN_TURN_FRAC", DEFAULT_TURN_FRAC)
AIR_MIN = _env_float("MICRODUCK_RUN_AIR_MIN", DEFAULT_AIR_MIN)
AIR_MAX = _env_float("MICRODUCK_RUN_AIR_MAX", DEFAULT_AIR_MAX)

# Backward/lateral/angular command ranges (fixed; only the forward ceiling is
# on a curriculum).
RUN_LIN_VEL_X_MIN = -0.4
RUN_LIN_VEL_Y = (-0.3, 0.3)
RUN_ANG_VEL_Z = (-1.0, 1.0)

# ── Stage schedules, in ITERATIONS (× NUM_STEPS_PER_ENV = env steps). ─────────
# Forward speed ceiling. 0.4 is the walk recipe's ceiling: the run starts where
# walking ended and only then pushes. The late stages past 0.6 exist because
# this robot delivers roughly two thirds of the commanded speed, so a 0.6
# command caps ACHIEVED speed near 0.4 m/s by construction.
RUN_SPEED_STAGES = (
    (0, 0.4),
    (1500, 0.45),
    (2000, 0.5),
    (2500, 0.55),
    (3000, 0.6),
    (4000, 0.7),
    (4500, 0.8),
    (5000, 0.9),
    (5500, 1.0),
    (6000, 1.1),
)
RUN_STANDING_STAGES = (
    (0, 0.02),
    (500, 0.05),
    (1000, 0.08),
    (1500, 0.10),
)
# action_rate_l2 magnitudes: -0.1 → -0.5 (velocity ramps to -1.0 by iter 1500).
RUN_ACTION_RATE_STAGES = (
    (0, 0.1),
    (1000, 0.2),
    (1500, 0.3),
    (2250, 0.4),
    (3000, 0.5),
)

# ── Reward weights (the run recipe). ──────────────────────────────────────────
W_TRACK_LIN = 4.0          # "keep_pace": the largest term in the stack
W_TRACK_ANG = 2.0
W_AIR_TIME = 3.0
W_UPRIGHT = 2.0
W_POSE = 1.0
W_HEAD_POSE = 3.5
W_FOOT_CLEARANCE = -2.0
W_FOOT_SLIP = -0.1
W_FOOT_SWING_HEIGHT = -0.25  # template value, kept
W_BODY_ANG_VEL = -0.025
W_ANGULAR_MOMENTUM = -0.01
W_SELF_COLLISIONS = -1.0     # template value, kept
W_ACTION_RATE_START = -0.1

# Tracking Gaussian variances (the cfgs take std, the recipe quotes std²).
STD2_TRACK_LIN = 0.1
STD2_TRACK_ANG = 0.5
STD2_UPRIGHT = 0.12
HEAD_POSE_STD = 0.5

# pose (variable_posture) regimes. Standing/walking match the walk recipe; the
# RUNNING regime is the new one — sagittal joints get ~1.5× the walking
# tolerance so a real stride is not taxed as a posture error.
RUN_WALKING_THRESHOLD = 0.01
RUN_RUNNING_THRESHOLD = 0.40
RUN_STD_STANDING = {
    r".*hip_yaw.*": 0.1,
    r".*hip_roll.*": 0.05,
    r".*hip_pitch.*": 0.15,
    r".*knee.*": 0.15,
    r".*ankle.*": 0.1,
}
RUN_STD_WALKING = {
    r".*hip_yaw.*": 0.3,
    r".*hip_roll.*": 0.05,
    r".*hip_pitch.*": 0.4,
    r".*knee.*": 0.4,
    r".*ankle.*": 0.25,
}
RUN_STD_RUNNING = {
    r".*hip_yaw.*": 0.4,
    r".*hip_roll.*": 0.08,
    r".*hip_pitch.*": 0.6,
    r".*knee.*": 0.6,
    r".*ankle.*": 0.4,
}

FOOT_CLEARANCE_TARGET = 0.03  # 3 cm of swing lift (walk asks 2 cm)
COMMAND_THRESHOLD = 0.01      # gate the gait terms on a live command


def _scaled_stages(stages, scale: float) -> list[tuple[int, float]]:
    """Iteration-indexed stages → env-step-indexed stages, times STAGE_SCALE."""
    return [
        (int(round(it * scale)) * NUM_STEPS_PER_ENV, value) for it, value in stages
    ]


def run_speed_stages(
    ceiling: float | None = None, scale: float | None = None
) -> list[dict]:
    """The speed curriculum as CurriculumTermCfg stages, clipped to `ceiling`.

    Clipping (not truncating) keeps the schedule shape: with a 0.8 ceiling the
    late stages simply repeat 0.8, so the final stage always equals the ceiling
    and no stage ever commands more than the experiment allows.
    """
    ceiling = SPEED_CEILING if ceiling is None else ceiling
    scale = STAGE_SCALE if scale is None else scale
    stages = list(RUN_SPEED_STAGES)
    # Above the last fixed rung the ladder keeps climbing at the same pace,
    # +0.1 m/s every 500 iterations, until it reaches the ceiling: a 1.5
    # ceiling is a real experiment, not a 1.1 run with a different label.
    while stages[-1][1] < ceiling - 1e-9:
        it, v = stages[-1]
        stages.append((it + 500, min(round(v + 0.1, 3), ceiling)))
    return [
        {"step": step, "ceiling": min(value, ceiling)}
        for step, value in _scaled_stages(tuple(stages), scale)
    ]


def make_microduck_run_env_cfg(
    play: bool = False, rough: bool = False
) -> ManagerBasedRlEnvCfg:
    """Build the run env on top of the velocity recipe."""
    # Knobs are re-read here so an HF Jobs environment wins over whatever was
    # set when this module happened to be imported.
    speed_ceiling = _env_float("MICRODUCK_RUN_SPEED_CEILING", DEFAULT_SPEED_CEILING)
    stage_scale = _env_float("MICRODUCK_RUN_STAGE_SCALE", DEFAULT_STAGE_SCALE)
    forward_frac = _env_float("MICRODUCK_RUN_FORWARD_FRAC", DEFAULT_FORWARD_FRAC)
    air_min = _env_float("MICRODUCK_RUN_AIR_MIN", DEFAULT_AIR_MIN)
    air_max = _env_float("MICRODUCK_RUN_AIR_MAX", DEFAULT_AIR_MAX)

    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # ── Commands ─────────────────────────────────────────────────────────────
    command = cfg.commands["twist"]
    # Start at the walk ceiling; the speed curriculum below raises the top.
    command.ranges.lin_vel_x = (RUN_LIN_VEL_X_MIN, min(0.4, speed_ceiling))
    command.ranges.lin_vel_y = RUN_LIN_VEL_Y
    command.ranges.ang_vel_z = RUN_ANG_VEL_Z
    command.rel_standing_envs = RUN_STANDING_STAGES[0][1]
    # Forward-only bucket, from the base template: |vx| clamped to ≥ 0.3 with
    # vy = wz = 0. (Because it re-uses the lin_vel_x sample, low draws pile up
    # at exactly 0.3 — that is the intended "always a real forward command",
    # and the ceiling stage is what moves the top of the bucket.)
    command.rel_forward_envs = forward_frac
    # Turn-in-place bucket (lin = 0, |wz| forced to [0.4*max, max]). The
    # recipe had 0; the office found the run policy could not turn on the
    # spot and fell there, so it is a knob (MICRODUCK_RUN_TURN_FRAC).
    command.rel_turn_in_place_envs = _env_float("MICRODUCK_RUN_TURN_FRAC", DEFAULT_TURN_FRAC)

    # ── Rewards ──────────────────────────────────────────────────────────────
    # Speed is the biggest term in the stack.
    cfg.rewards["track_linear_velocity"].weight = W_TRACK_LIN
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(STD2_TRACK_LIN)
    # Yaw was priced at half of speed (2:1) and collected 7% of its weight: a
    # half-rad/s DC yaw drift was nearly free (docs/research/yaw-asymmetry.md).
    # MICRODUCK_RUN_TRACK_ANG_W raises it; 4.0 puts it back at 1:1 with speed.
    cfg.rewards["track_angular_velocity"].weight = _env_float("MICRODUCK_RUN_TRACK_ANG_W", W_TRACK_ANG)
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(STD2_TRACK_ANG)

    # Lean is cheap: a run leans, and the walk recipe's std²=0.05 erases it.
    cfg.rewards["upright"].weight = W_UPRIGHT
    cfg.rewards["upright"].params["std"] = math.sqrt(STD2_UPRIGHT)

    # Longer flight phase, still gated on a live command.
    cfg.rewards["air_time"].weight = W_AIR_TIME
    cfg.rewards["air_time"].params["command_threshold"] = COMMAND_THRESHOLD
    cfg.rewards["air_time"].params["threshold_min"] = air_min
    cfg.rewards["air_time"].params["threshold_max"] = air_max

    # pose: the running regime becomes reachable and genuinely looser.
    cfg.rewards["pose"].weight = W_POSE
    cfg.rewards["pose"].params["std_standing"] = RUN_STD_STANDING
    cfg.rewards["pose"].params["std_walking"] = RUN_STD_WALKING
    cfg.rewards["pose"].params["std_running"] = RUN_STD_RUNNING
    cfg.rewards["pose"].params["walking_threshold"] = RUN_WALKING_THRESHOLD
    cfg.rewards["pose"].params["running_threshold"] = RUN_RUNNING_THRESHOLD

    # Head tracking matters more at speed than at walk.
    cfg.rewards["head_pose_tracking"].weight = W_HEAD_POSE
    cfg.rewards["head_pose_tracking"].params["std"] = HEAD_POSE_STD

    # Feet: 3 cm of lift, weak slip cost, template swing-height cost.
    cfg.rewards["foot_clearance"].weight = W_FOOT_CLEARANCE
    cfg.rewards["foot_clearance"].params["target_height"] = FOOT_CLEARANCE_TARGET
    cfg.rewards["foot_clearance"].params["command_threshold"] = COMMAND_THRESHOLD
    cfg.rewards["foot_slip"].weight = W_FOOT_SLIP
    cfg.rewards["foot_slip"].params["command_threshold"] = COMMAND_THRESHOLD
    cfg.rewards["foot_swing_height"].weight = W_FOOT_SWING_HEIGHT

    # Motion blockers stay LOW for a dynamic task (AGENTS.md).
    cfg.rewards["body_ang_vel"].weight = W_BODY_ANG_VEL
    cfg.rewards["angular_momentum"].weight = W_ANGULAR_MOMENTUM
    cfg.rewards["self_collisions"].weight = W_SELF_COLLISIONS

    # Smoothness starts gentle; the curriculum below caps it at -0.5.
    cfg.rewards["action_rate_l2"].weight = W_ACTION_RATE_START

    # ── Curricula ────────────────────────────────────────────────────────────
    # Speed ceiling: raises ONLY the upper bound of lin_vel_x, which is also the
    # top of the forward-only bucket.
    cfg.curriculum["speed_ceiling"] = CurriculumTermCfg(
        func=microduck_mdp.speed_ceiling_curriculum,
        params={
            "command_name": "twist",
            "speed_stages": run_speed_stages(speed_ceiling, stage_scale),
        },
    )

    # Later, gentler smoothness ramp than the walk recipe's.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": step, "weight": -value}
                for step, value in _scaled_stages(RUN_ACTION_RATE_STAGES, stage_scale)
            ],
        },
    )

    # Standing envs top out at 0.10, not the walk recipe's 0.25.
    cfg.curriculum["standing_envs"] = CurriculumTermCfg(
        func=microduck_mdp.standing_envs_curriculum,
        params={
            "command_name": "twist",
            "standing_stages": [
                {"step": step, "rel_standing_envs": value}
                for step, value in _scaled_stages(RUN_STANDING_STAGES, stage_scale)
            ],
        },
    )

    return cfg


MicroduckRunRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="run",
    run_name="run",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    # The speed curriculum's last stage lands at iter 6000; 8000 leaves room to
    # consolidate the top of the ladder.
    max_iterations=8_000,
)
