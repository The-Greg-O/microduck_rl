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
  * `_pv_update` latches that sign for the episode and derives the pin foot
    from it — the pin is the INSIDE foot of the turn, so +1 (counter-clockwise,
    turning left) pivots on the LEFT foot and -1 on the right;
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
    violence past ~1 turn/s) and capped in total at one turn.
  * `pivot_complete` is ONE step, not a per-step "you are done" — that would be
    the jackpot the playbook warns about and would buy a ballistic whip. Speed
    is bought by the SETTLE phase instead: finish early and more of the 4 s is
    spent collecting settle pay, which only pays while upright, stopped, on
    both feet and back in the STAND pose.
  * `pin` is what makes this a PIVOT and not a spin, and it is weighted to say
    so. Per step during the turn a pivot collects pin 3.0 + paddle 1.0 ≈ 4.0
    while a two-footed spin collects ~0 of the pin pay AND pays the two pin
    costs (-2.0 displacement, -0.5 slip): a ~5.5/step gap on an otherwise
    identical progress score, over ~150 steps. Nothing subtle about it.
  * `paddle` is tippy-taps' tap term with the roles split: it pays the FREE
    foot for being mid-stroke, in an air-time window, while the pin is planted
    and the turn is unfinished. Air time resets on contact, so every paid
    window has to be bought with a fresh plant — that is the alternation — and
    a foot parked in the air runs past the window and stops paying (that would
    be a flamingo).
  * `settle` is a PRODUCT (stopped × pose × both feet down), not a sum: an
    additive stack has a compromise basin where a still-turning one-legged
    crouch keeps most of it.
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
# The arithmetic (dt = 0.02, 200 steps). `pivot_progress` integrates to a FIXED
# W_PROGRESS/dt = 200 over a completed turn however fast it is done, so the tie
# is broken by the per-step stacks:
#   pivoting   pin 3.0 + paddle ~1.0 + height 1.0 + upright 2.0   ≈ 7/step
#   settled    pin 3.0 + settle 4.0 + height 1.0 + upright 2.0    ≈ 10/step
#   spinning   pin ~0  + paddle 0    + height 1.0 + upright 2.0
#              − displacement 2.0 − slip 0.5                      ≈ 0.5/step
#   standing   pin 3.0 + height 1.0 + upright 2.0, no progress    ≈ 6/step
# So: turning beats standing, finishing early beats dawdling, and pivoting
# beats spinning by ~6.5/step over the ~150 steps a turn takes.
W_PROGRESS = 4.0
W_COMPLETE = 5.0
W_PIN = 3.0
W_SETTLE = 4.0
W_PADDLE = 1.0
W_HEIGHT = 1.0
W_PIN_DISPLACEMENT = -2.0
W_PIN_SLIP = -0.5
W_RADIUS = -1.5
W_TILT = -0.5
W_ACTION_RATE = -0.05


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
        params={**shared(), "pin_std": microduck_mdp.PV_PIN_STD_M},
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
        weight=W_PIN_DISPLACEMENT,
        params={**shared(), "saturate_m": microduck_mdp.PV_PIN_SAT_M},
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
