import os
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab_microduck.actuator import (
    BacklashEncoderBamActuatorCfg,
    FrictionDRBamActuatorCfg,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "microduck"

MICRODUCK_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"
# Ground-contact model (formerly "allcollisions"): curated collision set for
# the parts that touch the floor in body-on-ground tasks (soles, legs, trunk
# shells, head shells, jaw, battery, hips) — NOT every geom. Shared by
# standup / ground-pick / sitstand / roulade / walk-rollers tasks.
MICRODUCK_GROUNDCONTACT_XML: Path = _ROBOT_DIR / "robot_groundcontact.xml"
# TRUE all-collisions model: every part carries a collision geom (70 geoms,
# 37 meshes; power_support demoted to self_collision_only like every variant).
# No task uses it yet — exported 2026-09 for future envs needing full contact.
MICRODUCK_ALLCOLLISIONS_XML: Path = _ROBOT_DIR / "robot_allcollisions.xml"
# 70mm / 15g ball prop for the BallKick task.
MICRODUCK_BALL_XML: Path = _ROBOT_DIR / "ball.xml"
# Roller-skate model: 14 actuated joints + passive wheel hinges (passive_*wheel).
MICRODUCK_GROUNDCONTACT_ROLLERS_XML: Path = _ROBOT_DIR / "robot_groundcontact_rollers.xml"
# Backlash models: every servo joint gets an unactuated passive_<joint>_backlash
# hinge in series (±1° play, 2° total). Exported via
# config_mjcf_{groundcontact,walk}_backlash.json (add_backlash.py post-processor).
MICRODUCK_GROUNDCONTACT_BACKLASH_XML: Path = _ROBOT_DIR / "robot_groundcontact_backlash.xml"
MICRODUCK_WALK_BACKLASH_XML: Path = _ROBOT_DIR / "robot_walk_backlash.xml"
MICRODUCK_GROUNDCONTACT_ROLLERS_BACKLASH_XML: Path = _ROBOT_DIR / "robot_groundcontact_rollers_backlash.xml"

assert MICRODUCK_WALK_XML.exists(), f"XML not found: {MICRODUCK_WALK_XML}"
assert MICRODUCK_GROUNDCONTACT_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_XML}"
assert MICRODUCK_ALLCOLLISIONS_XML.exists(), f"XML not found: {MICRODUCK_ALLCOLLISIONS_XML}"
assert MICRODUCK_BALL_XML.exists(), f"XML not found: {MICRODUCK_BALL_XML}"
assert MICRODUCK_GROUNDCONTACT_ROLLERS_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_ROLLERS_XML}"
assert MICRODUCK_GROUNDCONTACT_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_BACKLASH_XML}"
assert MICRODUCK_WALK_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_WALK_BACKLASH_XML}"
assert MICRODUCK_GROUNDCONTACT_ROLLERS_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_ROLLERS_BACKLASH_XML}"


# --- The shell (go-grgs ADR 0011 / docs/specs/shell.md) ----------------------
# add_shell.py injects world-collision primitives into the walk models
# (shell_trunk, shell_neck, shell_head, shell_thigh_left/right,
# shell_shank_left/right) so a policy can feel a wall, and gives the shell AND
# the feet conaffinity 0: they touch the world (default contype/conaffinity
# 1/1) and can never touch another robot geom, so the self_collision subtree
# sensor cannot see them. MICRODUCK_NO_SHELL=1 strips the shell at load and
# puts the feet back on conaffinity 1, reproducing the pre-shell model exactly.
# Read once at import — reload this module if you change the env var in a test.
NO_SHELL: bool = os.environ.get("MICRODUCK_NO_SHELL", "0") == "1"
SHELL_GEOM_PREFIX = "shell_"


def strip_shell(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Delete the shell geoms and give the feet their conaffinity back."""
    for geom in list(spec.geoms):
        if geom.name.startswith(SHELL_GEOM_PREFIX):
            spec.delete(geom)
    for geom in spec.geoms:
        if geom.name.endswith("_collision"):
            geom.conaffinity = 1
    return spec


def _maybe_strip_shell(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    return strip_shell(spec) if NO_SHELL else spec


def get_walk_spec() -> mujoco.MjSpec:
    return _maybe_strip_shell(mujoco.MjSpec.from_file(str(MICRODUCK_WALK_XML)))


def get_standup_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_XML))


def get_ground_pick_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_XML))


def get_walk_rollers_spec() -> mujoco.MjSpec:
    # NOTE: was loading robot_groundcontact.xml (no wheels) — the roller env
    # silently ran on the wheel-less standup model.
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_ROLLERS_XML))


def get_allcollisions_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))


def get_ball_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_BALL_XML))


def get_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_BACKLASH_XML))


def get_walk_backlash_spec() -> mujoco.MjSpec:
    return _maybe_strip_shell(mujoco.MjSpec.from_file(str(MICRODUCK_WALK_BACKLASH_XML)))


def get_rollers_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_ROLLERS_BACKLASH_XML))


HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        # Lower body — STAND2 pose: trunk shifted ~5mm forward over the feet so
        # the CoM sits over the ankle axis (was ~5mm behind it at the old HOME,
        # which biased the robot backward and made the standup policy droop its
        # head forward as a counterweight). Leg pitch chain leaned forward:
        # hip_pitch 30°→26.24°, ankle 30°→25.95°, knee 0°→0.28°. Matches the
        # STAND keyframe in scene.xml / scene_walk.xml.
        r".*hip_yaw.*": 0.0,
        r".*left_hip_roll.*": -0.0873,
        r".*right_hip_roll.*": 0.0873,
        r".*left_hip_pitch.*": -0.4579,
        r".*right_hip_pitch.*": 0.4579,
        r".*left_knee.*": -0.0049,
        r".*right_knee.*": 0.0049,
        r".*left_ankle.*": 0.4530,
        r".*right_ankle.*": -0.4530,
        # Head
        r".*neck_pitch.*": 0.3491,
        r".*head_pitch.*": 0.3491,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

# Shelled walk models only. CollisionCfg REWRITES contype/conaffinity on every
# geom it matches (and disables every named geom it does not), so the XML's
# conaffinity="0" is not enough — it has to be repeated here or mjlab would put
# the feet back on 1 and delete the shell. conaffinity=0 on the whole matched
# set is the ADR 0011 rule: world contact only, no robot-to-robot pair.
# `.*_collision` keeps its condim/priority/friction table untouched; the shell
# is a separate pattern so it never falls into the feet's rules or into the
# `feet_ground_contact` sensor's `^(left|right)_foot_collision$` match.
SHELL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision", r"^shell_.*"],
    contype=1,
    conaffinity=0,
    condim={r"^(left|right)_foot_collision$": 3, r"^shell_.*": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

# Under MICRODUCK_NO_SHELL=1 the spec has no shell geoms left, so the walk
# models fall back to the pre-shell collision cfg byte for byte.
WALK_COLLISION = FULL_COLLISION if NO_SHELL else SHELL_COLLISION

# -- Old actuator (XML position, MuJoCo built-in PD + friction) --
# actuators = DelayedActuatorCfg(
    # delay_min_lag=0,
    # delay_max_lag=3,
    # base_cfg=XmlPositionActuatorCfg(joint_names_expr=(r".*",)),
# )

# -- BAM M6 actuator (full voltage control + load-dependent friction) --
# Exclude passive_* joints (jaw linkage in the new model has no XML actuator).
# Voltage domain randomization (mirrors mjlab_microban):
#   - vin_range: per-env battery voltage sampled at startup (replaces fixed vin)
#   - vin_drop_gain_range: load-dependent voltage sag V_drop = gain * sum(|tau|)
#   - vin_min: hard floor on the effective voltage after sag
# kp_fw kept at 200 (microduck's preserved firmware stiffness; microban uses 125).
_BAM_ACTUATOR_KWARGS = dict(
    motor_name="xl330",
    model="m6",
    target_names_expr=(r"^(?!passive_).*",),
    kp_fw=200.0,  # microduck's preserved firmware stiffness (microban uses 125)
    # vin_range=(6.9, 7.9),
    vin_range=(6.5, 8.2),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=6.0,
    # max_current=1.75,
    delay_min_lag=3,
    delay_max_lag=6,
)
actuators = FrictionDRBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)

# Same BAM actuator, but the firmware position loop reads the encoder THROUGH
# the passive_<joint>_backlash hinges (the real encoder is on the output side
# of the gear play). Only for the backlash model; the target regex already
# excludes the passive_* backlash joints from actuation.
backlash_actuators = BacklashEncoderBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)

# -- BAM M4 actuator
# actuators = DelayedActuatorCfg(
    # delay_min_lag=0,
    # delay_max_lag=3,
    # base_cfg=make_bam_m4_actuator_cfg(),
# )

# HOME frame for the backlash model. HOME_FRAME's unanchored patterns
# (e.g. r".*left_hip_roll.*") would also match passive_left_hip_roll_backlash
# and try to initialize it at -0.0873 rad — outside its ±1° range. Pattern
# matching is first-match-wins in declaration order, so the anchored backlash
# rule placed FIRST pins every backlash joint at 0 and the servo joints fall
# through to the normal HOME values.
BACKLASH_HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={r".*_backlash$": 0.0, **HOME_FRAME.joint_pos},
    joint_vel={".*": 0.0},
)

MICRODUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=HOME_FRAME,
    collisions=(WALK_COLLISION,),  # shelled model — see SHELL_COLLISION
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_STANDUP_ROBOT_CFG = EntityCfg(
    spec_fn=get_standup_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_GROUND_PICK_ROBOT_CFG = EntityCfg(
    spec_fn=get_ground_pick_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Backlash robots: base model + ±1° serial backlash hinge per servo.
# Encoder reads through the backlash (BacklashEncoderBamActuator feedback +
# joint_pos/vel_rel_backlash observations — see tasks/backlash.py).
# Groundcontact variant → VelStand/StandUp backlash tasks (mirrors
# MICRODUCK_STANDUP_ROBOT_CFG); walk variant → Velocity backlash
# tasks (mirrors MICRODUCK_WALK_ROBOT_CFG, keeps backlash-vs-base comparisons
# unconfounded by the collision model).
MICRODUCK_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_WALK_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(WALK_COLLISION,),  # shelled model — mirrors the base walk cfg
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Roller-skate backlash robot: wheels stay free (passive_*wheel untouched by
# add_backlash.py). collisions=() mirrors MICRODUCK_WALK_ROLLERS_ROBOT_CFG —
# roller wheel collision geoms have no explicit names; XML defaults apply.
MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_rollers_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Free-floating, non-articulated ball prop for the BallKick task. Position is
# set each episode by the reset_ball_in_front_of_foot event; the init pos here
# only matters for the pristine pre-first-reset state.
MICRODUCK_BALL_CFG = EntityCfg(
    spec_fn=get_ball_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.0, 0.035)),
)

# Roller skate robot: the 4 passive wheel joints (passive_*wheel) have no XML
# actuators; the BAM cfg's target regex already excludes them, so the action
# space stays 14-dimensional. Uses the SAME canonical BAM actuator as every
# other variant (was a plain XmlActuatorCfg PD — an actuator-physics mismatch
# vs the rest of the family, and joint-friction DR was impossible).
MICRODUCK_WALK_ROLLERS_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_rollers_spec,
    init_state=HOME_FRAME,
    collisions=(),  # roller wheel collision geoms have no explicit names; XML defaults apply
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainImporterCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": MICRODUCK_WALK_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")
    viewer.launch(scene.compile())
