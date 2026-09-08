"""The shell — go-grgs ticket #44, ADR 0011, docs/specs/shell.md.

Every assertion here is made against a COMPILED MuJoCo model, never against
add_shell.py's XML text: a re-export or a refactor of the injection script
cannot make a wrong shell pass.

What is locked in:
  * containment — each primitive's AABB sits inside its body's visual-mesh AABB
    with at least the 3 mm margin on every side;
  * the bitmask — over EVERY pair of robot geoms, no contact is possible except
    between the two self_collision_only geoms that could already touch;
  * the feet still contact the floor, and the `feet_ground_contact` sensor's
    geom pattern still matches exactly the two soles;
  * a wall in front of the trunk at distance d touches and at d + 2 cm does not;
  * REST HEIGHT — a robot dropped on each of its four sides settles as low as it
    did on the pre-shell ground-contact model, and never on a thigh (#44 refit);
  * every registered task's env cfg still builds and its robot spec compiles;
  * MICRODUCK_NO_SHELL=1 reproduces the pre-shell geom set byte for byte.
"""

from __future__ import annotations

import hashlib
import importlib
import itertools
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

import mjlab_microduck.robot.microduck_constants as C

ROBOT_DIR = Path(C.MICRODUCK_WALK_XML).parent
SCENE_WALK_XML = ROBOT_DIR / "scene_walk.xml"
# The PRE-SHELL reference for the rest-height test: scene.xml wraps
# robot_groundcontact.xml, the model grgworld loaded before ticket #44 pointed
# it at robot_walk.xml. Same floor, same STAND keyframe, so the two scenes are
# directly comparable. (The strip switch is NOT the reference here: the walk
# export's only world-colliding geoms are the two soles, so a stripped robot on
# its side has nothing to land on and falls through the floor — measured at
# -113 mm, not a rest height at all.)
OLD_SCENE_XML = ROBOT_DIR / "scene.xml"
ADD_SHELL = ROBOT_DIR / "add_shell.py"

MARGIN = 0.003  # m — add_shell.py's DEFAULT_MARGIN

# The thigh deliberately carries NO shell and the head is a capsule, not a box:
# see add_shell.py's header and test_a_fallen_robot_rests_as_low_as_before.
SHELL_GEOMS = {
    "shell_trunk": ("trunk_base", mujoco.mjtGeom.mjGEOM_BOX),
    "shell_neck": ("neck", mujoco.mjtGeom.mjGEOM_CAPSULE),
    "shell_head": ("jaw_soft", mujoco.mjtGeom.mjGEOM_CAPSULE),
    "shell_shank_left": ("leg", mujoco.mjtGeom.mjGEOM_CAPSULE),
    "shell_shank_right": ("leg_2", mujoco.mjtGeom.mjGEOM_CAPSULE),
}

THIGH_BODIES = ("upper_leg_left", "upper_leg_right")

FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")

# group the ToF range grid and the detector cast their rays against
# (grgworld senses.py masks groups 0 and 3).
RAY_VISIBLE_GROUP = 3

# sha256 over the compiled geom table (name|type|size, in model order) of
# robot_walk.xml and robot_walk_backlash.xml BEFORE the shell was injected
# (commit 4aaa4df, 75 geoms). Regenerate only if the export itself changes:
#   name|type|sx,sy,sz  per geom, "\n"-joined, %.9g on the sizes.
PRE_SHELL_GEOM_SHA256 = "b77dae9d167d9fe9cf9243e6e15249f57d0e2429799aeafabb20b2a15a910986"
PRE_SHELL_NGEOM = 75


# ── helpers (all read the compiled model) ────────────────────────────────────


def geom_name(model: mujoco.MjModel, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""


def body_name(model: mujoco.MjModel, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) or ""


def geom_local_aabb(model: mujoco.MjModel, gid: int):
    """The geom's own AABB expressed in its BODY's frame (8 transformed corners)."""
    center, half = model.geom_aabb[gid][:3], model.geom_aabb[gid][3:]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
    rot = rot.reshape(3, 3)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for signs in itertools.product((-1, 1), repeat=3):
        corner = model.geom_pos[gid] + rot @ (center + np.array(signs) * half)
        lo = np.minimum(lo, corner)
        hi = np.maximum(hi, corner)
    return lo, hi


def body_visual_aabb(model: mujoco.MjModel, bid: int):
    """Union AABB of the body's VISUAL mesh geoms (default class `visual` = group 2)."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid or model.geom_group[gid] != 2:
            continue
        glo, ghi = geom_local_aabb(model, gid)
        lo = np.minimum(lo, glo)
        hi = np.maximum(hi, ghi)
    assert np.isfinite(lo).all(), "body has no visual geoms"
    return lo, hi


def can_collide(model: mujoco.MjModel, a: int, b: int) -> bool:
    """MuJoCo's bitmask rule for a candidate pair."""
    return bool(
        (model.geom_contype[a] & model.geom_conaffinity[b])
        or (model.geom_contype[b] & model.geom_conaffinity[a])
    )


def collidable_geoms(model: mujoco.MjModel) -> list[int]:
    return [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]


def geom_table(model: mujoco.MjModel) -> list[str]:
    return [
        f"{geom_name(model, g)}|{int(model.geom_type[g])}|"
        f"{model.geom_size[g][0]:.9g},{model.geom_size[g][1]:.9g},{model.geom_size[g][2]:.9g}"
        for g in range(model.ngeom)
    ]


def geom_table_sha256(model: mujoco.MjModel) -> str:
    return hashlib.sha256("\n".join(geom_table(model)).encode()).hexdigest()


def geom_keys(model: mujoco.MjModel) -> dict[int, str]:
    """A stable identity per geom.

    Every mesh geom in the export is unnamed, so unnamed ones are numbered
    within their own BODY — a key that survives the shell being inserted in
    front of them, unlike the model-wide geom index.
    """
    keys: dict[int, str] = {}
    seen: dict[str, int] = {}
    for gid in range(model.ngeom):
        body = body_name(model, gid)
        ordinal = seen.get(body, 0)
        seen[body] = ordinal + 1
        keys[gid] = geom_name(model, gid) or f"<{body}#{ordinal}>"
    return keys


def possible_robot_pairs(model: mujoco.MjModel) -> set[tuple[str, str]]:
    """Every robot-to-robot geom pair MuJoCo's broadphase could ever produce.

    Keyed by (name-or-body#index, ...) so unnamed geoms are still identified.
    Welded-together geoms are excluded exactly as mj_collision does.
    """
    key = geom_keys(model)
    out = set()
    geoms = collidable_geoms(model)
    for a, b in itertools.combinations(geoms, 2):
        if model.body_weldid[model.geom_bodyid[a]] == model.body_weldid[model.geom_bodyid[b]]:
            continue
        if can_collide(model, a, b):
            out.add(tuple(sorted((key[a], key[b]))))
    return out


@pytest.fixture(scope="module")
def walk_model() -> mujoco.MjModel:
    """The shelled walk model, straight off disk."""
    return mujoco.MjModel.from_xml_path(str(C.MICRODUCK_WALK_XML))


@pytest.fixture(scope="module")
def walk_backlash_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(C.MICRODUCK_WALK_BACKLASH_XML))


@pytest.fixture(scope="module")
def stripped_walk_model() -> mujoco.MjModel:
    """The same file with the shell stripped — i.e. the pre-shell model."""
    spec = C.strip_shell(mujoco.MjSpec.from_file(str(C.MICRODUCK_WALK_XML)))
    return spec.compile()


@pytest.fixture(scope="module")
def walk_entity_model() -> mujoco.MjModel:
    """What training actually gets: the walk robot after mjlab's collision cfg."""
    from mjlab.entity import Entity

    return Entity(C.MICRODUCK_WALK_ROBOT_CFG).spec.compile()


# ── the shell is there, on the right bodies, in the ray-visible group ────────


@pytest.mark.parametrize("model_fixture", ["walk_model", "walk_backlash_model"])
def test_shell_geoms_exist_on_their_bodies(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    for name, (body, geom_type) in SHELL_GEOMS.items():
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid >= 0, f"{name} missing"
        assert body_name(model, gid) == body
        assert model.geom_type[gid] == geom_type


def test_shell_is_in_the_ray_visible_group(walk_model):
    for name in SHELL_GEOMS:
        gid = mujoco.mj_name2id(walk_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert walk_model.geom_group[gid] == RAY_VISIBLE_GROUP, (
            f"{name} must sit in group {RAY_VISIBLE_GROUP} or the other grg's "
            "range grid cannot see it"
        )


def test_the_shell_adds_no_mass(walk_model, stripped_walk_model):
    # The bodies carry explicit <inertial>, and the shell class is density=0:
    # a shell that changed the inertia would change every trained policy.
    assert np.allclose(walk_model.body_mass, stripped_walk_model.body_mass)
    assert np.allclose(walk_model.body_inertia, stripped_walk_model.body_inertia)


# ── containment ──────────────────────────────────────────────────────────────


def test_each_shell_primitive_sits_inside_its_body_mesh_extent(walk_model):
    for name in SHELL_GEOMS:
        gid = mujoco.mj_name2id(walk_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        lo, hi = geom_local_aabb(walk_model, gid)
        blo, bhi = body_visual_aabb(walk_model, walk_model.geom_bodyid[gid])
        slack_lo = lo - blo
        slack_hi = bhi - hi
        # tol: the compiled AABBs are float32
        assert (slack_lo >= MARGIN - 1e-6).all(), f"{name} pokes out at the low side: {slack_lo}"
        assert (slack_hi >= MARGIN - 1e-6).all(), f"{name} pokes out at the high side: {slack_hi}"


# ── the bitmask: world contact only ──────────────────────────────────────────


@pytest.mark.parametrize("model_fixture", ["walk_model", "walk_entity_model"])
def test_no_robot_to_robot_pair_through_the_shell_or_the_feet(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    geoms = collidable_geoms(model)
    for a, b in itertools.combinations(geoms, 2):
        if not can_collide(model, a, b):
            continue
        # the only robot-internal pairs left must be self_collision_only (2/2)
        for gid in (a, b):
            assert (model.geom_contype[gid], model.geom_conaffinity[gid]) == (2, 2), (
                f"{geom_name(model, a) or a} and {geom_name(model, b) or b} can "
                "collide, but only the self_collision_only geoms may"
            )


@pytest.mark.parametrize("model_fixture", ["walk_model", "walk_entity_model"])
def test_shell_and_feet_can_still_touch_the_world(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    # the world's geoms are MuJoCo's default contype/conaffinity 1/1 (floor,
    # walls, furniture, people, the ball, the dock post).
    for name in list(SHELL_GEOMS) + list(FOOT_GEOMS):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid >= 0, name
        assert model.geom_contype[gid] & 1, f"{name} cannot touch the world"
        assert model.geom_conaffinity[gid] == 0, f"{name} must have zero affinity"


def test_self_collision_sensor_pairs_are_unchanged(walk_model, stripped_walk_model):
    """The subtree self_collision sensor sees exactly what it saw before.

    Its pairs are the self_collision_only geoms (trunk power_support, both
    shanks). The shell adds nothing to that set. The ONE deliberate removal is
    sole-vs-sole: ADR 0011 gives every world-colliding geom zero affinity, so a
    foot can no longer kick the other foot (nor the other shin's capsule).
    """
    before = possible_robot_pairs(stripped_walk_model)
    after = possible_robot_pairs(walk_model)

    assert not (after - before), f"the shell created new robot-robot pairs: {after - before}"
    removed = before - after
    assert removed == {tuple(sorted(FOOT_GEOMS))}, f"unexpected pair change: {removed}"

    def self_collision_only(model, pairs):
        keys = geom_keys(model)
        ids = {
            keys[g]
            for g in range(model.ngeom)
            if (model.geom_contype[g], model.geom_conaffinity[g]) == (2, 2)
        }
        return {p for p in pairs if p[0] in ids and p[1] in ids}

    # 3 pairs: trunk-vs-left-shank, trunk-vs-right-shank, shank-vs-shank
    assert len(self_collision_only(walk_model, after)) == 3
    assert len(self_collision_only(walk_model, after)) == len(
        self_collision_only(stripped_walk_model, before)
    )


# ── the sensors and the FULL_COLLISION regex are untouched ───────────────────


def test_feet_ground_contact_sensor_pattern_still_matches_exactly_the_soles(walk_model):
    import re

    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    cfg = make_microduck_velocity_env_cfg()
    sensor = next(s for s in cfg.scene.sensors if s.name == "feet_ground_contact")
    pattern = re.compile(sensor.primary.pattern)
    matched = sorted(
        geom_name(walk_model, g)
        for g in range(walk_model.ngeom)
        if pattern.match(geom_name(walk_model, g))
    )
    assert matched == sorted(FOOT_GEOMS)


def test_no_shell_geom_is_named_like_a_collision_geom(walk_model):
    import re

    collision_expr = re.compile(r".*_collision")
    for name in SHELL_GEOMS:
        assert not collision_expr.match(name), (
            f"{name} would fall into FULL_COLLISION's condim/friction table "
            "and into the feet sensor's world"
        )
    # and FULL_COLLISION's expression still selects exactly the pre-shell set
    assert sorted(
        geom_name(walk_model, g)
        for g in range(walk_model.ngeom)
        if collision_expr.match(geom_name(walk_model, g))
    ) == sorted(FOOT_GEOMS)


# ── contact facts on a posed robot ───────────────────────────────────────────


def _stand(model: mujoco.MjModel, drop: float = 0.0) -> mujoco.MjData:
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.qpos[2] -= drop
    mujoco.mj_forward(model, data)
    return data


def _contact_pairs(model, data):
    return {
        tuple(sorted((geom_name(model, c.geom1), geom_name(model, c.geom2))))
        for c in data.contact[: data.ncon]
    }


@pytest.fixture(scope="module")
def scene_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE_WALK_XML))


def test_the_feet_still_contact_the_floor(scene_model):
    data = _stand(scene_model, drop=0.003)
    pairs = _contact_pairs(scene_model, data)
    assert ("floor", "left_foot_collision") in pairs
    assert ("floor", "right_foot_collision") in pairs
    # and nothing else of the robot is on the floor in a standing pose
    assert all("floor" not in p or "foot_collision" in p[0] + p[1] for p in pairs)


def _scene_with_wall(x_face: float) -> mujoco.MjModel:
    """scene_walk.xml plus a box whose near face is at world x = `x_face`.

    Half a metre thick, +/-3 cm in y and covering z 0.05-0.20 m, so it faces the
    TRUNK only: the thighs sit outside |y| < 3 cm and the head is above 0.20 m.
    """
    spec = mujoco.MjSpec.from_file(str(SCENE_WALK_XML))
    wall = spec.worldbody.add_geom()
    wall.name = "wall"
    wall.type = mujoco.mjtGeom.mjGEOM_BOX
    wall.size = [0.25, 0.03, 0.075]
    wall.pos = [x_face + 0.25, 0.0, 0.125]
    return spec.compile()


def _trunk_shell_front_x(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "shell_trunk")
    center, half = model.geom_aabb[gid][:3], model.geom_aabb[gid][3:]
    rot = data.geom_xmat[gid].reshape(3, 3)
    return max(
        (data.geom_xpos[gid] + rot @ (center + np.array(s) * half))[0]
        for s in itertools.product((-1, 1), repeat=3)
    )


def test_a_wall_in_front_of_the_trunk_touches_and_two_centimetres_further_does_not(
    scene_model,
):
    front = _trunk_shell_front_x(scene_model, _stand(scene_model))

    d = front - 0.002  # 2 mm inside the trunk shell's front face
    touching = _scene_with_wall(d)
    pairs = _contact_pairs(touching, _stand(touching))
    assert ("shell_trunk", "wall") in pairs, (
        f"a wall at x={d:.4f} must touch the trunk shell; contacts were {pairs}"
    )

    clear = _scene_with_wall(d + 0.02)
    pairs = _contact_pairs(clear, _stand(clear))
    assert not any("wall" in p[0] + p[1] for p in pairs), (
        f"a wall 2 cm further must not touch; contacts were {pairs}"
    )


def test_without_the_shell_the_same_wall_is_a_phantom(scene_model):
    """The regression this ticket exists to fix, stated as a test."""
    front = _trunk_shell_front_x(scene_model, _stand(scene_model))
    spec = C.strip_shell(mujoco.MjSpec.from_file(str(SCENE_WALK_XML)))
    wall = spec.worldbody.add_geom()
    wall.name = "wall"
    wall.type = mujoco.mjtGeom.mjGEOM_BOX
    wall.size = [0.25, 0.03, 0.075]
    wall.pos = [front - 0.002 + 0.25, 0.0, 0.125]
    model = spec.compile()
    pairs = _contact_pairs(model, _stand(model))
    assert not any("wall" in p[0] + p[1] for p in pairs)


# ── rest height: a fallen grg must lie as low as it did before the shell ─────
#
# grgworld #45: with the shell as first merged, a grg that fell on its side came
# to rest on shell_head plus a thigh capsule with its trunk 74 mm up (37 mm on
# the old ground-contact model). The stand policy could never fold its legs
# under a trunk that high, so the chain cycled fallen -> recovering -> fallen,
# 1858 falls in a 2 h office day. This is that regression, at the model door.

REST_TOL = 0.010  # m — the shelled model must settle within 10 mm of the old one
DROP_SECONDS = 2.0
DROP_LIFT = 0.03  # m of air under the lowest colliding point before letting go

# free-joint rotations off STAND: (axis, angle). +90 deg of roll puts the robot
# on its left side, -90 deg of pitch on its back.
DROP_SIDES: dict[str, tuple[tuple[float, float, float], float]] = {
    "left": ((1.0, 0.0, 0.0), np.pi / 2),
    "right": ((1.0, 0.0, 0.0), -np.pi / 2),
    "back": ((0.0, 1.0, 0.0), -np.pi / 2),
    "front": ((0.0, 1.0, 0.0), np.pi / 2),
}

# Bodies the robot is ALLOWED to come to rest on. The old ground-contact model
# lands on its head shells, its hips, its trunk battery and its soles; the shell
# has no hip primitive, so the shelled model uses the trunk and the shanks
# instead. `upper_leg_*` is absent on purpose — resting on a thigh is exactly
# the failure this test exists to catch.
RESTING_BODIES = frozenset(
    {"jaw_soft", "trunk_base", "leg", "leg_2", "ankle_left", "ankle_right", "hip_l", "hip_l_2"}
)


def _settle_on_side(xml_path: Path, side: str) -> tuple[float, set[str], set[str]]:
    """Drop the robot onto `side` with the servos LIMP and let it settle.

    Limp = zero actuator gain and bias, i.e. no motor torque at all, so what the
    robot comes to rest on is decided by its collision geometry and gravity and
    by nothing else. Returns (trunk height, floor-contact geom names, the bodies
    those geoms belong to).
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))  # fresh: we mutate it
    model.actuator_gainprm[:, 0] = 0.0
    model.actuator_biasprm[:, :] = 0.0

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    )
    axis, angle = DROP_SIDES[side]
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array(axis), angle)
    data.qpos[3:7] = quat

    # lift it so DROP_LIFT of air sits under its lowest colliding point
    mujoco.mj_forward(model, data)
    lowest = min(
        data.geom_xpos[g][2] - model.geom_rbound[g]
        for g in range(model.ngeom)
        if model.geom_bodyid[g] != 0 and (model.geom_contype[g] or model.geom_conaffinity[g])
    )
    data.qpos[2] += DROP_LIFT - lowest
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    for _ in range(int(DROP_SECONDS / model.opt.timestep)):
        mujoco.mj_step(model, data)

    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    on_floor = {
        c.geom1 if geom_name(model, c.geom2) == "floor" else c.geom2
        for c in data.contact[: data.ncon]
        if "floor" in (geom_name(model, c.geom1), geom_name(model, c.geom2))
    }
    return (
        float(data.xpos[trunk][2]),
        {geom_name(model, g) for g in on_floor},
        {body_name(model, g) for g in on_floor},
    )


@pytest.fixture(scope="module")
def old_rest_heights() -> dict[str, float]:
    """Settled trunk height per side on the PRE-SHELL ground-contact model."""
    return {side: _settle_on_side(OLD_SCENE_XML, side)[0] for side in DROP_SIDES}


@pytest.mark.parametrize("side", list(DROP_SIDES))
def test_a_fallen_robot_rests_as_low_as_before_the_shell(side, old_rest_heights):
    """Trunk height 2 s after a limp drop, shelled walk vs old ground-contact.

    Measured (mm, trunk_base world z), old -> new:

        side     old    new    delta
        left     36.6   38.7   +2.1
        right    36.6   39.3   +2.7
        back     46.0   48.7   +2.7
        front    28.5   33.7   +5.2

    For the record, what the rejected fits gave on the two side falls (see
    add_shell.py's header for the full table): the shell as first merged 73.6 /
    73.5, a thigh box 79.7 / 79.8, a thigh capsule at the shank's radius 63.3 /
    63.5 — all far outside the 10 mm the stand policy can work with.
    """
    old = old_rest_heights[side]
    new, geoms, bodies = _settle_on_side(SCENE_WALK_XML, side)

    assert abs(new - old) <= REST_TOL, (
        f"fallen on its {side}, the shelled robot's trunk settles at "
        f"{new * 1000:.1f} mm against {old * 1000:.1f} mm on the pre-shell "
        f"ground-contact model ({(new - old) * 1000:+.1f} mm) — it is resting on "
        f"{sorted(geoms)}, and a stand policy cannot get its legs under a trunk "
        "held that high"
    )


@pytest.mark.parametrize("side", list(DROP_SIDES))
def test_a_fallen_robot_never_rests_on_a_thigh(side):
    """It comes down on head / trunk / shanks / soles — the parts that did before."""
    _, geoms, bodies = _settle_on_side(SCENE_WALK_XML, side)
    assert geoms, f"the robot dropped on its {side} is not touching the floor at all"
    assert not (bodies & set(THIGH_BODIES)), (
        f"fallen on its {side}, the robot is resting on a thigh: {sorted(geoms)}"
    )
    assert bodies <= RESTING_BODIES, (
        f"fallen on its {side}, the robot is resting on {sorted(bodies - RESTING_BODIES)}, "
        f"which the pre-shell model never came down on (contacts: {sorted(geoms)})"
    )


def test_the_thighs_carry_no_shell_primitive(walk_model):
    """#44 refit: the thigh AABB is the hip servo block, so its inscribed capsule
    was a near-sphere of radius 26.1 mm — the fattest thing on the robot, and the
    leg had to fold through it. The old ground-contact model had hip collision
    meshes and none on the thigh; the shell now matches that."""
    for body in THIGH_BODIES:
        bid = mujoco.mj_name2id(walk_model, mujoco.mjtObj.mjOBJ_BODY, body)
        assert bid >= 0, body
        assert not [
            g
            for g in range(walk_model.ngeom)
            if walk_model.geom_bodyid[g] == bid and geom_name(walk_model, g).startswith("shell_")
        ], f"{body} must carry no shell primitive"


# ── every registered task still builds, on CPU ───────────────────────────────


def test_every_registered_task_cfg_builds_and_its_robot_spec_compiles():
    import mjlab_microduck.tasks  # noqa: F401  registers on import
    from mjlab.entity import Entity
    from mjlab.tasks.registry import list_tasks, load_env_cfg

    tasks = list_tasks()
    assert tasks, "no tasks registered"
    compiled: dict[object, int] = {}
    for task in tasks:
        for play in (False, True):
            cfg = load_env_cfg(task, play=play)
            robot = cfg.scene.entities.get("robot")
            if robot is None:  # the cartpole demo tasks carry no microduck
                continue
            if robot.spec_fn not in compiled:
                compiled[robot.spec_fn] = Entity(robot).spec.compile().ngeom
            assert compiled[robot.spec_fn] > 0


def test_the_walk_tasks_run_on_the_shelled_model():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.entity import Entity
    from mjlab.tasks.registry import list_tasks, load_env_cfg

    seen = set()
    for task in list_tasks():
        robot = load_env_cfg(task).scene.entities.get("robot")
        if robot is None or robot.spec_fn not in (C.get_walk_spec, C.get_walk_backlash_spec):
            continue
        seen.add(task)
        model = Entity(robot).spec.compile()
        for name in SHELL_GEOMS:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert gid >= 0, f"{task}: {name} was dropped by the collision cfg"
            assert model.geom_contype[gid] == 1 and model.geom_conaffinity[gid] == 0
    assert seen, "no task loads the walk model any more"


# ── the strip switch ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "xml", ["MICRODUCK_WALK_XML", "MICRODUCK_WALK_BACKLASH_XML"]
)
def test_strip_switch_reproduces_the_pre_shell_geom_set(xml):
    model = C.strip_shell(mujoco.MjSpec.from_file(str(getattr(C, xml)))).compile()
    assert model.ngeom == PRE_SHELL_NGEOM
    assert geom_table_sha256(model) == PRE_SHELL_GEOM_SHA256, (
        "stripping the shell no longer reproduces the pre-shell geom set "
        "(names, types, sizes, in order)"
    )
    for name in FOOT_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_conaffinity[gid] == 1, "the feet must get their affinity back"


def test_the_env_var_wires_all_the_way_to_the_built_robot(monkeypatch):
    from mjlab.entity import Entity

    monkeypatch.setenv("MICRODUCK_NO_SHELL", "1")
    try:
        mod = importlib.reload(C)
        assert mod.NO_SHELL is True
        assert mod.WALK_COLLISION is mod.FULL_COLLISION
        model = Entity(mod.MICRODUCK_WALK_ROBOT_CFG).spec.compile()
        assert model.ngeom == PRE_SHELL_NGEOM
        assert geom_table_sha256(model) == PRE_SHELL_GEOM_SHA256
        assert not [g for g in range(model.ngeom) if geom_name(model, g).startswith("shell_")]
        for name in FOOT_GEOMS:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert model.geom_conaffinity[gid] == 1
    finally:
        monkeypatch.delenv("MICRODUCK_NO_SHELL", raising=False)
        importlib.reload(C)
    assert C.NO_SHELL is False


# ── the injection script itself ──────────────────────────────────────────────


def test_add_shell_is_idempotent(tmp_path):
    """A second run must add nothing (the committed models are already shelled)."""
    result = subprocess.run(
        [sys.executable, str(ADD_SHELL), str(C.MICRODUCK_WALK_XML)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "already contains a shell" in result.stdout
    # and the file is untouched
    assert mujoco.MjModel.from_xml_path(str(C.MICRODUCK_WALK_XML)).ngeom == PRE_SHELL_NGEOM + len(
        SHELL_GEOMS
    )
