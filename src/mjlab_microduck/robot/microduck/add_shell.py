#!/usr/bin/env python3
"""Inject the SHELL — world-collision primitives — into an onshape-to-robot MJCF.

The exported walk model touches the world through its two foot soles and
nothing else, so a policy trained on it cannot feel a wall or a chair leg.
This script adds one hand-fit primitive per major link:

    trunk  box       neck   capsule    head   box
    thigh  capsule   shank  capsule    (feet keep their sole meshes)

and rewires the contact bitmask so those primitives touch the WORLD only:

    <default class="shell">   <geom contype="1" conaffinity="0" group="3"/>
    <default class="collision">  ... conaffinity="0"   (the feet)

MuJoCo pairs two geoms when ``contype1 & conaffinity2 || contype2 &
conaffinity1``. With conaffinity 0 on every world-colliding geom of the robot
(shell and feet alike) the shell touches the floor/walls/furniture/people
(contype 1 vs the world's default conaffinity 1) and can never touch another
part of the robot — so the velocity task's ``self_collision`` subtree sensor
cannot see the shell, and a foot cannot kick the other shin's capsule. The
``self_collision_only`` geoms (contype/conaffinity 2) are untouched.

Group 3 is the group the range grid and the ToF detector cast rays against
(grgworld ``senses.py`` masks groups 0 and 3), so the shell is what another
grg sees.

Sizes are fitted from the COMPILED visual-mesh AABB of each body, in that
body's own local frame, shrunk by ``--margin`` (3 mm) on every side — see
``BODY_AABB`` below for the measured extents, and ``measure_body_visual_aabb``
for the helper that produced them. ``--refit`` re-measures the file being
processed instead of trusting the table (needs mujoco installed).

Meant to run as the LAST post_import_command of an onshape-to-robot config
(after add_backlash.py, see config_mjcf_walk.json / config_mjcf_walk_backlash.json)
but works standalone on any already-exported robot xml:

    python3 add_shell.py robot_walk.xml

Idempotent: a second run detects its own ``class="shell"`` default and refuses.

The shell can be stripped at load time with ``MICRODUCK_NO_SHELL=1`` — see
``microduck_constants.py`` (the geoms are deleted from the MjSpec and the feet
get their conaffinity back, reproducing the pre-shell model exactly).
"""

import argparse
import re
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Fitted geometry
# ---------------------------------------------------------------------------

DEFAULT_MARGIN = 0.003  # m, taken off every side of the mesh extent

# Axis-aligned bounding box of every VISUAL geom of the body (group 2), in the
# body's own local frame, metres, as (lo_xyz, hi_xyz). Measured on the compiled
# model by measure_body_visual_aabb() below; identical for robot_walk.xml,
# robot_walk_backlash.xml and robot_groundcontact.xml (the visual mesh set does
# not depend on the export's collision `ignore` block). Re-measure with --refit.
BODY_AABB: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    # size 0.0885 x 0.0842 x 0.1001 — trunk shells + battery + power support
    "trunk_base": ((-0.051784, -0.041828, -0.044236), (0.036758, 0.042399, 0.055829)),
    # size 0.0200 x 0.0691 x 0.0291 — neck plates + the two neck xl330s
    "neck": ((-0.010000, -0.059569, -0.000087), (0.010000, 0.009569, 0.029047)),
    # size 0.0743 x 0.0924 x 0.1324 — head shells, jaw, face, lens, pi
    "jaw_soft": ((-0.028660, -0.045950, -0.089621), (0.045664, 0.046417, 0.042823)),
    # size 0.0734 x 0.0774 x 0.0582 — thigh bracket + rigidity plate + xl330s
    "upper_leg_left": ((-0.033118, -0.018365, -0.007421), (0.040252, 0.059013, 0.050744)),
    "upper_leg_right": ((-0.040252, -0.018366, -0.007421), (0.033118, 0.059013, 0.050743)),
    # size 0.0200 x 0.0617 x 0.0330 — shank plate + ankle xl330 (axes swap L/R)
    "leg": ((-0.010000, -0.010138, -0.026087), (0.010000, 0.051569, 0.006865)),
    "leg_2": ((-0.051569, -0.010000, -0.026087), (0.010138, 0.010000, 0.006865)),
}


@dataclass(frozen=True)
class ShellPart:
    """One shell primitive: which body it goes on and what shape it is."""

    name: str
    body: str
    kind: str  # "box" | "capsule"


# Names are `shell_*`, deliberately NOT `*_collision`: FULL_COLLISION's
# geom_names_expr, the feet_ground_contact sensor and the condim/friction
# tables in microduck_constants.py all key on `.*_collision`, and the shell
# must not fall into any of them.
SHELL_PARTS: tuple[ShellPart, ...] = (
    ShellPart("shell_trunk", "trunk_base", "box"),
    ShellPart("shell_neck", "neck", "capsule"),
    ShellPart("shell_head", "jaw_soft", "box"),
    ShellPart("shell_thigh_left", "upper_leg_left", "capsule"),
    ShellPart("shell_thigh_right", "upper_leg_right", "capsule"),
    ShellPart("shell_shank_left", "leg", "capsule"),
    ShellPart("shell_shank_right", "leg_2", "capsule"),
)

SHELL_RGBA = "0.9 0.45 0.1 0.3"


def fit_box(lo, hi, margin: float):
    """Largest box inside the AABB shrunk by `margin` on every side.

    Returns (pos, half_sizes) in the body's local frame.
    """
    pos = tuple((a + b) / 2.0 for a, b in zip(lo, hi))
    half = tuple((b - a) / 2.0 - margin for a, b in zip(lo, hi))
    if min(half) <= 0.0:
        raise ValueError(f"margin {margin} too large for extent {tuple(b - a for a, b in zip(lo, hi))}")
    return pos, half


def fit_capsule(lo, hi, margin: float):
    """Largest capsule inside the AABB shrunk by `margin`, along its long axis.

    The capsule's own AABB is +/-(half_len + radius) on the long axis and
    +/-radius on the other two, so radius is set by the SHORT axes and the
    half-length by what is left of the long one. Returns (axis, p0, p1, radius)
    with p0/p1 the fromto endpoints in the body's local frame.
    """
    half = [(b - a) / 2.0 - margin for a, b in zip(lo, hi)]
    center = [(a + b) / 2.0 for a, b in zip(lo, hi)]
    if min(half) <= 0.0:
        raise ValueError(f"margin {margin} too large for extent {tuple(b - a for a, b in zip(lo, hi))}")
    axis = max(range(3), key=lambda i: half[i])
    radius = min(half[i] for i in range(3) if i != axis)
    half_len = half[axis] - radius
    if half_len <= 0.0:
        raise ValueError(f"body is not elongated enough for a capsule: half-extents {half}")
    p0 = list(center)
    p1 = list(center)
    p0[axis] -= half_len
    p1[axis] += half_len
    return axis, tuple(p0), tuple(p1), radius


def _fmt(values) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def build_shell_geoms(aabb: dict, margin: float) -> list[tuple[ShellPart, str, str]]:
    """Return (part, geom_xml_line_body, human_summary) for every shell part."""
    out = []
    for part in SHELL_PARTS:
        lo, hi = aabb[part.body]
        if part.kind == "box":
            pos, half = fit_box(lo, hi, margin)
            attrs = f'type="box" pos="{_fmt(pos)}" size="{_fmt(half)}"'
            summary = (
                f"box   half={_fmt(half)} pos={_fmt(pos)}  "
                f"(from extent {_fmt(b - a for a, b in zip(lo, hi))} - 2*{margin})"
            )
        elif part.kind == "capsule":
            axis, p0, p1, radius = fit_capsule(lo, hi, margin)
            attrs = f'type="capsule" fromto="{_fmt(p0)} {_fmt(p1)}" size="{radius:.6g}"'
            summary = (
                f"caps  r={radius:.6g} along {'xyz'[axis]} fromto={_fmt(p0)} -> {_fmt(p1)}  "
                f"(from extent {_fmt(b - a for a, b in zip(lo, hi))} - 2*{margin})"
            )
        else:  # pragma: no cover - guarded by the table above
            raise ValueError(f"unknown shell kind {part.kind!r}")
        out.append((part, f'<geom name="{part.name}" class="shell" {attrs}/>', summary))
    return out


# ---------------------------------------------------------------------------
# Measurement helper (the numbers in BODY_AABB came from here)
# ---------------------------------------------------------------------------


def measure_body_visual_aabb(xml_path: str, bodies) -> dict:
    """Compile `xml_path` and return each body's visual-geom AABB, local frame.

    Uses the COMPILED model (mjModel.geom_aabb / geom_pos / geom_quat), i.e. the
    real mesh extents MuJoCo computed, not anything read out of the XML text.
    Visual geoms are the ones in group 2 (default class `visual`).
    """
    import mujoco  # imported lazily: the export pipeline may not have it
    import numpy as np

    model = mujoco.MjModel.from_xml_path(xml_path)
    result = {}
    for body in bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            raise ValueError(f"body {body!r} not found in {xml_path}")
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != bid or model.geom_group[gid] != 2:
                continue
            center, halfext = model.geom_aabb[gid][:3], model.geom_aabb[gid][3:]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
            rot = rot.reshape(3, 3)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        corner = center + np.array([sx, sy, sz]) * halfext
                        world = model.geom_pos[gid] + rot @ corner
                        lo = np.minimum(lo, world)
                        hi = np.maximum(hi, world)
        if not np.isfinite(lo).all():
            raise ValueError(f"body {body!r} has no visual geoms in {xml_path}")
        result[body] = (tuple(lo), tuple(hi))
    return result


# ---------------------------------------------------------------------------
# XML editing (line-based, same style as add_backlash.py)
# ---------------------------------------------------------------------------

BODY_OPEN_RE = re.compile(r'^(\s*)<body\s[^>]*\bname="([^"]+)"')
BODY_CLOSE_RE = re.compile(r"^\s*</body>")
GEOM_RE = re.compile(r"^(\s*)<geom\b")
COLLISION_DEFAULT_RE = re.compile(r'^\s*<default class="collision">')
GEOM_OPEN_RE = re.compile(r"^(\s*)<geom\b(.*?)/>\s*$")


def build_shell_default(margin: float) -> str:
    return (
        f"  <!-- Shell injected by add_shell.py: world-collision primitives fitted\n"
        f"       to each body's visual mesh extent minus a {margin * 1000:g} mm margin.\n"
        f"       contype 1 / conaffinity 0 = touches the world (default 1/1) and\n"
        f"       NEVER another robot geom, so the self_collision subtree sensor\n"
        f"       cannot see it. group 3 = the group the ToF rays are cast against.\n"
        f"       Stripped at load by MICRODUCK_NO_SHELL=1. -->\n"
        f"  <default>\n"
        f'    <default class="shell">\n'
        f'      <geom contype="1" conaffinity="0" group="3" density="0"'
        f' rgba="{SHELL_RGBA}"/>\n'
        f"    </default>\n"
        f"  </default>\n"
    )


def patch_collision_default(lines: list[str]) -> bool:
    """Give the `collision` default class conaffinity 0 (world contact only).

    The feet are the only geoms of that class in the walk model; in a model that
    keeps more of them they all become world-only too, which is the ADR's rule.
    Returns True if the class was found and patched.
    """
    for i, line in enumerate(lines):
        if not COLLISION_DEFAULT_RE.match(line):
            continue
        for j in range(i + 1, min(i + 6, len(lines))):
            m = GEOM_OPEN_RE.match(lines[j])
            if m is None:
                continue
            indent, attrs = m.group(1), m.group(2)
            attrs = re.sub(r'\s*\bcontype="[^"]*"', "", attrs)
            attrs = re.sub(r'\s*\bconaffinity="[^"]*"', "", attrs)
            lines[j] = (
                f'{indent}<geom{attrs} contype="1" conaffinity="0"/>'
                f"  <!-- world contact only: no robot-robot pair (add_shell.py) -->\n"
            )
            return True
    return False


def insert_geoms(lines: list[str], geoms: dict[str, str]) -> list[str]:
    """Insert each body's shell geom after that body's last direct-child geom.

    `geoms` maps body name -> the geom element text (no indentation).
    """
    stack: list[str] = []
    anchor: dict[str, int] = {}
    body_indent: dict[str, str] = {}
    for i, line in enumerate(lines):
        m = BODY_OPEN_RE.match(line)
        if m is not None:
            stack.append(m.group(2))
            body_indent[m.group(2)] = m.group(1)
            if not line.rstrip().endswith("/>"):
                pass
            else:  # self-closing body: nothing can be inserted into it
                stack.pop()
            continue
        if BODY_CLOSE_RE.match(line):
            if stack:
                stack.pop()
            continue
        gm = GEOM_RE.match(line)
        if gm is not None and stack and stack[-1] in geoms:
            anchor[stack[-1]] = i

    missing = [b for b in geoms if b not in anchor]
    if missing:
        raise ValueError(f"no geom found to anchor the shell on body/bodies: {missing}")

    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        for body, geom in geoms.items():
            if anchor[body] == i:
                indent = body_indent[body] + "  "
                out.append(f"{indent}{geom}\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", help="MJCF file to modify in place")
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=f"metres taken off every side of the mesh extent (default: {DEFAULT_MARGIN})",
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="re-measure the body extents from THIS file's compiled model "
        "instead of the BODY_AABB table (requires mujoco)",
    )
    args = parser.parse_args()

    aabb = BODY_AABB
    if args.refit:
        aabb = measure_body_visual_aabb(args.xml, [p.body for p in SHELL_PARTS])
        print("[add_shell] re-measured extents (paste into BODY_AABB):")
        for body, (lo, hi) in aabb.items():
            print(
                f'    "{body}": (({lo[0]:.6f}, {lo[1]:.6f}, {lo[2]:.6f}), '
                f"({hi[0]:.6f}, {hi[1]:.6f}, {hi[2]:.6f})),"
            )

    with open(args.xml) as f:
        lines = f.readlines()

    if any('class="shell"' in line for line in lines):
        print(f"[add_shell] {args.xml} already contains a shell — aborting.")
        return 1

    try:
        built = build_shell_geoms(aabb, args.margin)
    except ValueError as exc:
        print(f"[add_shell] ERROR: {exc}")
        return 1

    out: list[str] = []
    default_inserted = False
    for line in lines:
        if not default_inserted and "<worldbody>" in line:
            out.append(build_shell_default(args.margin))
            default_inserted = True
        out.append(line)
    if not default_inserted:
        print("[add_shell] ERROR: no <worldbody> found — is this an MJCF file?")
        return 1

    if not patch_collision_default(out):
        print('[add_shell] ERROR: no <default class="collision"> block found.')
        return 1

    try:
        out = insert_geoms(out, {part.body: geom for part, geom, _ in built})
    except ValueError as exc:
        print(f"[add_shell] ERROR: {exc}")
        return 1

    with open(args.xml, "w") as f:
        f.writelines(out)

    print(f"[add_shell] added {len(built)} shell geoms to {args.xml} (margin {args.margin} m):")
    for part, _, summary in built:
        print(f"    {part.name:<19s} on {part.body:<16s} {summary}")
    print("[add_shell] feet (class collision) set to conaffinity 0: world contact only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
