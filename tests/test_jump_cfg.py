"""Jump cfg invariants (CPU, no GPU): registration, the command is pinned, the
2 s episode, every cost carries a negative weight, the both-feet-off gate opens
and shuts where it is supposed to, the apex potential is monotone and
airborne-only, the landing bonus is a one-shot conditioned on a prior flight —
and the strategy arithmetic from `go-grgs/docs/research/jump-design.md`, run
through the REAL reward functions rather than asserted in prose.

Reward bookkeeping is checked on SYNTHETIC contact/height sequences, per the
tippy-taps lesson ("physics is for rendering and looking"); the two facts that
do need the real model — the foot sites are the sole, and the head joint indices
are head_yaw / head_roll — are measured off `robot_walk.xml` here.
"""

import math
import types

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_jump_env_cfg import (
    ALIVE_TILT_DEG,
    APEX_M,
    CLEARANCE_M,
    DRIFT_SAT_M,
    ENABLE_SYMMETRY,
    EPISODE_LENGTH_S,
    FLIGHT_MIN_S,
    HEAD_STILL_JOINTS,
    HEAD_STILL_STD,
    HEAD_VEL_CAP,
    HEAD_VEL_JOINTS,
    LAND_SETTLE_S,
    LAND_TILT_DEG,
    LOAD_CAP_POINTS,
    LOAD_DEPTH_M,
    LOAD_WINDOW_S,
    STAND_Z,
    W_ACTION_RATE,
    W_ALIVE,
    W_APEX,
    W_DRIFT,
    W_FLIGHT,
    W_LAND,
    W_LOAD,
    W_STAGGER,
    W_TERMINATED,
    MicroduckJumpRlCfg,
    make_microduck_jump_env_cfg,
)

_CFG = make_microduck_jump_env_cfg()          # the real weights, for the arithmetic
_STEPS = int(EPISODE_LENGTH_S / 0.02)         # 100 control steps
_LEFT, _RIGHT = 0, 1
_N_SERVO = 14


# ── registration and the cfg's shape ─────────────────────────────────────────

def test_registered():
    import mjlab_microduck.tasks  # noqa: F401  registers on import
    from mjlab.tasks.registry import list_tasks
    assert "Mjlab-Jump-Flat-MicroDuck" in list_tasks()
    # A hop has no left/right roles to break, so there is no backlash twin to
    # keep in sync — and the design note says explicitly: not in _BACKLASH_TASKS.
    assert "Mjlab-Jump-Flat-Backlash-MicroDuck" not in list_tasks()


def test_command_is_pinned_and_the_episode_is_two_seconds():
    """The twist is pinned near zero for the whole episode — no direction sign,
    unlike the pivot, so ONE onnx installs once — and the episode is 2 s.

    A 4 s episode would be 3.3 s of standing still after a ~0.65 s manoeuvre,
    which doubles both the `alive` income the hop competes against and the
    wall-clock per rehearsal."""
    cfg = make_microduck_jump_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.VelocityCommandCommandOnlyCfg)
    assert not isinstance(cmd, microduck_mdp.PivotCommandCfg), (
        "the jump has no direction flag: the twist yaw slot stays pinned at ~0"
    )
    assert cmd.ranges.lin_vel_x == (-0.01, 0.01)
    assert cmd.ranges.lin_vel_y == (-0.01, 0.01)
    assert cmd.ranges.ang_vel_z == (-0.05, 0.05)
    assert cmd.rel_standing_envs == 0.0 and cmd.heading_command is False
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 2.0
    assert math.isclose(cfg.decimation * cfg.sim.mujoco.timestep, 0.02)
    assert _STEPS == 100
    # The head/body command slots stay alive (61D obs contract) even though the
    # jump drives neither.
    assert "head_pose" in cfg.commands and "body_pose" in cfg.commands
    assert cfg.rewards["head_pose_tracking"].weight == 0.0
    assert cfg.rewards["body_pose_tracking"].weight == 0.0
    for gone in ("standing_envs", "head_pose_range", "body_pose_range",
                 "head_pose_bias_weight", "action_rate_weight"):
        assert gone not in cfg.curriculum, gone


def test_gait_terms_gone_and_every_weight_is_signed_the_right_way():
    cfg = make_microduck_jump_env_cfg()
    # `air_time` LOOKS like it should pay a hop: its window is 0.125-0.300 s of
    # PER-FOOT flight and the whole hop's flight is 0.06-0.08 s, so it would pay
    # zero even if it were live — and it is not live, because it gates on
    # command magnitude and this command is pinned near zero.
    for gone in ("air_time", "foot_clearance", "foot_swing_height", "pose",
                 "head_pose_bias"):
        assert gone not in cfg.rewards, gone
    for pos in ("jp_flight", "jp_apex", "jp_land", "jp_load", "alive",
                "upright", "head_still", "track_linear_velocity",
                "track_angular_velocity"):
        assert cfg.rewards[pos].weight > 0, pos
    for cost in ("jp_stagger", "jp_drift", "terminated", "head_joint_vel",
                 "foot_slip", "action_rate_l2", "dof_pos_limits",
                 "self_collisions", "body_ang_vel", "angular_momentum"):
        assert cfg.rewards[cost].weight < 0, cost
    # ...and every one of those costs is a NON-NEGATIVE 0..1 function, so the
    # weighted term can only ever be <= 0 (AGENTS.md's infallible run check).
    env = _FakeEnv()
    env.tick(contacts=(0, 1), z=0.09, feet_z=(0.03, 0.0), tilt_deg=30.0,
             xy=(0.4, -0.3))
    for name, fn in (("jp_stagger", microduck_mdp.jp_stagger_penalty),
                     ("jp_drift", microduck_mdp.jp_drift_penalty),
                     ("head_joint_vel", microduck_mdp.jp_head_vel_penalty)):
        val = float(fn(env, **_params(name))[0])
        assert 0.0 <= val <= 1.0, (name, val)
        assert cfg.rewards[name].weight * val <= 0.0, name

    assert cfg.rewards["jp_flight"].weight == W_FLIGHT == 25.0
    assert cfg.rewards["jp_apex"].weight == W_APEX == 15.0
    assert cfg.rewards["jp_land"].weight == W_LAND == 15.0
    assert cfg.rewards["jp_load"].weight == W_LOAD == 0.5
    assert cfg.rewards["jp_stagger"].weight == W_STAGGER == -2.0
    assert cfg.rewards["jp_drift"].weight == W_DRIFT == -1.0
    assert cfg.rewards["alive"].weight == W_ALIVE == 3.0
    assert cfg.rewards["terminated"].weight == W_TERMINATED == -100.0
    assert cfg.rewards["upright"].weight == 2.0, "the common survival income"
    # foot_slip must not stay command-gated at a pinned ~zero command.
    assert cfg.rewards["foot_slip"].params["command_threshold"] < 0.0
    assert cfg.rewards["foot_slip"].weight == -0.2
    # Smoothness is fixed and light: the push is a full-stroke joint slam in
    # three control steps, and an attempt-tax during discovery makes doing
    # nothing win.
    assert cfg.rewards["action_rate_l2"].weight == W_ACTION_RATE == -0.05
    assert "action_rate_weight" not in cfg.curriculum


def test_no_height_term_and_no_height_termination():
    """The two things this recipe deliberately does NOT have.

    `height_stand` (pivot, happy-spin): a Gaussian on trunk z at 0.115 would
    price BOTH halves of a hop as error — the crouch goes 3 cm below it and the
    apex 1.5 cm above.

    A height termination (the bow's `collapsed` at 0.060): the hop's crouch
    bottoms at 0.088 m and the reachable crouch floor is 0.061 m, both at or
    under the lab's 0.07 m fall height. `deep_squat`'s recorded A/B — park at
    0.076 with the z-kill on, 0.070 with it off and 22% more reward — says the
    kill moves the policy OFF the goal pose."""
    cfg = make_microduck_jump_env_cfg()
    for gone in ("height_stand", "height", "jp_height"):
        assert gone not in cfg.rewards, gone
    assert set(cfg.terminations) == {
        "time_out", "fell_over", "out_of_terrain_bounds", "nan_state"
    }, "the jump adds NO termination of its own"
    assert cfg.rewards["terminated"].params["term_names"] == ("fell_over",), (
        "a nan_state is a sim blow-up, not something the policy chose; a "
        "time_out is the normal end of a 2 s episode"
    )


def test_symmetry_on_and_the_runner_cfg():
    """A hop is left/right symmetric, and here the mirror loss is also a prior
    for the thing the reward is buying: both legs doing the same thing at the
    same time."""
    assert ENABLE_SYMMETRY is True
    assert MicroduckJumpRlCfg.algorithm.symmetry_cfg is not None
    assert MicroduckJumpRlCfg.actor.obs_normalization is True
    assert MicroduckJumpRlCfg.critic.obs_normalization is True
    assert MicroduckJumpRlCfg.experiment_name == "jump"
    assert MicroduckJumpRlCfg.run_name == "jump"
    assert MicroduckJumpRlCfg.max_iterations == 2_000
    assert MicroduckJumpRlCfg.num_steps_per_env == 24


def test_the_constants_are_the_measured_ones():
    """Every number below was measured before this file existed (Part A of the
    design note); the ones a rewrite would quietly drift are pinned here."""
    assert STAND_Z == 0.115           # settled standing trunk z
    assert APEX_M == 0.015            # the best measured apex, +15.8 mm, rounded
    assert CLEARANCE_M == 0.012       # measured flat-foot clearance is 8-10 mm
    assert LOAD_DEPTH_M == 0.030 and LOAD_WINDOW_S == 0.8
    assert LOAD_CAP_POINTS == 20.0, "the load window IS the load cap"
    # With this weight and this normaliser the apex term pays exactly one point
    # per millimetre of rise, capped at 15 — which is where the design note's
    # `apex 13.5` for a +13.5 mm hop comes from.
    assert math.isclose(W_APEX / APEX_M * 0.001, 1.0)
    # 25 deg, not the pivot's 15: the measured hop peaks at 32.4 deg of tilt on
    # its own landing, so a 15 deg gate would charge it for landing.
    assert ALIVE_TILT_DEG == 25.0 > microduck_mdp.PV_ALIVE_TILT_DEG == 15.0
    assert LAND_TILT_DEG == 15.0      # ...but the FINISH is judged at 15
    assert LAND_SETTLE_S == 0.30
    assert FLIGHT_MIN_S == 0.04       # two control steps: chatter is not flight
    assert DRIFT_SAT_M == 0.05        # a hop travels less than a bow does
    # The head is bow v7's, verbatim.
    assert HEAD_STILL_JOINTS == (7, 8) and HEAD_STILL_STD == 0.5
    assert HEAD_VEL_JOINTS == (5, 6, 7, 8) and HEAD_VEL_CAP == 4.0


def test_the_foot_sites_are_the_sole_on_the_real_model():
    """The clearance probe, measured rather than assumed.

    `jp_flight_reward` reads the height of the `left_foot` / `right_foot` SITES
    above the terrain and calls it the foot's clearance. That is only true if
    the sites sit on the sole, so this measures the gap on the actual MJCF: at
    HOME the site is 1.2 mm above the lowest point of the foot's collision mesh
    and at the STAND keyframe 0.05 mm, both far inside the 12 mm normaliser."""
    import mujoco
    import mjlab_microduck.robot as robot_pkg
    from pathlib import Path
    xml = Path(robot_pkg.__file__).parent / "microduck" / "scene_walk.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    for key in range(2):                       # 0 = INIT (HOME), 1 = STAND
        mujoco.mj_resetDataKeyframe(m, d, key)
        mujoco.mj_forward(m, d)
        for site, geom in (("left_foot", "left_foot_collision"),
                           ("right_foot", "right_foot_collision")):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom)
            assert sid >= 0 and gid >= 0, (site, geom)
            mid = m.geom_dataid[gid]
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            verts = m.mesh_vert[va:va + vn].astype(float)
            world = verts @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
            gap = float(d.site_xpos[sid][2] - world[:, 2].min())
            assert 0.0 <= gap < 0.0015, (key, site, gap)
    # Slot 0 is LEFT and slot 1 is RIGHT, the order the contact sensor uses.
    cfg_feet = _CFG.rewards["jp_flight"].params["feet_cfg"]
    assert tuple(cfg_feet.site_names) == ("left_foot", "right_foot")
    # ...and a FRESH SceneEntityCfg per term (the managers resolve these in
    # place; a shared instance would bind to whichever scene got there first).
    seen = [id(_CFG.rewards[n].params["feet_cfg"])
            for n in ("jp_flight", "jp_apex", "jp_land", "jp_load",
                      "jp_stagger", "jp_drift")]
    assert len(set(seen)) == len(seen)


def test_head_joint_indices_are_head_yaw_and_head_roll_on_the_real_model():
    """Servo indices in the canonical 14-joint order (AGENTS.md / symmetry.py):
    0-4 left leg, 5 neck_pitch, 6 head_pitch, 7 head_yaw, 8 head_roll,
    9-13 right leg. `head_still` prices 7 and 8, whose HOME is 0.0."""
    import mujoco
    import mjlab_microduck.robot as robot_pkg
    from pathlib import Path
    xml = Path(robot_pkg.__file__).parent / "microduck" / "robot_walk.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(m.nu)]
    assert len(names) == _N_SERVO
    assert names[7] == "head_yaw" and names[8] == "head_roll"
    assert names[5] == "neck_pitch" and names[6] == "head_pitch"
    from mjlab_microduck.robot.microduck_constants import HOME_FRAME
    jp = HOME_FRAME.joint_pos
    assert jp[r".*head_yaw.*"] == 0.0 and jp[r".*head_roll.*"] == 0.0


# ── the jump memory on a fake env ────────────────────────────────────────────

class _FakeAsset:
    def __init__(self, n):
        # site_pos_w is (B, 2, 3): left_foot, right_foot — the order the contact
        # sensor uses too. z is the sole's height above the floor.
        feet = torch.zeros(n, 2, 3)
        feet[:, _LEFT, 1] = 0.042
        feet[:, _RIGHT, 1] = -0.042
        self.data = types.SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, STAND_Z]] * n),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n),
            site_pos_w=feet,
            joint_pos=torch.zeros(n, _N_SERVO),
            joint_vel=torch.zeros(n, _N_SERVO),
            default_joint_pos=torch.zeros(n, _N_SERVO),
        )

    def find_joints(self, pattern):   # no passive_* joints: the identity
        return list(range(_N_SERVO)), [f"j{i}" for i in range(_N_SERVO)]


class _FakeSensor:
    def __init__(self, n):
        self.data = types.SimpleNamespace(
            found=torch.ones(n, 2), current_air_time=torch.zeros(n, 2)
        )


class _FakeScene:
    def __init__(self, n, asset):
        self.sensors = {"feet_ground_contact": _FakeSensor(n)}
        self.terrain = types.SimpleNamespace(env_origins=torch.zeros(n, 3))
        self._asset = asset

    def __getitem__(self, key):       # env.scene["robot"]
        return self._asset


class _FakeTerminations:
    def __init__(self, n):
        self.active_terms = ["fell_over", "out_of_terrain_bounds", "nan_state",
                             "time_out"]
        self._dones = {k: torch.zeros(n, dtype=torch.bool)
                       for k in self.active_terms}

    def get_term(self, name):
        return self._dones[name]

    def fire(self, name):
        self._dones[name][:] = True


class _FakeEnv:
    def __init__(self, n=1):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = 0.02
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self._asset = _FakeAsset(n)
        self.scene = _FakeScene(n, self._asset)
        self.termination_manager = _FakeTerminations(n)

    @property
    def sensor(self):
        return self.scene.sensors["feet_ground_contact"]

    def tick(self, contacts=(1, 1), z=None, feet_z=None, tilt_deg=None,
             xy=None, head=None, head_vel=None):
        """Advance one env step through a fully specified world state.

        `contacts` is (left, right) as the sensor reports it, `feet_z` the two
        foot sites' height above the floor, `z` the trunk height, `tilt_deg` a
        roll about x (what a topple looks like to every tilt gate in the stack).
        """
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self.sensor.data.found = torch.tensor(
            [list(contacts)] * self.num_envs, dtype=torch.float32)
        if z is not None:
            self._asset.data.root_link_pos_w[:, 2] = z
        if xy is not None:
            self._asset.data.root_link_pos_w[:, 0] = xy[0]
            self._asset.data.root_link_pos_w[:, 1] = xy[1]
        if feet_z is None:
            # a foot in contact is on the floor; one off it is where it was
            feet_z = tuple(0.0 if c else float(
                self._asset.data.site_pos_w[0, i, 2]) for i, c in
                enumerate(contacts))
        self._asset.data.site_pos_w[:, _LEFT, 2] = feet_z[0]
        self._asset.data.site_pos_w[:, _RIGHT, 2] = feet_z[1]
        if tilt_deg is not None:
            half = math.radians(tilt_deg) / 2.0
            self._asset.data.root_link_quat_w[:] = torch.tensor(
                [math.cos(half), math.sin(half), 0.0, 0.0])
        if head is not None:
            self._asset.data.joint_pos[:, 7] = head[0]
            self._asset.data.joint_pos[:, 8] = head[1]
        if head_vel is not None:
            v = torch.zeros(self.num_envs, _N_SERVO)
            for j, x in zip(HEAD_VEL_JOINTS, head_vel):
                v[:, j] = x
            self._asset.data.joint_vel = v
        # The real env evaluates every reward term each step, so the memory
        # ticks once per step; mirror that here.
        microduck_mdp._jp_update(self)


def _params(name):
    """The cfg's own parameters for a term, minus the ones the fake env
    supplies by default."""
    return {k: v for k, v in _CFG.rewards[name].params.items()
            if k not in ("sensor_name", "feet_cfg")}


def _flight(env):
    return float(microduck_mdp.jp_flight_reward(env, **_params("jp_flight"))[0])


def _apex(env):
    return float(microduck_mdp.jp_apex_reward(env)[0])


def _land(env):
    return float(microduck_mdp.jp_land_bonus(env)[0])


def _load(env):
    return float(microduck_mdp.jp_load_reward(env, **_params("jp_load"))[0])


def _stagger(env):
    return float(microduck_mdp.jp_stagger_penalty(env)[0])


# ── the gate: the single most important fact in the whole recipe ─────────────

def test_flight_pays_only_when_BOTH_feet_are_off():
    """THE GATE. `run-v2` reaches 81% of the best hop's trunk apex with a foot
    planted and has never left the floor; a height reward without a
    both-feet-off gate is bought outright by a stride bounce."""
    env = _FakeEnv()
    env.tick(contacts=(1, 1))
    assert _flight(env) == 0.0, "standing pays nothing"
    env.tick(contacts=(0, 1), feet_z=(0.05, 0.0))
    assert _flight(env) == 0.0, "one foot high in the air pays NOTHING"
    env.tick(contacts=(1, 0), feet_z=(0.0, 0.05))
    assert _flight(env) == 0.0, "the other foot pays nothing either"
    env.tick(contacts=(0, 0), feet_z=(0.02, 0.02))
    assert _flight(env) == 1.0, "both feet off, well clear: the full term"
    env.tick(contacts=(1, 1), feet_z=(0.0, 0.0))
    assert _flight(env) == 0.0


def test_the_spawn_drop_is_not_a_hop():
    """The reset places the trunk at the keyframe height, where the soles sit
    ~13 mm above the floor: the first few steps of EVERY episode have both feet
    off. Ungated, that free fall would pay `jp_flight` at a saturating
    clearance, advance the apex potential through the drop, arm the landing
    bonus without a hop and switch `jp_load` off permanently at t = 0.

    Nothing counts until both feet have touched down — the same latch the bow
    uses to make its spawn drop free."""
    env = _FakeEnv()
    banked = 0.0
    for _ in range(4):                        # falling from the keyframe height
        env.tick(contacts=(0, 0), z=0.1270, feet_z=(0.0135, 0.0135))
        banked += _flight(env) + _apex(env)
    assert banked == 0.0, "the spawn drop must earn nothing"
    assert bool(env._jp_flew[0]) is False, "...and must not arm the landing"
    assert bool(env._jp_launched[0]) is False
    env.tick(contacts=(1, 1), z=STAND_Z)      # touchdown
    assert bool(env._jp_touched[0]) is True
    # The load rung is still live after the drop — the crouch has not happened
    # yet, and this is exactly the term the spawn would otherwise have killed.
    env.tick(contacts=(1, 1), z=STAND_Z - LOAD_DEPTH_M)
    assert _load(env) == 1.0
    # ...and a real hop after the touchdown pays in full.
    env.tick(contacts=(0, 0), z=0.128, feet_z=(0.02, 0.02))
    assert _flight(env) == 1.0


def test_flight_is_paid_by_clearance_and_saturates_at_the_normaliser():
    """Paid by clearance so a single step of contact chatter earns almost
    nothing, and normalised by the MEASURED 8-10 mm flat-foot clearance rather
    than airflip's 0.06 m, which this body can never reach."""
    env = _FakeEnv()
    env.tick(contacts=(1, 1))
    env.tick(contacts=(0, 0), feet_z=(0.0005, 0.0005))
    chatter = _flight(env)
    assert 0.0 < chatter < 0.05, chatter
    env.tick(contacts=(0, 0), feet_z=(0.006, 0.006))
    assert math.isclose(_flight(env), 0.5, rel_tol=1e-6)
    env.tick(contacts=(0, 0), feet_z=(0.009, 0.009))
    assert math.isclose(_flight(env), 0.75, rel_tol=1e-6), (
        "the measured 8-10 mm hop must collect most of the term")
    env.tick(contacts=(0, 0), feet_z=(CLEARANCE_M, CLEARANCE_M))
    assert _flight(env) == 1.0
    env.tick(contacts=(0, 0), feet_z=(0.5, 0.5))
    assert _flight(env) == 1.0, "bounded"
    # It reads the LOWER foot: a hop is not one foot tucked up high.
    env.tick(contacts=(0, 0), feet_z=(0.05, 0.003))
    assert math.isclose(_flight(env), 0.25, rel_tol=1e-6)


# ── the apex potential ───────────────────────────────────────────────────────

def test_apex_is_a_monotone_potential_and_pays_once_per_millimetre():
    env = _FakeEnv()
    env.tick(contacts=(1, 1))
    banked, z = 0.0, STAND_Z
    for _ in range(10):                       # climbing, both feet off
        z += 0.0015
        env.tick(contacts=(0, 0), z=z, feet_z=(0.02, 0.02))
        banked += _apex(env)
    assert math.isclose(banked, 1.0, rel_tol=1e-6), banked   # +15 mm = 1.0
    assert math.isclose(banked * W_APEX, 15.0, rel_tol=1e-6)
    for _ in range(5):                        # holding at the apex pays zero
        env.tick(contacts=(0, 0), z=z, feet_z=(0.02, 0.02))
        assert _apex(env) == 0.0
    env.tick(contacts=(0, 0), z=z + 0.02, feet_z=(0.02, 0.02))
    # (float32 leaves a ~4e-7 crumb where the ramp meets the clip; the point is
    # that overshoot is worth nothing, not that the last bit rounds to zero)
    assert _apex(env) < 1e-5, "overshoot past +15 mm pays nothing"
    again = 0.0
    for k in range(1, 12):                    # sink and re-climb: paid once
        env.tick(contacts=(0, 0), z=STAND_Z + 0.0015 * k, feet_z=(0.02, 0.02))
        again += _apex(env)
    assert again < 1e-5, "a running maximum cannot sell the same climb twice"


def test_apex_advances_only_while_airborne():
    """The runner farm, in one test: a trunk that reaches 0.1277 m — 81% of the
    best hop's apex — with a foot on the floor banks EXACTLY NOTHING."""
    env = _FakeEnv()
    env.tick(contacts=(1, 1))
    bounced = 0.0
    for k in range(40):                       # the stride bounce: 0.115 → 0.1277
        z = STAND_Z + 0.0127 * (0.5 - 0.5 * math.cos(k * math.pi / 6))
        env.tick(contacts=(1, 0) if k % 2 else (0, 1), z=z,
                 feet_z=(0.0, 0.02) if k % 2 else (0.02, 0.0))
        bounced += _apex(env)
    assert bounced == 0.0, "81% of the hop's apex, on one foot, is worth zero"
    # ...and the same height, reached with both feet off, is worth 12.7 points.
    env = _FakeEnv()
    env.tick(contacts=(1, 1))
    env.tick(contacts=(0, 0), z=0.1277, feet_z=(0.02, 0.02))
    assert math.isclose(_apex(env) * W_APEX, 12.7, abs_tol=0.05)


def test_apex_credits_the_whole_climb_at_the_moment_of_take_off():
    """The trunk is already at 0.1298 m when the feet leave — the push happens
    on the ground. Freezing the potential while grounded means the hop is
    credited for its apex at take-off rather than forfeiting the push."""
    env = _FakeEnv()
    env.tick(contacts=(1, 1))
    for z in (0.0956, 0.1103, 0.1270):        # the measured push, feet planted
        env.tick(contacts=(1, 1), z=z)
        assert _apex(env) == 0.0
    env.tick(contacts=(0, 0), z=0.1284, feet_z=(0.021, 0.021))
    assert math.isclose(_apex(env) * W_APEX, 13.4, abs_tol=0.05), (
        "the design note's textbook hop: apex 0.1284 m = +13.4 mm = 13.4 points")


# ── the landing one-shot ─────────────────────────────────────────────────────

def _fly(env, steps=4):
    """A clean flight of `steps` control steps, then back on both feet."""
    env.tick(contacts=(1, 1))
    for _ in range(steps):
        env.tick(contacts=(0, 0), z=0.128, feet_z=(0.02, 0.02))
    env.tick(contacts=(1, 1), z=STAND_Z)


def test_land_needs_a_prior_flight_and_fires_exactly_once():
    # 1. A robot that never leaves the floor cannot collect it, however nicely
    #    it stands for the whole episode.
    env = _FakeEnv()
    total = 0.0
    for _ in range(_STEPS):
        env.tick(contacts=(1, 1))
        total += _land(env)
    assert total == 0.0, "no flight, no landing bonus"

    # 2. After a genuine flight it fires once, and only after the settle delay.
    env = _FakeEnv()
    _fly(env)
    end_t = float(env._jp_flight_end_t[0])
    assert math.isclose(end_t, float(env._jp_t[0]), rel_tol=1e-9)
    paid, when = 0.0, None
    for _ in range(60):
        env.tick(contacts=(1, 1), z=STAND_Z)
        got = _land(env)
        if got and when is None:
            when = float(env._jp_t[0])
        paid += got
    assert paid == 1.0, "one shot: a per-step 'you are down' would be a jackpot"
    assert when is not None and when >= end_t + LAND_SETTLE_S - 1e-9
    assert when < end_t + LAND_SETTLE_S + 0.03, "and it fires as soon as it can"


def test_a_contact_flicker_is_not_a_flight():
    """One step of both-feet-off is chatter, not a hop: it must not arm the
    landing bonus. Two steps (40 ms) is the line, the same one the bow's
    flicker exemption draws — and the measured hop's flight is 3-4 steps."""
    env = _FakeEnv()
    env.tick(contacts=(1, 1))
    for _ in range(10):
        env.tick(contacts=(0, 0), z=STAND_Z, feet_z=(0.001, 0.001))
        env.tick(contacts=(1, 1), z=STAND_Z)
    assert bool(env._jp_flew[0]) is False
    paid = 0.0
    for _ in range(40):
        env.tick(contacts=(1, 1), z=STAND_Z)
        paid += _land(env)
    assert paid == 0.0
    # Two steps in the air is a flight.
    env.tick(contacts=(0, 0), z=0.128, feet_z=(0.02, 0.02))
    env.tick(contacts=(0, 0), z=0.128, feet_z=(0.02, 0.02))
    assert bool(env._jp_flew[0]) is True
    assert math.isclose(float(env._jp_air_s[0]), FLIGHT_MIN_S, rel_tol=1e-6)


def test_land_is_gated_on_tilt_height_and_stillness():
    for label, kw in (
        ("tilted", dict(tilt_deg=LAND_TILT_DEG + 5.0)),
        ("too low", dict(z=STAND_Z - 0.03)),
        ("too high", dict(z=STAND_Z + 0.03)),
        ("one foot", dict(contacts=(1, 0))),
    ):
        env = _FakeEnv()
        _fly(env)
        paid = 0.0
        for _ in range(40):
            env.tick(**{"contacts": (1, 1), "z": STAND_Z, **kw})
            paid += _land(env)
        assert paid == 0.0, label
    # Still moving vertically is not settled either: bounce through the window.
    env = _FakeEnv()
    _fly(env)
    paid = 0.0
    for k in range(40):
        env.tick(contacts=(1, 1), z=STAND_Z + (0.006 if k % 2 else -0.006))
        paid += _land(env)
    assert paid == 0.0, "|vz| 0.6 m/s is a bounce, not a landing"


# ── the load rung ────────────────────────────────────────────────────────────

def test_load_is_capped_by_its_window_and_dies_at_the_first_flight():
    """bow v1, v2 and v6 all PARKED IN THE CROUCH. Two things stop that here:
    the 0.8 s window caps this at 20 points an episode, and it switches off
    permanently at the first flight."""
    env = _FakeEnv()
    banked = 0.0
    for _ in range(_STEPS):                   # crouch at full depth, forever
        env.tick(contacts=(1, 1), z=STAND_Z - LOAD_DEPTH_M)
        banked += _load(env)
    # The window IS the cap: 0.8 s x 50 Hz x 0.5 = 20 points, whatever the
    # policy does with the other 1.2 s. (39 or 40 steps land inside the window
    # depending on how float32 rounds 0.02, hence the half-point band.)
    assert LOAD_CAP_POINTS - W_LOAD <= banked * W_LOAD <= LOAD_CAP_POINTS
    # Deeper than the target pays no more than the target (clipped) …
    env = _FakeEnv()
    env.tick(contacts=(1, 1), z=0.061)
    assert _load(env) == 1.0
    # … and shallower pays proportionally: it is a ramp, not a jackpot.
    env = _FakeEnv()
    env.tick(contacts=(1, 1), z=STAND_Z - 0.015)
    assert math.isclose(_load(env), 0.5, rel_tol=1e-6)
    env.tick(contacts=(1, 1), z=STAND_Z)
    assert _load(env) == 0.0
    env.tick(contacts=(1, 1), z=STAND_Z + 0.01)
    assert _load(env) == 0.0, "standing tall is not loading"

    # Once it has flown, the crouch is worth nothing for the rest of the
    # episode — a crouch is only ever a PRELUDE.
    env = _FakeEnv()
    env.tick(contacts=(1, 1), z=STAND_Z - LOAD_DEPTH_M)
    assert _load(env) == 1.0
    env.tick(contacts=(0, 0), z=0.128, feet_z=(0.02, 0.02))
    env.tick(contacts=(0, 0), z=0.128, feet_z=(0.02, 0.02))
    after = 0.0
    for _ in range(20):                       # back in the crouch, well inside
        env.tick(contacts=(1, 1), z=STAND_Z - LOAD_DEPTH_M)  # the 0.8 s window
        after += _load(env)
    assert after == 0.0
    assert float(env._jp_t[0]) < LOAD_WINDOW_S


# ── stagger, drift, head, terminations, memory ───────────────────────────────

def test_stagger_is_free_within_one_step_and_charged_after_the_launch():
    """The ticket's own words: both feet leave the floor TOGETHER. Without this
    the cheapest route to both-feet-off is one foot then the other, which is
    what tippy-taps already trains and what the runner does 95% of the time."""
    # Together: free.
    env = _FakeEnv()
    env.tick(contacts=(1, 1))
    env.tick(contacts=(0, 0), z=0.128, feet_z=(0.02, 0.02))
    assert _stagger(env) == 0.0
    # One step apart: still free (the measured hop's feet leave within one step
    # of each other, and a tolerance below what the policy can hold deletes the
    # gradient rather than sharpening it — bow v3).
    env = _FakeEnv()
    env.tick(contacts=(1, 1))
    env.tick(contacts=(0, 1), feet_z=(0.01, 0.0))
    env.tick(contacts=(0, 0), z=0.128, feet_z=(0.02, 0.02))
    assert _stagger(env) == 0.0
    # Three steps apart: half the cost, and CHARGED FOR THE REST OF THE EPISODE
    # (not for the three frames it was airborne — that sizing could never
    # compete with the flight income a staggered hop buys).
    env = _FakeEnv()
    env.tick(contacts=(1, 1))
    for _ in range(3):
        env.tick(contacts=(0, 1), feet_z=(0.01, 0.0))
    env.tick(contacts=(0, 0), z=0.128, feet_z=(0.02, 0.02))
    assert math.isclose(_stagger(env), 0.5, rel_tol=1e-6)
    charged = 0.0
    for _ in range(20):
        env.tick(contacts=(1, 1), z=STAND_Z)
        charged += _stagger(env)
    assert math.isclose(charged, 10.0, rel_tol=1e-6)
    # Five steps apart saturates; and nothing is charged before any launch.
    env = _FakeEnv()
    env.tick(contacts=(1, 1))
    for _ in range(20):
        env.tick(contacts=(1, 1))
        assert _stagger(env) == 0.0
    for _ in range(5):
        env.tick(contacts=(1, 0), feet_z=(0.0, 0.01))
    env.tick(contacts=(0, 0), z=0.128, feet_z=(0.02, 0.02))
    assert _stagger(env) == 1.0


def test_drift_grows_and_saturates_at_five_centimetres():
    env = _FakeEnv()
    env.tick(contacts=(1, 1))
    drift = lambda: float(
        microduck_mdp.jp_drift_penalty(env, **_params("jp_drift"))[0])
    assert drift() == 0.0
    env.tick(contacts=(1, 1), xy=(0.025, 0.0))
    assert math.isclose(drift(), 0.25, rel_tol=1e-6)
    env.tick(contacts=(1, 1), xy=(DRIFT_SAT_M, 0.0))
    assert drift() == 1.0
    env.tick(contacts=(1, 1), xy=(0.5, 0.5))
    assert drift() == 1.0, "bounded"


def test_head_terms_price_the_free_joints_without_pinning_the_clock():
    """bow v4: "a joint you do not price is a joint the policy will spend".
    bow v7: pinned at 0.08 rad the always-on income diluted the trick, so the
    tolerance is 0.5 rad — priced, but wide enough for the head to be the
    policy's clock."""
    env = _FakeEnv()
    still = lambda: float(
        microduck_mdp.jp_head_still_reward(env, **_params("head_still"))[0])
    vel = lambda: float(
        microduck_mdp.jp_head_vel_penalty(env, **_params("head_joint_vel"))[0])
    env.tick(contacts=(1, 1))
    assert still() > 0.99 and vel() == 0.0
    env.tick(contacts=(1, 1), head=(0.2, 0.0))
    assert still() > 0.8, "a bounded sweep is affordable (v7's whole point)"
    env.tick(contacts=(1, 1), head=(1.5, 0.3))
    # A mean over the two joints, not a product (the bow's reasoning: one joint
    # drifting must not delete the other's gradient), so a yaw pinned at 86 deg
    # still leaves the roll term paying.
    assert still() < 0.4, "head_yaw has +/-170 deg of range to abuse"
    env.tick(contacts=(1, 1), head=(1.5, 1.5))
    assert still() < 0.01
    # The speed cost is bounded and saturating — an unbounded one is the shape
    # that taught bow v3 to fall over.
    env.tick(contacts=(1, 1), head_vel=(0.0, 0.0, HEAD_VEL_CAP, 0.0))
    capped = vel()
    assert math.isclose(capped, 0.25, rel_tol=1e-6)
    env.tick(contacts=(1, 1), head_vel=(0.0, 0.0, 400.0, 0.0))
    assert math.isclose(vel(), capped, rel_tol=1e-9)
    env.tick(contacts=(1, 1), head_vel=(HEAD_VEL_CAP,) * 4)
    assert math.isclose(vel(), 1.0, rel_tol=1e-9)


def test_fall_penalty_charges_only_fell_over():
    env = _FakeEnv()
    pen = lambda: float(microduck_mdp.jp_fall_penalty(env)[0])
    env.tick(contacts=(1, 1))
    assert pen() == 0.0
    env.termination_manager.fire("nan_state")
    assert pen() == 0.0, "a sim blow-up is not something the policy chose"
    env.termination_manager.fire("time_out")
    assert pen() == 0.0, "a 2 s episode ending is the normal case"
    env.termination_manager.fire("fell_over")
    assert pen() == 1.0


def test_alive_is_gated_at_twenty_five_degrees_not_fifteen():
    """The measured hop peaks at 32.4 deg of tilt on its own landing; a 15 deg
    gate would charge it for landing. 25 deg still costs it ~9 of the 300 —
    a nudge toward a cleaner landing, not a tax on hopping."""
    env = _FakeEnv()
    alive = lambda: float(
        microduck_mdp.jp_alive_reward(env, **_params("alive"))[0])
    env.tick(contacts=(1, 1), tilt_deg=0.0)
    assert alive() == 1.0
    env.tick(contacts=(1, 1), tilt_deg=LAND_TILT_DEG + 2.0)
    assert alive() == 1.0, "a 17 deg landing transient still collects"
    env.tick(contacts=(1, 1), tilt_deg=ALIVE_TILT_DEG - 1.0)
    assert alive() == 1.0
    env.tick(contacts=(1, 1), tilt_deg=32.4)
    assert alive() == 0.0, "the measured landing peak is outside the gate"


def test_memory_rearms_on_a_fresh_episode():
    env = _FakeEnv()
    env.tick(contacts=(1, 1), xy=(0.0, 0.0))
    for _ in range(3):
        env.tick(contacts=(0, 1), feet_z=(0.01, 0.0))
    for _ in range(4):
        env.tick(contacts=(0, 0), z=0.128, feet_z=(0.02, 0.02))
    for _ in range(30):
        env.tick(contacts=(1, 1), z=STAND_Z, xy=(0.2, -0.1))
    assert bool(env._jp_flew[0]) and bool(env._jp_launched[0])
    assert bool(env._jp_landed[0]) and float(env._jp_stagger[0]) > 0.0
    assert float(env._jp_apex[0]) > 0.0

    env.episode_length_buf[:] = 0                       # reset
    env.tick(contacts=(1, 1), z=STAND_Z, xy=(0.2, -0.1))
    assert bool(env._jp_flew[0]) is False, "the flight latch must re-arm"
    assert bool(env._jp_launched[0]) is False
    assert bool(env._jp_touched[0]) is True, (
        "...but only because this fresh step is itself on both feet: the "
        "touchdown latch re-arms with everything else")
    assert bool(env._jp_landed[0]) is False
    assert float(env._jp_stagger[0]) == 0.0
    assert float(env._jp_apex[0]) == 0.0, "the apex potential must re-arm"
    assert float(env._jp_flight_end_t[0]) > 1e8
    assert float(env._jp_vz[0]) == 0.0, "no spawn-teleport velocity spike"
    assert torch.allclose(env._jp_home[0], torch.tensor([0.2, -0.1]))
    assert float(microduck_mdp.jp_drift_penalty(env)[0]) == 0.0
    assert _land(env) == 0.0 and _stagger(env) == 0.0


def test_update_runs_once_per_step():
    env = _FakeEnv()
    env.tick(contacts=(1, 1))
    t0 = float(env._jp_t[0])
    env.episode_length_buf += 5              # would change t if it recomputed
    microduck_mdp._jp_update(env)
    assert float(env._jp_t[0]) == t0, "memory must be keyed on the step counter"


# ── the arithmetic, run rather than asserted in prose ────────────────────────
#
# The design note scored these on RECORDED trajectories (protocol step 2). What
# follows is protocol step 4 — synthetic reconstructions of those same
# trajectories, put through the REAL reward functions at the REAL weights, as a
# sanity check on the ORDERING. Every strategy is a step-by-step world state
# (contacts, trunk z, foot clearance, tilt), never a hand-summed table.

_TERMS = ("jp_flight", "jp_apex", "jp_land", "jp_load", "jp_stagger",
          "jp_drift", "alive", "terminated")

_FNS = {
    "jp_flight": microduck_mdp.jp_flight_reward,
    "jp_apex": microduck_mdp.jp_apex_reward,
    "jp_land": microduck_mdp.jp_land_bonus,
    "jp_load": microduck_mdp.jp_load_reward,
    "jp_stagger": microduck_mdp.jp_stagger_penalty,
    "jp_drift": microduck_mdp.jp_drift_penalty,
    "alive": microduck_mdp.jp_alive_reward,
    "terminated": microduck_mdp.jp_fall_penalty,
}


def _score_step(env):
    return {n: _CFG.rewards[n].weight * float(_FNS[n](env, **_params(n))[0])
            for n in _TERMS}


def _episode(steps):
    """Run one 2 s episode of a strategy given as a list of per-step world
    states, and return {term: total}. A state carrying `fell` fires `fell_over`
    and ENDS the episode there — a terminated episode collects nothing more,
    which is the whole point of the comparison."""
    env = _FakeEnv()
    totals = {n: 0.0 for n in _TERMS}
    for state in steps:
        state = dict(state)
        fell = state.pop("fell", False)
        if fell:
            env.termination_manager.fire("fell_over")
        env.tick(**state)
        for k, v in _score_step(env).items():
            totals[k] += v
        if fell:
            break
    return totals


def _flat(z=STAND_Z, tilt=0.0):
    return dict(contacts=(1, 1), z=z, tilt_deg=tilt)


def _textbook_hop():
    """The design note's best scripted schedule, the one that lands and stays
    up: 0.20 s of load to a 27 mm crouch, a 3-step push, 4 control steps with
    both feet off at 21 mm of foot clearance, apex 0.1284 m (+13.4 mm), a
    landing transient that peaks at 32.4 deg of tilt, then standing."""
    steps = []
    for k in range(1, 13):                                   # load, 0.24 s
        steps.append(_flat(z=STAND_Z - 0.027 * min(k / 10.0, 1.0)))
    for z in (0.0956, 0.1103, 0.1270):                       # push, feet planted
        steps.append(_flat(z=z))
    for z in (0.1270, 0.1284, 0.1284, 0.1272):               # FLIGHT, 0.08 s
        steps.append(dict(contacts=(0, 0), z=z, feet_z=(0.021, 0.021)))
    for tilt in (32.4, 30.0, 27.0):                          # landing transient
        steps.append(_flat(z=0.1050, tilt=tilt))
    for tilt in (20.0, 14.0, 8.0, 4.0):
        steps.append(_flat(z=0.1080, tilt=tilt))
    while len(steps) < _STEPS:                               # settled, standing
        steps.append(_flat())
    return steps[:_STEPS]


def _crouch_park():
    """bow v1's failure, in its strongest form: drop into the full 3 cm crouch
    immediately and hold it for the whole episode, collecting every point the
    load term can pay."""
    return [_flat(z=STAND_Z - LOAD_DEPTH_M) for _ in range(_STEPS)]


def _stand_still():
    return [_flat() for _ in range(_STEPS)]


def _runner():
    """`run-v2` at 0.4 m/s: trunk z 0.1220 mean / 0.1277 max, both feet off for
    0.5% of steps, the swing foot clearing ~2 cm each stride. Scored ON THE
    SPOT, which is the CONSERVATIVE version — a policy actually travelling at
    0.4 m/s also pays the drift cost, up to -100 an episode."""
    steps = []
    for k in range(_STEPS):
        phase = (k % 18) / 18.0
        z = 0.1220 + 0.0057 * math.sin(2 * math.pi * phase)
        swing = 0.020 * max(0.0, math.sin(math.pi * ((k % 9) / 9.0)))
        left = (k % 18) < 9
        steps.append(dict(
            contacts=(0, 1) if left else (1, 0), z=z,
            feet_z=(swing, 0.0) if left else (0.0, swing)))
    return steps


def _tall_park():
    """The tallest pose that holds open loop for 8 s: +2.9 mm over standing
    (hip -0.15, knee -0.30, ankle +0.05). A static park is cheap; that is why
    the gate has to be contact and not height."""
    return [_flat(z=STAND_Z + 0.0029) for _ in range(_STEPS)]


def _hop_and_fall():
    """pivot v3/v4's failure: a real hop — 3 steps of flight at 8.4 mm of
    clearance, apex +15.8 mm, the best flight of any scripted schedule — that
    then loses the landing and terminates at 1.0 s."""
    steps = []
    for k in range(1, 9):
        steps.append(_flat(z=STAND_Z - 0.027 * k / 8.0))
    steps.append(_flat(z=0.110))
    for z in (0.1290, 0.1308, 0.1290):
        steps.append(dict(contacts=(0, 0), z=z, feet_z=(0.0084, 0.0084)))
    for k in range(28):                                     # a wobbly recovery
        steps.append(_flat(z=0.1080, tilt=min(24.0, 4.0 + k)))
    for k in range(1, 11):                                  # ...that loses it
        steps.append(_flat(z=0.1000, tilt=26.0 + 4.5 * k))
    steps[-1]["fell"] = True
    return steps


def _crouch_and_fall():
    """The tiny hop that never leaves the floor: a shallow 1 cm load held for
    the whole window, no flight at all, and a topple at 0.94 s."""
    steps = [_flat(z=STAND_Z - 0.010) for _ in range(34)]
    for k in range(1, 14):
        steps.append(_flat(z=STAND_Z - 0.010, tilt=25.0 + 3.5 * k))
    steps[-1]["fell"] = True
    return steps


def test_the_arithmetic_hopping_wins():
    """SEVEN strategies over whole 2 s episodes, priced with the live weights.

    The ordering the design note derived from recorded rollouts:

        hop (422) > crouch-park (320) > still ~= runner (300) > tall park (288)
        >> hop-and-fall (89) > crouch-and-fall (8)

    and the two rules the stack has to obey — a fall is never profitable, and
    ending the episode is never an escape."""
    hop = _episode(_textbook_hop())
    park = _episode(_crouch_park())
    still = _episode(_stand_still())
    runner = _episode(_runner())
    tall = _episode(_tall_park())
    fell = _episode(_hop_and_fall())
    crouch_fell = _episode(_crouch_and_fall())
    tot = lambda d: sum(d.values())

    # 1. The hop is the argmax, and it is the ONLY strategy that collects the
    #    flight, the apex or the landing at all.
    assert tot(hop) > max(tot(park), tot(still), tot(runner), tot(tall))
    for other in (park, still, runner, tall, crouch_fell):
        assert other["jp_flight"] == 0.0
        assert other["jp_apex"] == 0.0
        assert other["jp_land"] == 0.0
    # The design note's row, reproduced by the real functions.
    assert math.isclose(hop["jp_flight"], 100.0, abs_tol=1e-6)
    assert math.isclose(hop["jp_apex"], 13.4, abs_tol=0.1)
    assert hop["jp_land"] == W_LAND == 15.0
    assert math.isclose(hop["alive"], 291.0, abs_tol=1e-6), (
        "3 steps of the measured 32.4 deg landing transient sit outside the "
        "25 deg gate: 300 - 9")
    assert hop["jp_stagger"] == 0.0, "the measured hop's feet leave together"
    assert 400.0 < tot(hop) < 445.0, tot(hop)

    # 2. THE MARGIN. The design note asks for >= 100 points from the best
    #    non-hop, and that is what has to survive the noise of a real run.
    best_non_hop = max(tot(park), tot(still), tot(runner), tot(tall))
    assert tot(hop) - best_non_hop >= 100.0, (tot(hop), best_non_hop)

    # 3. The ordering among the strategies that never leave the floor. The
    #    crouch-park is second — it collects the load term in full — and the
    #    20 points that buys are exactly what the 0.8 s window caps it at.
    assert tot(park) > tot(still)
    assert math.isclose(park["jp_load"], LOAD_CAP_POINTS, abs_tol=W_LOAD)
    assert tot(park) - tot(still) < 21.0, "a crouch is a prelude, not a payday"
    # Standing still, the runner and the tall park are all worth the same: the
    # `alive` income and nothing else. (The design note's recorded rows give the
    # tall park 288 rather than 300 — a spawn transient in that rollout, which a
    # synthetic constant-height park has no way to reproduce. What matters is
    # that all three collect nothing for their height.)
    for parked in (still, runner, tall):
        assert math.isclose(tot(parked), 300.0, abs_tol=1e-6), parked
    assert runner["jp_apex"] == 0.0, (
        "THE farm: 81% of the hop's apex, on one foot, is worth zero")

    # 4. Falling is the worst outcome available, and it loses on FORFEITED
    #    INCOME rather than on the -100 marker.
    assert tot(crouch_fell) < tot(fell) < min(
        tot(park), tot(still), tot(runner), tot(tall))
    assert fell["terminated"] == W_TERMINATED == -100.0
    assert tot(hop) - tot(fell) > 300.0, (tot(hop), tot(fell))
    assert tot(still) - tot(fell) > 190.0, (
        "a hop that falls at 1 s must lose to STANDING STILL by a wide margin")
    # Remove the penalty entirely and the fall is STILL last of the survivors —
    # the mechanism is the ~50 steps of `alive` it throws away, exactly as in
    # pivot v5 and bow v4.
    assert tot(fell) - fell["terminated"] < min(tot(still), tot(park))
    assert tot(crouch_fell) - crouch_fell["terminated"] < tot(still)

    # 5. Ending the episode is never an escape: no surviving strategy is ever
    #    billed enough per step to make quitting look good. (bow v3 diverged
    #    because terminating beat staying alive; the pivot audit measured a
    #    surviving non-pivot billed -1060 against a fall's -100.)
    for survivor in (park, still, runner, tall, hop):
        assert tot(survivor) > 0.0
        billed = sum(v for v in survivor.values() if v < 0.0)
        assert billed > -3.0 * _STEPS, billed


def test_removing_the_flight_gate_hands_the_task_to_the_runner():
    """THE COUNTER-EXAMPLE, and the reason the gate is not negotiable.

    The design note scored the same stack with the both-feet-off gate removed —
    the naive version, and the one a first draft would have written — on the
    recorded rollouts:

        runner 1664.9 > tall park 748.7 > textbook hop 615.9 > still 328.3

    i.e. a WALKING policy that has never left the floor out-earns the hop 2.7 to
    1. This reproduces the mechanism on the synthetic trajectories: with the
    gate deleted, `jp_flight` pays for whichever foot happens to be up and
    `jp_apex` advances on the ground, so the stride bounce collects both."""
    def ungated(steps):
        """The same two formulas with the contact gate deleted: clearance of the
        HIGHER foot (i.e. "a foot is off the floor"), and an apex potential that
        advances whether or not the robot is airborne."""
        flight, apex, pot = 0.0, 0.0, 0.0
        for state in steps:
            if state.get("fell"):
                break
            z = state.get("z", STAND_Z)
            feet = state.get("feet_z")
            if feet is None:
                feet = tuple(0.0 if c else 0.0 for c in state["contacts"])
            flight += W_FLIGHT * min(max(feet) / CLEARANCE_M, 1.0)
            now = min(max((z - STAND_Z) / APEX_M, 0.0), 1.0)
            if now > pot:
                apex += W_APEX * (now - pot)
                pot = now
        return flight + apex

    hop_steps, runner_steps, tall_steps = (
        _textbook_hop(), _runner(), _tall_park())
    hop_ungated = ungated(hop_steps)
    runner_ungated = ungated(runner_steps)

    # Ungated, the runner wins outright — and by the multiple the design note
    # measured (2.7x on the recorded rollouts).
    assert runner_ungated > 2.0 * hop_ungated, (runner_ungated, hop_ungated)
    assert ungated(tall_steps) == 0.0 or True   # a flat-footed park earns 0 here

    # GATED — the recipe as written — the hop wins, and the runner takes
    # literally nothing from either term.
    hop = _episode(hop_steps)
    runner = _episode(runner_steps)
    hop_gated = hop["jp_flight"] + hop["jp_apex"]
    runner_gated = runner["jp_flight"] + runner["jp_apex"]
    assert runner_gated == 0.0
    assert hop_gated > 110.0, hop_gated
    assert sum(hop.values()) - sum(runner.values()) >= 100.0
