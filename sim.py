"""A Panda arm turning a lid in robosuite, with a few motion primitives.

The lid is a short cylinder standing on the table. The arm grasps it from
above and turns it with the last joint only (joint 7, "the wrist"), whose
axis is vertical when the gripper points down. Joint 7 goes from -166 to
+166 degrees, so the rotation available in one stroke depends on where the
wrist starts.

All angles are in degrees. Lid angles are counterclockwise seen from above
(the unscrewing direction); turning the lid counterclockwise makes joint 7
decrease.
"""

import mujoco
import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

CONTROL_FREQ = 20       # actions per second
WRIST_LIMIT = 166.0     # joint 7 range is [-166, 166]
WRIST_MARGIN = 5.0      # a stroke stops this far from the limit
LIFT = 0.04             # height of the gripper above the lid when not holding it


class LidTurnEnv(Lift):
    """The robosuite Lift task with the cube replaced by a lid at a fixed place."""

    def _load_model(self):
        ManipulationEnv._load_model(self)
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size,
                           table_friction=self.table_friction,
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        # Lift calls its object "cube" everywhere, so we keep the attribute name
        self.cube = CylinderObject(name="lid", size=[0.03, 0.02], rgba=[0.9, 0.1, 0.1, 1])
        self.placement_initializer = UniformRandomSampler(
            name="sampler", mujoco_objects=self.cube, x_range=[0, 0], y_range=[0, 0],
            rotation=0.0, ensure_object_boundary_in_range=False, ensure_valid_placement=True,
            reference_pos=self.table_offset, z_offset=0.01)
        self.model = ManipulationTask(mujoco_arena=arena,
                                      mujoco_robots=[r.robot_model for r in self.robots],
                                      mujoco_objects=self.cube)


def joint_position_config():
    """Controller where each action is the 7 target joint angles (radians) + the gripper."""
    config = load_composite_controller_config(robot="Panda")
    arm = config["body_parts"]["right"]
    for key in ["output_max", "output_min", "uncouple_pos_ori", "input_ref_frame",
                "position_limits", "orientation_limits"]:
        arm.pop(key, None)
    arm.update(type="JOINT_POSITION", input_type="absolute", kp=150, damping_ratio=1,
               interpolation=None)
    return config


def wrap(angle):
    """Brings an angle in degrees back to [-180, 180)."""
    return (angle + 180) % 360 - 180


class LidSim:
    """The environment plus what we follow during an episode."""

    def __init__(self):
        self.env = LidTurnEnv(robots="Panda", controller_configs=joint_position_config(),
                              has_renderer=False, has_offscreen_renderer=False,
                              use_camera_obs=False, control_freq=CONTROL_FREQ,
                              horizon=100000, initialization_noise=None)
        self.robot = self.env.robots[0]

    def reset(self, wrist_start=45.0, wrist_limit=WRIST_LIMIT):
        """New episode with joint 7 at wrist_start (45 is the robosuite default).

        wrist_limit < 166 simulates a wrist with less range than the Panda's,
        e.g. closer to a human hand: strokes stop at +-wrist_limit.
        """
        self.wrist_limit = wrist_limit
        self.obs = self.env.reset()
        self.lid_pos = self.obs["cube_pos"].copy()
        q = self.obs["robot0_joint_pos"].copy()
        q[6] = np.radians(wrist_start)
        # arm postures (joints 1 to 6) with the gripper pointing down, above and on the lid
        self.arm_above = self._solve_arm(self.lid_pos + [0, 0, LIFT], q)
        self.arm_grasp = self._solve_arm(self.lid_pos, self.arm_above)
        self.wrist_target = wrist_start
        self.holding = False
        self.steps = 0
        self._lid_raw = self._read_lid_yaw()
        self.lid_angle = 0.0
        self.history = []
        self._settle(self.arm_above, -1)   # move to the start posture, above the lid
        # the episode starts now: time, lid angle and counters from zero
        self.steps = 0
        self.grasps = 0
        self.wrist_limit_hits = 0
        self.lid_angle = 0.0
        self.history = []          # one row per step: time, lid, wrist, wrist target, grip command
        return self

    # --- what we can read ---

    def wrist(self):
        return np.degrees(self.obs["robot0_joint_pos"][6])

    def time(self):
        return self.steps / CONTROL_FREQ

    def room_ccw(self):
        """Counterclockwise lid rotation still possible before the wrist limit."""
        return self.wrist_target - (-self.wrist_limit + WRIST_MARGIN)

    def room_cw(self):
        """Clockwise rotation possible before the other wrist limit."""
        return (self.wrist_limit - WRIST_MARGIN) - self.wrist_target

    def full_range(self):
        """Largest stroke the wrist allows, from one limit to the other."""
        return 2 * (self.wrist_limit - WRIST_MARGIN)

    def lid_drift(self):
        """How far the lid has slid on the table, in metres."""
        return np.linalg.norm(self.obs["cube_pos"][:2] - self.lid_pos[:2])

    def _read_lid_yaw(self):
        quat = T.convert_quat(self.env.sim.data.body_xquat[self.env.cube_body_id], to="xyzw")
        x_axis = T.quat2mat(quat)[:, 0]
        return np.degrees(np.arctan2(x_axis[1], x_axis[0]))

    # --- plumbing ---

    def _solve_arm(self, target, q_start):
        """Joints 1 to 6 putting the gripper at target, pointing down (inverse kinematics).

        Gauss-Newton on the kinematic model only; the simulation state is restored after.
        """
        model, data = self.env.sim.model._model, self.env.sim.data._data
        site = self.robot.eef_site_id["right"]
        pos_idx = self.robot._ref_joint_pos_indexes
        vel_idx = self.robot._ref_joint_vel_indexes[:6]
        saved = data.qpos.copy()
        data.qpos[pos_idx] = q_start
        jac_pos, jac_rot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for _ in range(100):
            mujoco.mj_forward(model, data)
            gripper_axis = data.site_xmat[site].reshape(3, 3)[:, 2]
            error = np.concatenate([target - data.site_xpos[site],
                                    np.cross(gripper_axis, [0, 0, -1])])
            mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site)
            jac = np.vstack([jac_pos, jac_rot])[:, vel_idx]
            data.qpos[pos_idx[:6]] += np.linalg.pinv(jac) @ error
        solution = data.qpos[pos_idx].copy()
        data.qpos[:] = saved
        mujoco.mj_forward(model, data)
        return solution

    def _step(self, arm, grip):
        """One step towards an arm posture with joint 7 at wrist_target. grip: -1 open, 1 closed."""
        q = arm.copy()
        q[6] = np.radians(self.wrist_target)
        self.obs, _, _, _ = self.env.step(np.append(q, grip))
        self.steps += 1
        lid_raw = self._read_lid_yaw()
        self.lid_angle += wrap(lid_raw - self._lid_raw)
        self._lid_raw = lid_raw
        self.history.append((self.time(), self.lid_angle, self.wrist(), self.wrist_target, grip))

    def _settle(self, arm, grip, max_steps=60):
        """Keeps the same command until the wrist is within 1 degree of its target (3 s max)."""
        for _ in range(max_steps):
            self._step(arm, grip)
            if abs(self.wrist() - self.wrist_target) < 1:
                break

    # --- primitives ---

    def grasp(self):
        """Goes down on the lid and closes the gripper."""
        for _ in range(15):
            self._step(self.arm_grasp, -1)
        for _ in range(10):
            self._step(self.arm_grasp, 1)
        self.holding = True
        self.grasps += 1

    def release(self):
        """Opens the gripper and moves up."""
        for _ in range(8):
            self._step(self.arm_grasp, -1)
        self.holding = False
        for _ in range(12):
            self._step(self.arm_above, -1)

    def turn(self, angle, speed):
        """Turns the wrist by angle (deg, positive = counterclockwise) at speed (deg/s).

        Holding the lid, this turns the lid; open, it only moves the gripper back.
        Stops early at the wrist limit. Returns how much the lid turned.
        """
        arm = self.arm_grasp if self.holding else self.arm_above
        grip = 1 if self.holding else -1
        lid_start = self.lid_angle
        per_step = speed / CONTROL_FREQ
        left = abs(angle)
        while left > 1e-9:
            room = self.room_ccw() if angle > 0 else self.room_cw()
            if room <= 0:
                self.wrist_limit_hits += 1
                break
            delta = min(per_step, left, room)
            self.wrist_target -= delta * np.sign(angle)
            left -= delta
            self._step(arm, grip)
        self._settle(arm, grip)
        return self.lid_angle - lid_start
