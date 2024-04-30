import json
import numpy as np
import transforms3d as t3d

from tongverselite.tcp import PersistentTcpClient, json2bin
from tongverselite.solver import TaskSolverBase


class BipedWalkingCtrlClient(PersistentTcpClient):
    def send_request(self, msg):
        data_bin = json2bin(msg)
        return json.loads(self.send(data_bin).decode("ascii"))

    def get_cmd(self, obs, vx, vy, theta, state):
        obs_agent = obs["agent"]
        q_leg = obs_agent["joint_state"]["legs_positions"]
        dq_leg = obs_agent["joint_state"]["legs_velocities"]

        q_arm = obs_agent["joint_state"]["arms_positions"]
        dq_arm = obs_agent["joint_state"]["arms_velocities"]

        p_wb = obs_agent["body_state"]["world_pos"]
        quat_wb = obs_agent["body_state"]["world_orient"]
        v_wb = obs_agent["body_state"]["linear_velocities"]
        w_wb = obs_agent["body_state"]["angular_velocities"]

        msg = {
            "q_leg": q_leg.tolist(),
            "dq_leg": dq_leg.tolist(),
            "q_arm": q_arm.tolist(),
            "dq_arm": dq_arm.tolist(),
            "p_wb": p_wb.tolist(),
            "quat_wb": quat_wb.tolist(),
            "v_wb": v_wb.tolist(),
            "w_wb": w_wb.tolist(),
            "command": [vx, vy, theta],
            "change_state": state,
        }
        joint_efforts = self.send_request(msg)

        return joint_efforts


class TaskSolver(TaskSolverBase):
    def __init__(self) -> None:
        super().__init__()

        # create a controller client that request joint effort commands from the server
        # note that ip MUST be "0.0.0.0" and port MUST be 8800
        self.ctrl_client_ = BipedWalkingCtrlClient(ip="0.0.0.0", port=8800)
        
        # the goal position as given in the task specification
        self.goal_ = np.array([1.94795, -9.1162, 0.1])

    def next_action(self, obs: dict) -> dict:
        # get the position and rotation of the agent
        agent = obs["agent"]
        pos = agent["body_state"]["world_pos"]
        quat = agent["body_state"]["world_orient"]
        rpy = t3d.euler.quat2euler(quat, axes="sxyz")

        # calculate the difference between current position and the goal
        dx = self.goal_[0] - pos[0]
        dy = self.goal_[1] - pos[1]

        # calculate the target direction
        target_dir = np.arctan2(dy, dx)

        # calculate the difference between the target direction and the current direction
        diff_rot = target_dir - rpy[2]

        # round the rotation difference, to avoid the agent to rotate more than 180 degrees
        if abs(diff_rot) > np.pi:
            diff_rot += 2 * np.pi if diff_rot < 0 else -2 * np.pi

        # if the agent is nearly facing the goal, move forward
        if abs(diff_rot) < np.pi / 90:
            # call bipedal controller to get joint effort given a target velocity
            joint_efforts = self.ctrl_client_.get_cmd(
                obs, 0.3, 0, 0, False
            )
        elif diff_rot > 0:
            # call bipedal controller to get joint effort given a target angular velocity
            joint_efforts = self.ctrl_client_.get_cmd(
                obs, 0, 0, 0.1, False
            )
        else:
            # call bipedal controller to get joint effort given a target angular velocity
            joint_efforts = self.ctrl_client_.get_cmd(
                obs, 0, 0, -0.1, False
            )

        # wrap joint effort into tongverse-lite action format
        action = {
            "legs": {
                "ctrl_mode": joint_efforts["mode"],
                "joint_values": joint_efforts["effort"],
                "stiffness": [],
                "dampings": [],
            }
        }

        return action
