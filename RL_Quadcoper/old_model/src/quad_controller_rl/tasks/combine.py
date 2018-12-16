"""Combine task"""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask


class Combine(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([-cube_size / 2, -cube_size / 2, 0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([cube_size / 2, cube_size / 2, cube_size, 1.0, 1.0, 1.0, 1.0]))
        # print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([max_force, max_force, max_force, max_torque, max_torque, max_torque]))
        # print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 5.0
        self.max_error_position = 8.0
        self.target_position = np.array([0.0, 0.0, 10.0])  # ideally hovers at 10 units
        self.position_weight = 0.6
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.orientation_weight = 0.0
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # ideally zero velocity
        self.velocity_weight = 0.4
        self.enter_time = None
        self.land_time = None


    def reset(self):
        # Nothing to reset; just return initial condition
        self.last_timestamp = None
        self.last_position = None

        return Pose(
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def land_reset(self):

        self.max_duration = 15.0  # secs
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.position_weight = 0.7
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.orientation_weight = 0.0
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # ideally zero velocity
        self.velocity_weight = 0.3

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        # Calculate velocity
        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03)

        # Create state space and update lag variables
        state = np.concatenate([position, orientation, velocity])
        self.last_timestamp = timestamp
        self.last_position = position

        # Compute reward / penalty and check if this episode is complete
        done = False
        error_position = np.linalg.norm(self.target_position - state[0:3])
        error_orientation = np.linalg.norm(self.target_orientation - state[3:7])
        error_velocity = np.linalg.norm(self.target_velocity - state[7:10])
        reward = - (self.position_weight * error_position +
                    self.orientation_weight * error_orientation +
                    self.velocity_weight * error_velocity)

        #  reward of takeoff
        self.enter_time = np.copy(timestamp)
        takeoff_time = abs(timestamp - self.enter_time)
        if position[2] >= self.target_position[2]:
            reward += 1000.
            done = False
            #  reward of hover
            self.enter_time = np.copy(timestamp)
            hover_time = abs(timestamp - self.enter_time)
            if error_position > self.max_error_position:
                reward -= 300.0
                done = True
            if hover_time >= 7:
                reward += 2000.0
                self.land_time = np.copy(timestamp)
                landing = abs(self.land_time - timestamp)
                # reward of landing
                self.land_reset()
                if position[2] == 0:
                    reward += 5000.0
                    done = True
                elif position[2] >= 12.5:
                    reward -= 100.0
                    done = True
                elif abs(velocity[2]) >= 10.0 and position[2] <= 5.0:
                    reward -= 100.0
                    done = True
                elif abs(velocity[2]) >= 5.0 and position[2] <= 2.0:
                    reward -= 100.0
                    done = True
                if landing > 7:
                    reward -= 100.
                    done = True
        elif takeoff_time > 7 or timestamp > 15:
            reward -= 500.0
            done = True
        else:
            reward -= 150.
            done = False

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low,
                             self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                force=Vector3(action[0], action[1], action[2]),
                torque=Vector3(action[3], action[4], action[5])
            ), done
        else:
            return Wrench(), done

