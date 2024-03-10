"""
This file was copied from openai gym for parking simulation, 
continuous action space.

"""

from typing import Optional, Union, List

import numpy as np

import gym
from gym import logger, spaces


from config import VehicleConfig

from se2state import SE2, SE2State

from settings import generate_obstacle_and_parking_polygon
from coordinate_transform import (
    change_polygon_list_coord,
    change_se2_coord,
    move_polygon,
    move_polygon_list,
)

import utils
import matplotlib.pyplot as plt


def kinematics(state: np.ndarray, action: np.ndarray, vehicle_cfg=VehicleConfig()):
    x, y, heading, v = state
    accel, delta = action

    beta = np.arctan(vehicle_cfg.lf * delta / vehicle_cfg.wheel_base)
    xdot = v * np.cos(heading + beta)
    ydot = v * np.sin(heading + beta)

    headingdot = v * np.cos(beta) / vehicle_cfg.wheel_base * np.tan(delta)

    return np.array([xdot, ydot, headingdot, accel])


def euler_step(
    state: np.ndarray, state_dot: np.ndarray, h: float, vehicle_cfg=VehicleConfig()
):
    x, y, heading, v = state
    current_se2 = SE2(x, y, heading)

    xdot, ydot, headingdot, accel = state_dot
    next_se2 = current_se2 + SE2(h * xdot, h * ydot, h * headingdot)

    x, y, heading = next_se2.x, next_se2.y, next_se2.so2.heading

    v = v + h * accel

    v = np.clip(v, -vehicle_cfg.max_v, vehicle_cfg.max_v)

    return x, y, heading, v


def euler_integration(
    state: np.ndarray, action: np.ndarray, h: float, vehicle_cfg=VehicleConfig()
):
    state_dot = kinematics(state, action, vehicle_cfg)
    return euler_step(state, state_dot, h, vehicle_cfg)


def runge_kutta_integration(
    state: np.ndarray, action: np.ndarray, h: float, vehicle_cfg=VehicleConfig()
):
    state = np.array(state, dtype=np.float32)

    f1 = kinematics(state, action)

    new_state = euler_step(state, f1, 0.5 * h)
    f2 = kinematics(new_state, action)

    new_state = euler_step(state, f2, 0.5 * h)
    f3 = kinematics(new_state, action)

    new_state = euler_step(state, f3, h)
    f4 = kinematics(new_state, action)

    state_dot = f1 + 2 * f2 + 2 * f3 + f4
    state = euler_step(state, state_dot, 1.0 / 6 * h)

    x, y, heading, v = state

    v = np.clip(v, -vehicle_cfg.max_v, vehicle_cfg.max_v)

    return x, y, heading, v


class ParkingEnvironment(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description

    This file was copied from openai gym for parking simulation. There are a car
    and a parking space in this environment. The goal is finding a best trajectory
    to drive to parking space.

    ### Action Space

    The action is a `ndarray` with shape `(2,)`  indicating the acceleration and
    steer angle rate of the car.

    | Num | Action                       |
    |-----|------------------------------|
    | 0   | acceleration                 |
    | 1   | front wheel steer angle      |


    ### State Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to
    the following positions, velocities and angle:

    | Num | State                 | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Car Position x        | -25.0               | 25.0              |
    | 1   | Car Position y        | -25.0               | 25.0              |
    | 2   | Car Theta             | -pi                 | pi                |
    | 3   | Car Velocity          | -2.0                | 2.0               |

    """

    def __init__(self, render_mode: Optional[str] = None):
        """
        Kinematics parameters.
        """

        self.tau = 0.02  # seconds between state updates
        self.horizon = 50000

        self.render_mode = render_mode
        self.figure = plt.figure(99, figsize=[16, 8])
        self.bev_axes = plt.subplot(121)
        self.bev_axes.set_xlim(-8, 8)
        self.bev_axes.set_ylim(-8, 8)
        self.global_axes = plt.subplot(122)
        self.image_list = []

        self.x_threshold = 99999
        self.y_threshold = 99999
        self.theta_threshold = np.pi
        self.vel_threshold = 3  # [m/s]
        self.delta_threshold = 0.46  # from tesla model3

        self.step_count = 0
        self.accel_threshold = 500.0  # [m/s^2]
        self.steer_rate_threshold = 500.0  # [rad / s]

        state_high = np.array(
            [
                self.x_threshold,
                self.y_threshold,
                np.pi,
                self.vel_threshold,
                self.delta_threshold,
            ],
            dtype=np.float32,
        )
        action_high = np.array(
            [
                self.accel_threshold,  # x
                self.steer_rate_threshold,
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(-state_high, state_high, dtype=np.float32)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

        self.isopen = True
        self.state = None
        self.se2state = None
        self.goal_se2state = None
        self.local_parking_polygon_list = None
        self.local_obstacle_polygon_list = None

        self.reference_trajectory = None
        self.tracking_trajectory = None
        self.steps_beyond_terminated = None

    def step(self, action: np.ndarray = np.array([0, 0], dtype=np.float32)):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        assert (
            self.se2state is not None
        ), "Call set_init_se2state before using step method."

        self.step_count += 1
        current_t = self.se2state.t + self.tau

        x, y, heading, v = runge_kutta_integration(
            self.se2state.array_state[:4], action, self.tau
        )

        self.state = (x, y, heading, v)

        self.se2state = SE2State(x, y, heading)
        self.se2state.v = v
        self.se2state.a = action[0]
        self.se2state.delta = action[1]
        self.se2state.t = current_t

        terminated = False

        if self.step_count >= self.horizon:
            terminated = True

        if not terminated:
            reward = -1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if len(self.local_obstacle_polygon_list) > 0:
            self.perception()

        """
        Here start tracking.
        """
        if self.tracking_trajectory is not None:
            self.tracking_trajectory += [self.se2state]

        if self.goal_se2state is not None:
            done = bool((self.goal_se2state.se2 - self.se2state.se2).norm() <= 1e-2)
        else:
            done = False

        if self.render_mode is not None:
            self.render()

        return (
            self.se2state,
            reward,
            terminated,
            done,
            {
                "local_obstacle_polygon_list": self.local_obstacle_polygon_list,
                "local_parking_polygon_list": self.local_parking_polygon_list,
            },
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        """
        TODO generate obstacles from environment.
        """

        self.global_parking_polygon_list = []
        self.global_obstacle_polygon_list = []

        self.local_parking_polygon_list = []
        self.local_obstacle_polygon_list = []

        self.steps_beyond_terminated = None
        self.step_count = 0
        self.se2state = options["start_se2state"]

        if self.render_mode is not None:
            self.render()

        return (
            self.se2state,
            {
                "local_obstacle_polygon_list": self.local_obstacle_polygon_list,
                "local_parking_polygon_list": self.local_parking_polygon_list,
            },
        )

    def perception(self):
        (
            self.global_obstacle_polygon_list,
            self.global_parking_polygon_list,
        ) = generate_obstacle_and_parking_polygon()

        self.local_obstacle_polygon_list = change_polygon_list_coord(
            self.global_obstacle_polygon_list, self.se2state.se2
        )

        self.local_parking_polygon_list = change_polygon_list_coord(
            self.global_parking_polygon_list, self.se2state.se2
        )

    def set_goal_se2state(self, goal_se2state: SE2State):
        self.goal_se2state = goal_se2state

    def set_reference_trajectory(self, reference_trajectory: List[SE2State]):
        self.reference_trajectory = reference_trajectory
        self.tracking_trajectory = []

    def render(self):
        """
        Plot
        """

        if self.step_count % 1 == 0:
            if self.goal_se2state is None:
                goal_se2state = self.se2state
            else:
                goal_se2state = self.goal_se2state

            plt.sca(self.bev_axes)
            self.bev_axes.cla()

            perception_current_se2 = change_se2_coord(
                self.se2state.se2, self.se2state.se2
            )
            perception_current_se2state = SE2State.from_se2(perception_current_se2)
            perception_current_se2state.delta = self.se2state.delta

            perception_goal_se2 = change_se2_coord(goal_se2state.se2, self.se2state.se2)
            perception_goal_se2state = SE2State.from_se2(perception_goal_se2)

            utils.plot_task(
                self.local_obstacle_polygon_list,
                perception_current_se2state,
                perception_goal_se2state,
            )

            self.bev_axes.set_xlabel("BEV")
            self.bev_axes.set_xlim(-8, 8)
            self.bev_axes.set_ylim(-8, 8)

            plt.sca(self.global_axes)
            if self.reference_trajectory is None:
                plt.cla()
                utils.plot_task(
                    self.global_obstacle_polygon_list, self.se2state, goal_se2state
                )
            else:
                utils.plot_vehicle(self.se2state)

            self.global_axes.set_xlabel("GLOBAL")
            plt.pause(0.001)

        if self.render_mode == "rgb_array":
            self.figure.canvas.draw()
            ncols, nrows = self.figure.canvas.get_width_height()
            image = np.frombuffer(
                self.figure.canvas.tostring_rgb(), dtype=np.uint8
            ).reshape(nrows, ncols, 3)
            self.image_list += [image]
            return image

    def close(self):
        if len(self.image_list) > 5:
            import imageio
            import time
            import pickle

            file_name = time.asctime() + "image_list" + ".pickle"

            with open(file_name, "wb") as f:
                pickle.dump(self.image_list, f)

            imageio.mimsave(
                time.asctime() + ".gif", self.image_list, "GIF", duration=0.02
            )
            self.isopen = False


def test_env():
    import random
    import time

    start_se2state = SE2State(4, 13, 3.14)
    goal_se2state = SE2State(13.8, 19.0, 3.14)

    env = ParkingEnvironment(render_mode="human")
    s, _ = env.reset(options={"start_se2state": start_se2state})

    start_perception_index = random.randint(3, 8)
    index = 0

    while True:
        index += 1
        time.sleep(0.2)
        if index == start_perception_index:
            env.perception()
            env.set_goal_se2state(goal_se2state)

        a = env.action_space.sample()
        # a = np.array([-1, 0.8], dtype=np.float32)
        s, r, t, d, _ = env.step(a)
        # print("state action", s, a)
        print(f"current index {index:5d}")
        if t or d:
            env.reset()
            print(" terminated")
        env.render()


def extract_gif():
    import imageio
    import time
    import pickle

    file_name = "image_list.pickle"
    with open(file_name, "rb") as f:
        image_list = pickle.load(f)

    imageio.mimsave(time.asctime() + ".gif", image_list, "GIF", duration=0.02)


if __name__ == "__main__":
    test_env()
    # extract_gif()
