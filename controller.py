import numpy as np
from se2state import SE2State
from config import VehicleConfig, ControlConfig
from parking_env import runge_kutta_integration
from jax import jit, jacobian
import jax.numpy as jnp
from functools import partial
from settings import get_closest_time_states
from copy import deepcopy


def dlqr(A, B, Q, R, eps=1e-3):
    P = Q

    K_last = np.ones((1, 4)) * 1e4

    while True:
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        P = Q + K.T @ R @ K + (A - B @ K).T @ P @ (A - B @ K)
        if np.linalg.norm(K - K_last, np.inf) < eps:
            break
        K_last = K

    return K


class LatLonController:
    def __init__(self, veh_cfg=VehicleConfig(), ctrl_cfg=ControlConfig()) -> None:
        self.ctrl_cfg = ctrl_cfg
        self.veh_cfg = veh_cfg
        self.dt = ctrl_cfg.dt
        self.Q = np.diag([10, 10, 0.1, 0.1])
        self.R = np.diag([0.1, 0.1])

        self.dfdx = jacobian(self.rk2, 0)
        self.dfdu = jacobian(self.rk2, 1)

        self.T = 0.8
        self.last_action = np.array([0, 0], dtype=np.float32)

    def action(self, current_se2state: SE2State, target_se2state: SE2State):
        ref_x = target_se2state.x
        ref_y = target_se2state.y
        ref_h = target_se2state.heading
        ref_v = target_se2state.v

        ref_acc = target_se2state.a
        ref_delta = target_se2state.delta

        ref_state = jnp.array([ref_x, ref_y, ref_h, ref_v])
        ref_action = jnp.array([ref_acc, ref_delta])

        A, B = self.get_AB_matrix(ref_state, ref_action)

        K = dlqr(A, B, self.Q, self.R)
        e = self.error_state(current_se2state, target_se2state)

        ff = np.array([target_se2state.a, target_se2state.delta])
        action = -K @ e + ff

        # acc = action[0]
        acc = np.clip(
            action[0],
            -self.veh_cfg.max_acc,
            self.veh_cfg.max_acc,
        )
        # delta = action[1]
        delta = np.clip(
            action[1],
            -self.veh_cfg.max_front_wheel_angle,
            self.veh_cfg.max_front_wheel_angle,
        )

        action = np.array([acc, delta], dtype=np.float32)
        action = self.T * action + (1 - self.T) * self.last_action
        self.last_action = action

        return action

    @partial(jit, static_argnums=(0,))
    def kinematics(self, state, action):
        x, y, heading, v = state
        accel, delta = action

        beta = jnp.arctan(self.veh_cfg.lf * delta / self.veh_cfg.wheel_base)
        xdot = v * jnp.cos(heading + beta)
        ydot = v * jnp.sin(heading + beta)

        headingdot = v * jnp.cos(beta) / self.veh_cfg.wheel_base * jnp.tan(delta)

        return jnp.array([xdot, ydot, headingdot, accel])

    @partial(jit, static_argnums=(0,))
    def rk2(self, state, action):
        f1 = self.kinematics(state, action)
        f2 = self.kinematics(state + self.dt * f1, action)

        return state + (self.dt / 2.0) * (f1 + f2)

    @partial(jit, static_argnums=(0,))
    def get_AB_matrix(self, state, action):
        A = self.dfdx(state, action)
        B = self.dfdu(state, action)

        return A, B

    def error_state(self, current_se2state: SE2State, target_se2state: SE2State):
        ese2 = target_se2state.se2 - current_se2state.se2
        ex = ese2.x
        ey = ese2.y
        eheading = ese2.so2.heading
        ev = current_se2state.v - target_se2state.v

        return np.array([ex, ey, eheading, ev], dtype=np.float32)


class Controller:
    def __init__(self) -> None:
        self.lon_controller = LonController()
        self.lat_controller = LatController()

    def action(self, current_se2state: SE2State, target_se2state: SE2State):
        accel = self.lon_controller.action(current_se2state, target_se2state)
        delta = self.lat_controller.action(current_se2state, target_se2state)

        return np.array([accel, delta], dtype=np.float32)

    def emergency_stop(self, current_se2state: SE2State):
        target_se2state = deepcopy(current_se2state)
        target_se2state.a = 0
        target_se2state.v = 0
        target_se2state.heading = 0
        accel = self.lon_controller.action(current_se2state, target_se2state)
        delta = self.lat_controller.action(current_se2state, target_se2state)
        return np.array([accel, delta])


class LatController:
    def __init__(self, veh_cfg=VehicleConfig(), ctrl_cfg=ControlConfig()) -> None:
        self.dt = ctrl_cfg.dt
        self.veh_cfg = veh_cfg
        self.ctrl_cfg = ctrl_cfg
        self.Q = ctrl_cfg.Q
        self.R = ctrl_cfg.R

        self.last_delta = 0
        self.T = 0.8  # 1 / T filter.
        self.K_save = []
        self.v_save = []

    def action(self, current_se2state: SE2State, target_se2state: SE2State):
        v = current_se2state.v
        delta = current_se2state.delta
        L = self.veh_cfg.wheel_base
        A = np.array(
            [[1, self.dt, 0, 0], [0, 0, v, 0], [0, 0, 1, self.dt], [0, 0, 0, 0]]
        )
        B = np.array([[0], [0], [v / L], [0]])

        K = dlqr(A, B, self.Q, self.R)
        e = self.error_state(current_se2state, target_se2state)

        delta = -K @ e + target_se2state.delta
        delta = np.clip(
            delta,
            -self.veh_cfg.max_front_wheel_angle,
            self.veh_cfg.max_front_wheel_angle,
        )

        delta = self.T * delta + (1 - self.T) * self.last_delta
        self.last_delta = delta
        return float(delta)

    def error_state(self, current_se2state: SE2State, target_se2state: SE2State):
        ese2 = target_se2state.se2 - current_se2state.se2

        d_err = np.array([ese2.x, ese2.y])
        nor = np.array(
            [-np.sin(target_se2state.heading), np.cos(target_se2state.heading)]
        )

        eheading = ese2.so2.heading
        v = current_se2state.v
        delta = current_se2state.delta

        d = d_err @ nor.T
        d_dot = v * np.sin(eheading)

        L = self.veh_cfg.wheel_base
        v_ref = target_se2state.v
        delta_ref = target_se2state.delta

        eheading_dot = -v_ref * np.tan(delta_ref) / L + v * np.tan(delta) / L

        return np.array([d, d_dot, eheading, eheading_dot])


class LonController:
    def __init__(self, veh_cfg=VehicleConfig(), ctrl_cfg=ControlConfig()) -> None:
        self.ctrl_cfg = ctrl_cfg
        self.veh_cfg = veh_cfg
        self.dt = ctrl_cfg.dt

        self.s_p = ctrl_cfg.s_p
        self.s_i = ctrl_cfg.s_i
        self.s_d = ctrl_cfg.s_d

        self.v_p = ctrl_cfg.v_p
        self.v_i = ctrl_cfg.v_i
        self.v_d = ctrl_cfg.v_d

        self.es_k1 = 0
        self.es_k2 = 0

        self.ev_k1 = 0
        self.ev_k2 = 0

        self.uv_k = 0
        self.ua_k = 0

        self.last_acc = 0
        self.T = 0.8

    def action(self, current_se2state: SE2State, target_se2state: SE2State):
        s_err = np.array(
            [
                target_se2state.x - current_se2state.x,
                target_se2state.y - current_se2state.y,
            ]
        )

        tor = np.array(
            [np.cos(target_se2state.heading), np.sin(target_se2state.heading)]
        )

        es_k = s_err @ tor.T

        self.uv_k += (
            self.s_p * (es_k - self.es_k1)
            + self.s_i * es_k
            + self.s_d * (es_k - 2 * self.es_k1 + self.es_k2)
        ) * self.dt

        self.es_k2 = self.es_k1
        self.es_k1 = es_k

        ev_k = target_se2state.v - current_se2state.v + self.uv_k

        self.ua_k += (
            self.v_p * (ev_k - self.ev_k1)
            + self.v_i * ev_k
            + self.v_d * (ev_k - 2 * self.ev_k1 + self.ev_k2)
        ) * self.dt

        self.ev_k2 = self.ev_k1
        self.ev_k1 = ev_k

        acc = np.clip(
            self.ua_k + target_se2state.a, -self.veh_cfg.max_acc, self.veh_cfg.max_acc
        )
        acc = self.T * acc + (1 - self.T) * self.last_acc

        self.last_acc = acc

        return acc

        # return float(self.ua_k)


def simulation_kinematics(se2state: SE2State, action, h=0.02):
    state = se2state.array_state[:4]

    action[0] += np.random.normal(0, 0.01)
    action[1] += np.random.normal(0, 0.01)

    x, y, heading, v = runge_kutta_integration(state, action, h)

    x += np.random.normal(0, 0.01)
    y += np.random.normal(0, 0.01)
    heading += np.random.normal(0, 0.01)
    v += np.random.normal(0, 0.01)

    next_se2state = SE2State(x, y, heading)
    next_se2state.v = float(v)
    next_se2state.a = float(action[0])
    next_se2state.delta = float(action[1])
    next_se2state.t = float(se2state.t + h)

    return next_se2state


def test_controller():
    from settings import generate_obstacle_and_parking_vertexes
    import matplotlib.pyplot as plt

    import pickle
    import utils
    from search import upsample_smooth, downsample_smooth

    (
        obstacle_vertexes_list,
        _,
    ) = generate_obstacle_and_parking_vertexes()
    plt.figure(0, figsize=[8, 8])

    TASK_NUM = 0
    file_name = "se2opti_path" + str(TASK_NUM) + ".pickle"

    with open(file_name, "rb") as f:
        opti_path = pickle.load(f)

    start_state = opti_path[0]
    goal_state = opti_path[-1]
    utils.plot_task(obstacle_vertexes_list, start_state, goal_state)

    ctrl_cfg = ControlConfig()
    interval = int((opti_path[1].t - opti_path[0].t) / ctrl_cfg.dt) - 1

    us_path = upsample_smooth(opti_path, interval)
    utils.plot_path(us_path, "us_path")
    plt.draw()
    plt.pause(0.1)

    controller = Controller()
    # controller = LatLonController()
    current_se2state = start_state
    tracking_path = [current_se2state]

    N = len(us_path)
    # N = 200
    for i in range(N):
        # target_se2state = us_path[i]
        target_se2state = get_closest_time_states(us_path, current_se2state)
        u = controller.action(current_se2state, target_se2state)
        current_se2state = simulation_kinematics(current_se2state, u)
        tracking_path += [current_se2state]
        if i % 10 == 0:
            print(f"Current step is {i} / {N}")

    utils.plot_path(tracking_path, "tracking_path")
    ds_tracking_path = downsample_smooth(tracking_path, interval * 2)
    utils.plot_trajectory_animation(ds_tracking_path)

    plt.figure(1, figsize=[8, 10])
    utils.plot_control(us_path)
    utils.plot_control(tracking_path)
    plt.draw()
    plt.show()
    print("123")


if __name__ == "__main__":
    test_controller()
