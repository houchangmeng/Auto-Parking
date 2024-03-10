import numpy as np


"""
TODO:   Different Parking task should have different Config.
"""


class SearchConfig:
    def __init__(self):
        self.max_iteration = 600000
        self.max_heading_index_error = 5
        self.heuristic_weight = 1.1
        self.penalty_change_gear = 1.0
        self.discrete_delta_num = 2
        self.discrete_velocity_num = 3

        self.visited_exam_interval = 300
        self.interval_interpolate_num = 2


class VehicleConfig:
    """
    ### Car Parameters

    From Tesla Model 3.
    """

    def __init__(self):
        self.length = 4.8
        self.width = 2.1
        self.baselink_to_rear = 1.0
        self.baselink_to_front = self.length - self.baselink_to_rear

        self.wheel_base = 2.9
        self.wheel_radius = 0.50
        self.wheel_width = 0.24
        self.wheel_distance = 1.59

        self.cf_alpha = 30000  # 侧偏刚度 f
        self.cr_alpha = 30000  # 侧偏刚度 r
        self.mass = 1752
        self.Iz = 1.0 / 12 * self.mass * (self.width**2 + self.length**2)

        self.lf = 1.8
        self.lr = self.wheel_base - self.lf  # 1.0
        self.max_front_wheel_angle = 0.46  # calc from min radius 11.7 [m]

        self.min_radius = self.wheel_base / np.tan(self.max_front_wheel_angle)

        self.T = 0.2  # discreate time
        self.max_acc = 20
        self.max_v = 2
        self.max_delta_dot = 20
        self.max_jerk = 100


class GridMapConfig:
    """
    Different Parking task should have different Config.
    """

    def __init__(self, vehicle_cfg=VehicleConfig(), search_cfg=SearchConfig()) -> None:
        self.world_width = 24

        self.world_height = 24

        self.bound = 5  # index

        min_vel = vehicle_cfg.max_v / search_cfg.discrete_velocity_num
        min_delta = vehicle_cfg.max_front_wheel_angle / search_cfg.discrete_delta_num

        self.heading_resolution = min_vel / vehicle_cfg.wheel_base * np.tan(min_delta)
        self.xy_resolution = (
            min_vel * np.cos(vehicle_cfg.max_front_wheel_angle) * vehicle_cfg.T
        )


class OptimizeConfig:

    """ """

    def __init__(self) -> None:
        self.Q = np.diag([1.0, 1.0, 1.0, 0, 0, 0])
        self.R = np.diag([1, 1])
        self.solver_opts = {
            "print_time": False,
            "verbose": False,
            "ipopt.print_level": 0,
        }


class ControlConfig:
    def __init__(self) -> None:
        self.dt = 0.02
        self.Q = np.diag([10, 10, 100, 100])
        self.R = np.diag([0.01])

        self.s_p = 100
        self.s_i = 0.01
        self.s_d = 0.01

        self.v_p = 100
        self.v_i = 0.01
        self.v_d = 0.00

        self.a_p = 1
        self.a_i = 0.0
        self.a_d = 0.0


class ParkingConfig:
    """
    Parking shape length 6, width 2.5
    """

    def __init__(self, parking_width=2.5, parking_length=6):
        self.parking_width = parking_width
        self.parking_length = parking_length
        self.parking_shape = [
            (-parking_length / 2, -parking_width / 2),
            (parking_length / 2, -parking_width / 2),
            (parking_length / 2, parking_width / 2),
            (-parking_length / 2, parking_width / 2),
        ]

        self.parking_array = np.array(self.parking_shape).T
