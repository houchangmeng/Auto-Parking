import numpy as np
import matplotlib.pyplot as plt
import utils

from geometry import Polygon, move_vertexes_array, ndarray_to_vertexlist
from config import ParkingConfig
from se2state import SE2State, SE2
from typing import List

from gridmap import GridMap, collision


def parkingse2_to_parking_vertexes(
    parkingse2: SE2, parking_cfg: ParkingConfig = ParkingConfig()
):
    parking_array = parking_cfg.parking_array
    rot_angle = parkingse2.so2.heading
    xy_offset = np.array([[parkingse2.x], [parkingse2.y]])
    parking_array = move_vertexes_array(parking_array, rot_angle, xy_offset)
    parking_vertexes = ndarray_to_vertexlist(parking_array)

    return parking_vertexes


def generate_obstacle_and_parking_polygon(parking_cfg=ParkingConfig()):
    """
    ### Generate obstacle and parking polygon.

    ---
    Return: ( parallel parking (vertexes), T parking task (vertexes) )
    """

    (
        obstacle_vertexes_list,
        parking_vertexes_list,
    ) = generate_obstacle_and_parking_vertexes()

    obstacle_polygon_list = []
    parking_polygon_list = []

    for obstacle_vertexes in obstacle_vertexes_list:
        obstacle_polygon_list += [Polygon(obstacle_vertexes)]

    for parking_vertexes in parking_vertexes_list:
        parking_polygon_list += [Polygon(parking_vertexes)]

    return obstacle_polygon_list, parking_polygon_list


def generate_obstacle_and_parking_vertexes(parking_cfg=ParkingConfig()):
    """
    ### Generate obstacle and parking vertexes.

    ---
    Return: ( parallel parking (vertexes), T parking task (vertexes) )
    """

    obstacle_vertexes_list = []
    parking_vertexes_list = []
    """
    Parking space param.
    """
    parking_width = parking_cfg.parking_width
    parking_length = parking_cfg.parking_length
    parking_array = parking_cfg.parking_array

    """
    Parallel Parking
    """
    x_start, y_start = 12, 19
    xy_offset = np.array([[x_start - parking_length], [y_start]])
    neighbor_array = move_vertexes_array(parking_array, 0, xy_offset)
    v_obstacles = ndarray_to_vertexlist(neighbor_array)
    obstacle_vertexes_list.append(v_obstacles)

    xy_offset = np.array([[x_start + parking_length], [y_start]])
    neighbor_array = move_vertexes_array(parking_array, np.pi, xy_offset)
    v_obstacles = ndarray_to_vertexlist(neighbor_array)
    obstacle_vertexes_list.append(v_obstacles)

    xy_offset = np.array([[x_start], [y_start]])
    neighbor_array = move_vertexes_array(parking_array, 0, xy_offset)
    goal_parking_vertexes = ndarray_to_vertexlist(neighbor_array)
    parking_vertexes_list.append(goal_parking_vertexes)

    """
    T Parking
    """
    x_start, y_start = 11, 6
    xy_offset = np.array([[x_start], [y_start]])
    for i in range(3):
        xy_offset = np.array([[x_start + (i + 1) * parking_width], [y_start]])
        neighbor_array = move_vertexes_array(parking_array, np.pi / 2, xy_offset)
        v_obstacles = ndarray_to_vertexlist(neighbor_array)
        obstacle_vertexes_list.append(v_obstacles)

        xy_offset = np.array([[x_start - (i + 1) * parking_width], [y_start]])
        neighbor_array = move_vertexes_array(parking_array, np.pi / 2, xy_offset)
        v_obstacles = ndarray_to_vertexlist(neighbor_array)

        obstacle_vertexes_list.append(v_obstacles)

    xy_offset = np.array([[x_start], [y_start]])
    neighbor_array = move_vertexes_array(parking_array, np.pi / 2, xy_offset)
    goal_parking_vertexes = ndarray_to_vertexlist(neighbor_array)
    parking_vertexes_list.append(goal_parking_vertexes)

    return obstacle_vertexes_list, parking_vertexes_list


def generate_goal_states_from_parking_rectpolygon(
    parking_polygon: Polygon,
    parking_cfg=ParkingConfig(),
) -> List[SE2State]:
    """
    input parking_space_vertexes only one rectangle polygon.
    TODO: Rect(Polygon): only four vertexes.
    """

    if len(parking_polygon.lines) != 4:
        raise ValueError("error, please give a parking rectangle.")

    parking_length = parking_cfg.parking_length
    candidate_headingvec_with_length = []
    for line in parking_polygon.lines:
        vec = np.array(line[1]) - np.array(line[0])
        vec_length = np.linalg.norm(vec)
        if vec_length >= parking_length - 1e-2:
            candidate_headingvec_with_length += [(vec, vec_length)]

    if len(candidate_headingvec_with_length) < 1:
        raise ValueError("error, the parking rectangle can't parking a vehicle.")

    vec = max(
        candidate_headingvec_with_length, key=lambda vec_with_length: vec_with_length[1]
    )[0]

    theta = np.arctan2(vec[1], vec[0])
    vertexes_array = np.array(parking_polygon.vertexes).T
    center_x, center_y = np.mean(vertexes_array, axis=1)

    mid_point = np.array([[center_x], [center_y]])

    goal_state_list = []
    """
    TODO Model3 L:977mm, here we set 1.2, can rewrite at VehicleConfig.
    """
    parking_offset1 = np.array([[parking_length / 2 - 1.2], [0]])
    init_position = move_vertexes_array(parking_offset1, theta, mid_point)
    goal_state_list += [
        SE2State(init_position[0, 0], init_position[1, 0], theta + np.pi)
    ]

    parking_offset2 = -np.array([[parking_length / 2 - 1.2], [0]])
    init_position = move_vertexes_array(parking_offset2, theta, mid_point)

    goal_state_list += [SE2State(init_position[0, 0], init_position[1, 0], theta)]

    return goal_state_list


def generate_random_start_state(gridmap: GridMap):
    while True:
        x = np.random.random() * gridmap.world_width
        y = np.random.random() * gridmap.world_height
        t = np.random.random() * np.pi
        start_state = SE2State(x, y, t)
        # vehicle_vertics = generate_vehicle_vertexes(start_state)
        # utils.plot_polygon_vertexes(vehicle_vertics, linetype="--r")
        # plt.draw()
        # plt.pause(0.1)
        if not collision(gridmap, start_state):
            return start_state


def sort_goal_states(goal_state_list: List[SE2State], start_state: SE2State):
    def greater(lhs: SE2State, rhs: SE2State, start: SE2State):
        # lhs_dse2 = lhs.se2 - start.se2
        # rhs_dse2 = rhs.se2 - start.se2

        # lhs_ds2 = np.sqrt(lhs_dse2.x * lhs_dse2.x + lhs_dse2.y * lhs_dse2.y)
        # lhs_dh2 = abs(lhs_dse2.so2.heading)
        # lhs_dh2 = lhs_dh2 / np.pi * lhs_ds2 * 2
        # lhs_d = lhs_dh2 + lhs_ds2

        # rhs_ds2 = np.sqrt(rhs_dse2.x * rhs_dse2.x + rhs_dse2.y * rhs_dse2.y)
        # rhs_dh2 = abs(rhs_dse2.so2.heading)
        # rhs_dh2 = rhs_dh2 / np.pi * rhs_ds2 * 2
        # rhs_d = rhs_dh2 + rhs_ds2

        # return lhs_d > rhs_d

        car_in_lhs = abs((lhs.so2 - start.so2).heading)  # start in parking
        car_in_rhs = abs((rhs.so2 - start.so2).heading)

        return car_in_lhs > car_in_rhs

    # sort.
    n = len(goal_state_list)
    for i in range(n):
        for j in range(n - 1 - i):
            if greater(goal_state_list[j], goal_state_list[j + 1], start_state):
                temp = goal_state_list[j]
                goal_state_list[j] = goal_state_list[j + 1]
                goal_state_list[j + 1] = temp

    return goal_state_list


"""
this two function should move to controller function.
"""


def get_minimum_distance_states(
    se2state_list: List[SE2State], current_state: SE2State, look_ahead_point: int = 4
) -> SE2State:
    def less(lhs: SE2State, rhs: SE2State, current: SE2State):
        lhs_dist = np.linalg.norm(lhs.array_state[:2] - current.array_state[:2])
        rhs_dist = np.linalg.norm(rhs.array_state[:2] - current.array_state[:2])
        return abs(lhs_dist) < abs(rhs_dist)

    # sort.
    n = len(se2state_list)

    min_index = 0
    for i in range(n - look_ahead_point):
        if less(se2state_list[i], se2state_list[min_index], current_state):
            min_index = i

    return se2state_list[min_index + look_ahead_point]


def get_closest_time_states(
    se2state_list: List[SE2State], current_state: SE2State, look_ahead_point: int = 4
) -> SE2State:
    def less(lhs: SE2State, rhs: SE2State, current: SE2State):
        lhs_time = lhs.t - current.t
        rhs_time = rhs.t - current.t
        return abs(lhs_time) < abs(rhs_time)

    # sort.
    n = len(se2state_list)

    min_index = 0
    for i in range(n - look_ahead_point):
        if less(se2state_list[i], se2state_list[min_index], current_state):
            min_index = i

    return se2state_list[min_index + look_ahead_point]


# def generate_random_obstacle_vertexes(grid_map: GridMap, obstacle_num:int=5):
#     import random

#     obstacle_vertexes_list = []
#     while len(obstacle_vertexes_list) < obstacle_num:
#         obstacle_vertexes = [random.randint(0, grid_map.world_width), \
#                              random.randint(0, grid_map.world_height) \
#                             for _ in range(random.randint(3, 5))]

#         obstacle_vertexes_list += [obstacle_vertexes]

#     return obstacle_vertexes_list


def test():
    (
        obstacle_vertexes_list,
        parking_vertexes_list,
    ) = generate_obstacle_and_parking_vertexes()

    for v_obstacles in obstacle_vertexes_list:
        utils.plot_polygon_vertexes(v_obstacles)
    plt.axis("equal")
    plt.draw()
    plt.pause(0.1)

    gridmap = GridMap()
    gridmap.add_vertexes_obstacle_list(obstacle_vertexes_list)
    start_state = generate_random_start_state(gridmap)
    utils.plot_vehicle(start_state)
    plt.text(
        start_state.x,
        start_state.y,
        f" {start_state.heading:5.2f}",
        color="b",
    )
    plt.draw()
    plt.pause(0.5)
    for vertex_goal in parking_vertexes_list:
        utils.plot_polygon_vertexes(vertex_goal, linetype="--y")
        plt.draw()
        plt.pause(0.2)

        goal_state_list = generate_goal_states_from_parking_rectpolygon(
            Polygon(vertex_goal)
        )
        n = len(goal_state_list)
        for i in range(n):
            utils.plot_vehicle(goal_state_list[i])
            plt.text(
                goal_state_list[i].x,
                goal_state_list[i].y,
                str(i) + f", {goal_state_list[i].heading:5.2f}",
                color="b",
            )
            plt.draw()
            plt.pause(0.5)

        goal_state_list = sort_goal_states(goal_state_list, start_state)
        n = len(goal_state_list)
        for i in range(n):
            utils.plot_vehicle(goal_state_list[i])
            plt.text(
                goal_state_list[i].x - 2,
                goal_state_list[i].y - 2,
                str(i) + f", {goal_state_list[i].heading:5.2f}",
                color="r",
                size=10,
            )
            plt.draw()
            plt.pause(0.5)

    plt.show()


pass


if __name__ == "__main__":
    test()
