import numpy as np
import matplotlib.pyplot as plt
import utils


import geometry
from geometry import Polygon, Point
from typing import List
from se2state import SE2, SE2State


def compute_planning_coord_se2_from_global_info(
    obstacle_polygon_list: List[Polygon],
    parking_polygon_list: List[Polygon],
    current_vehicle_se2: SE2State,
):
    """
    ### Compute planning coordinate se2 representation.
    """
    if len(parking_polygon_list) == 0:
        return None

    point_varray = np.array([current_vehicle_se2.x, current_vehicle_se2.y])
    for obstacle in obstacle_polygon_list:
        point_varray = np.vstack([point_varray, np.array(obstacle.vertexes)])

    for parking in parking_polygon_list:
        point_varray = np.vstack([point_varray, np.array(parking.vertexes)])

    planing_origin = np.min(point_varray, axis=0) - 2

    return SE2(planing_origin[0], planing_origin[1], 0.0)


def compute_planning_coord_se2_in_perception(
    obstacle_vertexes_list: List[List[Point]],
    parking_vertexes_list: List[List[Point]],
    current_vehicle_se2: SE2,
):
    """
    ### Compute planning coordinate se2 representation.
    """
    point_varray = np.array([current_vehicle_se2.x, current_vehicle_se2.y])
    for obstacle in obstacle_vertexes_list:
        point_varray = np.vstack([point_varray, np.array(obstacle)])

    for parking in parking_vertexes_list:
        point_varray = np.vstack([point_varray, np.array(parking)])

    planing_origin = np.min(point_varray, axis=0) - 2

    return SE2(planing_origin[0], planing_origin[1], 0.0)


def change_vertexes_array_coord(
    vertexes_array_in_current_coord: np.ndarray,
    target_coord_se2: SE2,
):
    """
    Same point, represent in different coord.
    """

    offset = np.array([[target_coord_se2.x], [target_coord_se2.y]])
    rot_angle = target_coord_se2.so2.heading

    return geometry.change_vertexes_array_coord(
        vertexes_array_in_current_coord, rot_angle, offset
    )


def change_vertexes_coord(
    vertexes_in_current_coord: np.ndarray,
    target_coord_se2: SE2,
):
    vertexes_array = change_vertexes_array_coord(
        np.array(vertexes_in_current_coord).T, target_coord_se2
    )
    return geometry.ndarray_to_vertexlist(vertexes_array)


def change_vertexes_list_coord(
    vertexes_list_in_current_coord: List[List[Point]],
    target_coord_se2: SE2,
):
    vertexes_list_in_target_coord = []
    for vertexes in vertexes_list_in_current_coord:
        vertexes_list_in_target_coord += [
            change_vertexes_coord(vertexes, target_coord_se2)
        ]

    return vertexes_list_in_target_coord


def change_polygon_coord(
    polygon_in_current_coord: Polygon,
    target_coord_se2: SE2,
) -> Polygon:
    array_in_target_coord = change_vertexes_array_coord(
        polygon_in_current_coord.ndarray, target_coord_se2
    )

    return Polygon(geometry.ndarray_to_vertexlist(array_in_target_coord))


def change_polygon_list_coord(
    polygon_list_in_current_coord: List[Polygon],
    target_coord_se2: SE2,
) -> Polygon:
    polygon_list_in_target_coord = []
    for polygon in polygon_list_in_current_coord:
        polygon_list_in_target_coord += [
            change_polygon_coord(polygon, target_coord_se2)
        ]

    return polygon_list_in_target_coord


def move_vertexes(vertexes_in_local_coord: np.ndarray, current_coord_se2: SE2):
    rot_angle = current_coord_se2.so2.heading
    offset = np.array([[current_coord_se2.x], [current_coord_se2.y]])

    vertexes_array_in_global = geometry.move_vertexes_array(
        np.array(vertexes_in_local_coord).T, rot_angle, offset
    )

    return geometry.ndarray_to_vertexlist(vertexes_array_in_global)


def move_polygon(polygon_in_local_coord: Polygon, current_coord_se2: SE2):
    rot_angle = current_coord_se2.so2.heading
    offset = np.array([[current_coord_se2.x], [current_coord_se2.y]])
    vertexes_array_in_global = geometry.move_vertexes_array(
        polygon_in_local_coord.ndarray, rot_angle, offset
    )

    return Polygon(geometry.ndarray_to_vertexlist(vertexes_array_in_global))


def move_polygon_list(polygon_list_in_local_coord: Polygon, current_coord_se2: SE2):
    polygon_list_in_global_coord = []
    for polygon_in_local in polygon_list_in_local_coord:
        polygon_list_in_global_coord += [
            move_polygon(polygon_in_local, current_coord_se2)
        ]

    return polygon_list_in_global_coord


def change_se2_coord(se2_in_current_coord: SE2, target_coord_se2: SE2):
    offset = np.array([[target_coord_se2.x], [target_coord_se2.y]])
    rot_angle = target_coord_se2.so2.heading

    """
    Same se2, represent in different coord.
    """
    rot = np.array(
        [
            [np.cos(rot_angle), np.sin(rot_angle)],
            [-np.sin(rot_angle), np.cos(rot_angle)],
        ]
    )

    current_xy = np.array([[se2_in_current_coord.x], [se2_in_current_coord.y]])

    target_xy = rot @ (current_xy - offset)

    current_heading = se2_in_current_coord.so2.heading
    current_heading_vec = np.array(
        [[np.cos(current_heading)], [np.sin(current_heading)]]
    )

    target_heading_vec = rot @ current_heading_vec
    target_heading = np.arctan2(target_heading_vec[1, 0], target_heading_vec[0, 0])

    return SE2(target_xy[0, 0], target_xy[1, 0], target_heading)


def change_se2list_coord(vehicle_se2list_in_current_coord: SE2, target_coord_se2: SE2):
    vehicle_se2list_in_target_coord = []
    for se2 in vehicle_se2list_in_current_coord:
        vehicle_se2list_in_target_coord += [change_se2_coord(se2, target_coord_se2)]
    return vehicle_se2list_in_target_coord


def test_coordinate_transform():
    from search import (
        upsample_smooth,
        bidirection_hybrid_a_star_search,
    )
    from settings import (
        generate_obstacle_and_parking_polygon,
        generate_goal_states_from_parking_rectpolygon,
        sort_goal_states,
    )

    from settings import generate_obstacle_and_parking_polygon
    from gridmap import generate_gridmap_from_polygon

    """
    Generate global polygon. perception information.
    """

    (
        obstacle_polygon_list,
        parking_polygon_list,
    ) = generate_obstacle_and_parking_polygon()

    """
    transform global to local.
    """
    start_state_global = SE2State(4, 15, -3.14)
    start_state_local = SE2State(0, 0, 0)

    perception_se2 = SE2(0, 0, 0)

    obstacle_polygon_list_local = change_polygon_list_coord(
        obstacle_polygon_list, start_state_global.se2
    )

    parking_polygon_list_local = change_polygon_list_coord(
        parking_polygon_list, start_state_global.se2
    )

    goal_state_list_local = generate_goal_states_from_parking_rectpolygon(
        parking_polygon_list_local[0]
    )

    goal_state_list = sort_goal_states(goal_state_list_local, start_state_local)
    goal_state_local = goal_state_list[0]

    """
    Plot local task.
    """
    plt.figure(0, figsize=[8, 8])
    utils.plot_task(obstacle_polygon_list_local, start_state_local, goal_state_local)
    plt.draw()

    """
    Local Grid.
    """

    gridmap_local = generate_gridmap_from_polygon(
        obstacle_polygon_list_local, parking_polygon_list_local, start_state_local
    )

    """
    Local Search.
    """

    local_path = bidirection_hybrid_a_star_search(
        start_state_local, goal_state_local, gridmap_local
    )

    utils.plot_path(local_path)
    utils.plot_trajectory_animation(local_path)
    plt.show()

    """
    Transform local to global
    """

    obstacle_polygon_list_global = move_polygon_list(
        obstacle_polygon_list_local, start_state_global.se2
    )

    parking_polygon_list_global = move_polygon_list(
        parking_polygon_list_local, start_state_global.se2
    )

    goal_state_list_global = generate_goal_states_from_parking_rectpolygon(
        parking_polygon_list_global[0]
    )

    goal_state_list_global = sort_goal_states(
        goal_state_list_global, start_state_global
    )
    goal_state_global = goal_state_list_global[0]

    """
    Plot global task.
    """
    plt.figure(1, figsize=[8, 8])
    utils.plot_task(obstacle_polygon_list_global, start_state_global, goal_state_global)
    plt.draw()
    plt.pause(0.1)

    """
    Global Grid.
    """

    gridmap_global = generate_gridmap_from_polygon(
        obstacle_polygon_list_global, parking_polygon_list_global, start_state_global
    )

    """
    Global Search.
    """

    global_path = bidirection_hybrid_a_star_search(
        start_state_global, goal_state_global, gridmap_global
    )

    utils.plot_path(global_path)
    utils.plot_trajectory_animation(global_path)
    plt.show()

    print("\033[32m==Finished coordinate transform test.===\033[0m")


if __name__ == "__main__":
    test_coordinate_transform()
