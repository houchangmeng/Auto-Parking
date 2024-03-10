import numpy as np
import matplotlib.pyplot as plt
import utils
import pickle

from search import (
    hybrid_a_star_search,
    breadth_first_search,
    bidirection_hybrid_a_star_search,
    multiprocess_bidirection_hybrid_a_star_search,
    upsample_smooth,
)
from gridmap import GridMap, generate_gridmap_from_polygon
from settings import (
    generate_obstacle_and_parking_vertexes,
    generate_obstacle_and_parking_polygon,
    generate_random_start_state,
    generate_goal_states_from_parking_rectpolygon,
    sort_goal_states,
)
from se2state import SE2State
from geometry import Polygon


def main():
    (
        obstacle_polygon_list,
        parking_polygon_list,
    ) = generate_obstacle_and_parking_polygon()

    """
    Select easy task for test.
    """
    TASK_NUM = 0
    """
    Task 0(Parallel parking).Optimial cost best.
    """
    start_state = SE2State(4, 15, -3.12)
    # start_state = SE2State(4, 15, 0)

    goal_state_list = generate_goal_states_from_parking_rectpolygon(
        parking_polygon_list[0]
    )

    goal_state_list = sort_goal_states(goal_state_list, start_state)
    goal_state = goal_state_list[0]

    """
    Task 1(T shape parking). Euclidian_cost best.
    """
    # start_state = SE2State(6.1, 14.03, 2.76)
    # start_state = SE2State(15.90, 14.03, 0.38)

    # goal_state_list = generate_goal_states_from_parking_rectpolygon(
    #     parking_polygon_list[1]
    # )

    # goal_state_list = sort_goal_states(goal_state_list, start_state)
    # goal_state = goal_state_list[1]

    """
    Plot task.
    """
    plt.figure(0, figsize=[8, 8])
    utils.plot_task(obstacle_polygon_list, start_state, goal_state)

    """
    Gridmap
    """

    gridmap = generate_gridmap_from_polygon(
        obstacle_polygon_list, parking_polygon_list, start_state
    )

    """
    Search.
    """

    # path = hybrid_a_star_search(start_state, goal_state, gridmap)
    # path = hybrid_a_star_search(goal_state, start_state, gridmap)
    path = bidirection_hybrid_a_star_search(start_state, goal_state, gridmap)

    # path = multiprocess_bidirection_hybrid_a_star_search(
    #     start_state, goal_state, gridmap
    # )
    print(f"origin path length{len(path):5d}")

    interval_interpolate_num = 3
    smoothpath = upsample_smooth(path, interval_interpolate_num)
    print(f"smooth path length{len(smoothpath):5d}")

    utils.plot_path(smoothpath)
    utils.plot_trajectory_animation(smoothpath)
    plt.show()

    print("\033[32m==Finished frontend test.===\033[0m")

    file_name = "se2path" + str(TASK_NUM) + ".pickle"

    with open(file_name, "wb") as f:
        pickle.dump(path, f)

    print("\033[32m==Save Result.===\033[0m")


def test_smooth():
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

    TASK_NUM = 1
    file_name = "se2opti_path" + str(TASK_NUM) + ".pickle"

    with open(file_name, "rb") as f:
        opti_path = pickle.load(f)

    start_state = opti_path[0]
    goal_state = opti_path[-1]

    utils.plot_task(obstacle_vertexes_list, start_state, goal_state)
    utils.plot_path(opti_path, "opti_path")

    ds_path = downsample_smooth(opti_path, 3)
    utils.plot_path(ds_path, "ds_path")

    us_path = upsample_smooth(opti_path, 3)
    utils.plot_path(us_path, "us_path")

    plt.draw()
    plt.pause(0.1)

    plt.figure(1, figsize=[8, 8])
    utils.plot_control(opti_path)
    utils.plot_control(ds_path)
    utils.plot_control(us_path)
    plt.show()

    print("123")


if __name__ == "__main__":
    main()
    # test_smooth()
