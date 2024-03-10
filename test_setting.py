import numpy as np
import matplotlib.pyplot as plt
import utils
from settings import (
    generate_goal_states_from_parking_rectpolygon,
    generate_obstacle_and_parking_vertexes,
    generate_random_start_state,
    sort_goal_states,
)

from geometry import Polygon
from gridmap import GridMap


def main():
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


if __name__ == "__main__":
    main()
