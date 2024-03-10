import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))


from geometry import (
    Polygon,
    PolygonContainer,
    Point,
    polygon_intersect_polygon,
    point_in_polygon,
)

from typing import List
from config import GridMapConfig
import utils

from se2state import (
    SE2,
    SE2State,
    generate_vehicle_vertexes,
    generate_wheels_vertexes,
)


class GridMap:  # Obstacle Manager..
    def __init__(
        self, min_x=0, min_y=0, max_x=24, max_y=24, gridmap_cfg=GridMapConfig()
    ):
        self.grid_cfg = gridmap_cfg
        self.bound = gridmap_cfg.bound

        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.world_width = max_x - min_x
        self.world_height = max_y - min_y

        self.map_w = int(self.world_width / gridmap_cfg.xy_resolution)
        self.map_h = int(self.world_height / gridmap_cfg.xy_resolution)
        self.headings = int(math.pi * 2 / gridmap_cfg.heading_resolution)

        self.obstacles = PolygonContainer()

    def add_polygon_obstacle(self, obstacle_polygon: Polygon):
        max_x, max_y = np.max(obstacle_polygon.ndarray, axis=1)
        min_x, min_y = np.min(obstacle_polygon.ndarray, axis=1)

        expansion_flag = True
        if min_x < self.min_x:
            self.min_x = min_x
            expansion_flag = True

        if min_y < self.min_y:
            self.min_y = min_y
            expansion_flag = True

        if max_x > self.max_x:
            self.max_x = max_x
            expansion_flag = True

        if max_y > self.max_y:
            self.max_y = max_y
            expansion_flag = True

        if expansion_flag:
            self.world_width = self.max_x - self.min_x
            self.world_height = self.max_y - self.min_y

            self.map_w = int(self.world_width / self.grid_cfg.xy_resolution)
            self.map_h = int(self.world_height / self.grid_cfg.xy_resolution)

        self.obstacles += obstacle_polygon

    def add_polygon_obstacle_list(self, obstacle_polygon_list: List[Polygon]):
        for polygon in obstacle_polygon_list:
            self.add_polygon_obstacle(polygon)

    def add_vertexes_obstacle_list(self, obstacle_vertexes_list: List[List[Point]]):
        for v_obstacles in obstacle_vertexes_list:
            self.add_vertexes_obstacle(v_obstacles)

    def add_vertexes_obstacle(self, obstacle_vertexes: List[Point]):
        self.obstacles += Polygon(obstacle_vertexes)


def generate_gridmap_from_polygon(
    obstacle_polygon_list: List[Polygon],
    parking_polygon_list: List[Polygon],
    current_se2state: SE2State,
):
    if len(parking_polygon_list) == 0:
        return None

    """
    ### generate grid map from polygon.
    """

    point_varray = np.array(generate_vehicle_vertexes(current_se2state))

    for obstacle in obstacle_polygon_list:
        point_varray = np.vstack([point_varray, np.array(obstacle.vertexes)])

    for parking in parking_polygon_list:
        point_varray = np.vstack([point_varray, np.array(parking.vertexes)])

    min_x, min_y = np.min(point_varray, axis=0) - 0.2
    max_x, max_y = np.max(point_varray, axis=0) + 0.2

    grid = GridMap(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)

    grid.add_polygon_obstacle_list(obstacle_polygon_list)

    return grid


def outboundary(grid: GridMap, state: SE2State):
    if (
        state.x_index >= grid.map_w
        or state.x_index < 0
        or state.y_index >= grid.map_h
        or state.y_index < 0
        or state.heading_index >= grid.headings
        or state.heading_index < 0
    ):
        return True
    return False


def collision(grid: GridMap, state: SE2State):
    if outboundary(grid, state):
        return True

    vehicle_vertices = generate_vehicle_vertexes(state)
    vehicle_array = np.array(vehicle_vertices).T
    maxx = np.max(vehicle_array[0, :])
    maxy = np.max(vehicle_array[1, :])
    minx = np.min(vehicle_array[0, :])
    miny = np.min(vehicle_array[1, :])

    if (
        maxx >= grid.max_x
        or maxy >= grid.max_y
        or minx < grid.min_x
        or miny < grid.min_y
    ):
        return True

    if grid.obstacles is not None:
        vehicle_polygon = Polygon(vehicle_vertices)
        for obstacle_polygon in grid.obstacles:
            if polygon_intersect_polygon(obstacle_polygon, vehicle_polygon):
                return True

    return False


def trajectory_collision(grid: GridMap, trajectory: List[SE2State]):
    for state in trajectory:
        if collision(grid, state):
            return True

    return False


def generate_visited_map_3d(grid: GridMap) -> List[List[List[bool]]]:
    visited_map = [
        [[False for _ in range(0, grid.headings)] for _ in range(0, grid.map_h)]
        for _ in range(grid.map_w)
    ]

    return visited_map


def generate_visited_map_2d(grid: GridMap) -> List[List[bool]]:
    visited_map = [[False for _ in range(0, grid.map_h)] for _ in range(0, grid.map_w)]
    return visited_map


def generate_heuristic_map(grid: GridMap) -> List[List[float]]:
    heuristic_map = [
        [200.0 for _ in range(0, grid.map_h)] for _ in range(0, grid.map_w)
    ]
    return heuristic_map


def generate_obstacle_grid(
    grid: GridMap, girdmap_cfg=GridMapConfig()
) -> List[List[bool]]:
    obstacle_field_map = generate_visited_map_2d(grid)
    obstacle_field_map = np.array(obstacle_field_map)
    obstacle_field_map[0, :] = False
    obstacle_field_map[:, 0] = False
    obstacle_field_map[-1, :] = False
    obstacle_field_map[:, -1] = False

    if grid.obstacles is None:
        return obstacle_field_map

    for x in range(grid.map_w):
        xx = x * girdmap_cfg.xy_resolution + grid.min_x
        for y in range(grid.map_h):
            yy = y * girdmap_cfg.xy_resolution + grid.min_y
            for obstacle_polygon in grid.obstacles:
                if point_in_polygon(obstacle_polygon, (xx, yy)):
                    obstacle_field_map[x][y] = True

    return obstacle_field_map


def calc_likelihood_field(grid: GridMap):
    pmap = np.array(
        [
            [200 for _ in range(grid.map_h + grid.bound)]
            for _ in range(grid.map_w + grid.bound)
        ]
    )

    return pmap


if __name__ == "__main__":
    from search import breadth_first_search

    grid_map = GridMap()
    start = np.array([2, 2, 0])
    goal = np.array([10, 10, 0])
    start_se2 = SE2State(start[0], start[1], start[2])
    goal_se2 = SE2State(goal[0], goal[1], goal[2])

    heatmap = breadth_first_search(goal_se2, start_se2, grid_map)
    heatmap = np.array(heatmap) * 5
    utils.plot_heatmap(heatmap)
    plt.show()
