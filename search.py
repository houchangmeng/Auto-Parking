from typing import List, Tuple
import numpy as np
from queue import PriorityQueue, Queue

from se2state import SE2State, SE2
from config import VehicleConfig, SearchConfig, GridMapConfig
from gridmap import GridMap


def compute_2d_index(gridmap, se2state: SE2State):
    se2state.x_index = int(
        (se2state.x - gridmap.min_x) / gridmap.grid_cfg.xy_resolution
    )
    se2state.y_index = int(
        (se2state.y - gridmap.min_y) / gridmap.grid_cfg.xy_resolution
    )
    se2state.heading_index = int(
        (se2state.heading + np.pi) / gridmap.grid_cfg.heading_resolution
    )


def compute_3d_index(gridmap, se2state: SE2State):
    se2state.x_index = int(
        (se2state.x - gridmap.min_x) / gridmap.grid_cfg.xy_resolution
    )
    se2state.y_index = int(
        (se2state.y - gridmap.min_y) / gridmap.grid_cfg.xy_resolution
    )
    se2state.heading_index = int(
        (se2state.heading + np.pi) / gridmap.grid_cfg.heading_resolution
    )


def update_se2state(state: SE2State, vel, delta, gridmap, vehicle_cfg=VehicleConfig()):
    """
    x_k+1 = x_k + vel * np.cos(heading_k) * T
    y_k+1 = y_k + vel * np.sin(heading_k) * T
    heading_k+1 = heading_k + vel / SE2State.vehicle_cfg.wheel_base * np.tan(delta) * T
    """

    T = vehicle_cfg.T
    dx = vel * np.cos(state.heading) * T
    dy = vel * np.sin(state.heading) * T
    dheading = vel / vehicle_cfg.wheel_base * np.tan(delta) * T
    se2 = state.se2 + SE2(dx, dy, dheading)

    new_se2state = SE2State.from_se2(se2=se2)

    new_se2state.v = vel
    new_se2state.delta = delta

    compute_3d_index(gridmap, new_se2state)

    return new_se2state


def get_next_states(
    state: SE2State,
    grid: GridMap,
    vehicle_cfg=VehicleConfig(),
    search_cfg=SearchConfig(),
):
    """
    ### Generate next SE2.

    rl  4 \  / 1  fl
    r    5 -- 2   f
    rr  6 /  \ 3  fr
    """

    next_states = []
    delta_discrete_num = search_cfg.discrete_delta_num
    vel_discrete_num = search_cfg.discrete_delta_num
    velocity_reso = vehicle_cfg.max_v / delta_discrete_num
    delta_reso = vehicle_cfg.max_front_wheel_angle / vel_discrete_num

    for discre_delta in range(-delta_discrete_num, delta_discrete_num + 1):
        for discrete_vel in range(-vel_discrete_num, vel_discrete_num + 1):
            if discrete_vel == 0:
                continue
            velocity = velocity_reso * discrete_vel
            delta = delta_reso * discre_delta
            new_state = update_se2state(state, velocity, delta, gridmap=grid)

            # new_state.delta = delta
            # new_state.v = velocity

            new_state.direction_index = get_direction(velocity, delta)
            next_states += [new_state]

    return next_states


def get_direction(velocity, delta):
    """
    ### Check direction.
    rl  4 \  / 1  fl
    r    5 -- 2   f
    rr  6 /  \ 3  fr
    """
    if velocity > 0:
        if delta > 0:
            direction = 1
        elif delta == 0:
            direction = 2
        else:
            direction = 3
    elif velocity < 0:
        if delta > 0:
            direction = 4
        elif delta == 0:
            direction = 5
        else:
            direction = 6
    else:
        raise ValueError(" Error velocity")

    return direction


def get_next_grid_state(state: SE2State, gridmap: GridMap):
    """
    ### Generate next Grid SE2.
            le
            7
    rl  4 \ | / 1  fl
    r    5 --- 2   f
    rr  6 / | \ 3  fr
            8
            ri
    """
    next_states = []
    gridmap_cfg = gridmap.grid_cfg

    offset_list = [
        [gridmap_cfg.xy_resolution, gridmap_cfg.xy_resolution, np.pi / 4, 1.18],
        [gridmap_cfg.xy_resolution, 0, 0, 1],
        [gridmap_cfg.xy_resolution, -gridmap_cfg.xy_resolution, -np.pi / 4, 1.18],
        [
            -gridmap_cfg.xy_resolution,
            gridmap_cfg.xy_resolution,
            np.pi * 3 / 4,
            1.18,
        ],
        [-gridmap_cfg.xy_resolution, 0, np.pi, 1],
        [
            -gridmap_cfg.xy_resolution,
            -gridmap_cfg.xy_resolution,
            -np.pi * 3 / 4,
            1.18,
        ],
        [0, gridmap_cfg.xy_resolution, np.pi * 2 / 4, 1],
        [0, -gridmap_cfg.xy_resolution, -np.pi * 2 / 4, 1],
    ]

    for offset in offset_list:
        x = state.x + offset[0]
        y = state.y + offset[1]
        heading = state.heading + offset[2]
        new_state = SE2State(x, y, heading)
        compute_2d_index(gridmap, new_state)
        new_state.v = offset[3]  # for cost
        new_state.delta = 0.0  # for cost
        next_states += [new_state]

    return next_states


from copy import deepcopy


def downsample_smooth(
    path: List[SE2State],
    gap: int = 3,
):
    if not path:
        print("no path")
        return []

    ds_path = deepcopy(path[::gap])
    if len(ds_path) < 3:
        return ds_path

    for i in range(1, len(ds_path) - 1):
        v = 0
        d = 0
        for k in range(gap + 2 - 1):
            v += path[i * gap + k].v
            v += path[i * gap - k].v

            d += path[i * gap + k].delta
            d += path[i * gap - k].delta

        ds_path[i].v = v / gap / 2
        ds_path[i].delta = d / gap / 2

    ds_path[-1] = path[-1]
    return ds_path


def upsample_smooth(
    path: List[SE2State],
    interval: int = 3,
) -> List[SE2State]:
    from scipy import interpolate

    if len(path) < 5 or interval < 3:
        raise ValueError("check path or interval number.")

    T = path[1].t - path[0].t
    t = np.array([0, T])
    delta_t = T / (interval + 1)  # segement num
    t_seq = np.linspace(0, T, interval + 2)  # start + end, 2 waypoints

    us_path = []
    # path = path[::interval]
    for i in range(0, len(path) - 1):
        x = np.array([path[i].x, path[i + 1].x])
        y = np.array([path[i].y, path[i + 1].y])
        heading = np.array([path[i].heading, path[i + 1].heading])
        vel = np.array([path[i].v, path[i + 1].v])
        acc = np.array([path[i].a, path[i + 1].a])
        jerk = np.array([path[i].jerk, path[i + 1].jerk])
        delta = np.array([path[i].delta, path[i + 1].delta])
        delta_dot = np.array([path[i].delta_dot, path[i + 1].delta_dot])

        yy = np.vstack((x, y, heading, vel, acc, jerk, delta, delta_dot))
        interp_func = interpolate.interp1d(t, yy)
        state_sub_seq = interp_func(t_seq)

        for k in range(interval + 2 - 1):
            x, y, heading, v, a, j, delta, ddelta = state_sub_seq[:, k]
            new_state = SE2State(x, y, heading)
            new_state.t = i * T + k * delta_t
            new_state.v = v
            new_state.a = a
            new_state.jerk = j
            new_state.delta = delta
            new_state.delta_dot = ddelta

            us_path += [new_state]

    path[-1].t = us_path[-1].t + delta_t
    us_path += [path[-1]]

    return us_path


def back_track_state(end: SE2State) -> List[SE2State]:
    path = []
    while end.parent is not None:
        path += [end]
        parent = end.parent
        end = deepcopy(parent)
    path += [end]

    return path[::-1]


def compute_curvature(x, y):
    """
    input  : the coordinate of the three point
    output : the curvature and norm direction
    refer to https://github.com/Pjer-zhang/PJCurvature for detail
    """

    t_a = np.linalg.norm([x[1] - x[0], y[1] - y[0]])
    t_b = np.linalg.norm([x[2] - x[1], y[2] - y[1]])

    M = np.array([[1, -t_a, t_a**2], [1, 0, 0], [1, t_b, t_b**2]])

    a = np.matmul(np.linalg.inv(M), x)
    b = np.matmul(np.linalg.inv(M), y)

    kappa = 2 * (a[2] * b[1] - b[2] * a[1]) / (a[1] ** 2.0 + b[1] ** 2.0) ** (1.5)
    return kappa, [b[1], -a[1]] / np.sqrt(a[1] ** 2.0 + b[1] ** 2.0)


def back_track_close(
    close_list,
    vehicle_cfg=VehicleConfig(),
):
    if len(close_list) < 1:
        print("empty close list")
        return
    end = close_list[-1]

    path = back_track_state(end)

    for i in range(0, len(path) - 1):
        path[i].t = i * vehicle_cfg.T
        # dx = path[i + 1].x - path[i].x
        # dy = path[i + 1].y - path[i].y
        # ds = np.sqrt(dx * dx + dy * dy)
        # dh = (path[i + 1].so2 - path[i].so2).heading
        # path[i].curv = ds / dh

    path[-1].t = len(path) * vehicle_cfg.T
    return path


def back_track_merge(
    merge_state_from_start: SE2State,
    merge_state_from_goal: SE2State,
    vehicle_cfg=VehicleConfig(),
):
    end: SE2State = merge_state_from_start  # take from start
    path1: List[SE2State] = back_track_state(end)

    end: SE2State = merge_state_from_goal  # take from goal
    path2: List[SE2State] = back_track_state(end)

    for se2state in path2:
        se2state.v = -se2state.v

    path2.reverse()

    path = path1 + path2
    for i in range(0, len(path)):
        path[i].t = i * vehicle_cfg.T

    return path


from gridmap import (
    GridMap,
    outboundary,
    collision,
    generate_heuristic_map,
    generate_obstacle_grid,
    generate_visited_map_2d,
    generate_visited_map_3d,
)


def breadth_first_search(
    start: SE2State,
    goal: SE2State,
    grid_map: GridMap,
):
    q = Queue()
    close_list: list[SE2State] = []
    open_dict: dict[Tuple, SE2State] = dict()

    start.cost_to_here = start.cost_to_gridstate(start)
    start.cost_to_goal = start.cost_to_gridstate(goal)
    q.put(start)

    open_dict[start.get_index_2d()] = start

    print("Heuristic initializing")
    visited_map = generate_visited_map_2d(grid_map)
    obstacle_map = generate_obstacle_grid(grid_map)
    heuristic = generate_heuristic_map(grid_map)

    print("Heuristic start search")

    it = 0
    while not q.empty():
        current_state: SE2State = q.get()
        it += 1
        if it % 10000 == 0:
            print(f"Heuristic search iter {it:5d}")

        visited_map[current_state.x_index][current_state.y_index] = True
        heuristic[current_state.x_index][
            current_state.y_index
        ] = current_state.cost_to_here

        close_list += [current_state]

        for next_state in get_next_grid_state(current_state, grid_map):
            if outboundary(grid_map, next_state):
                continue
            elif obstacle_map[next_state.x_index][next_state.y_index]:
                continue
            elif visited_map[next_state.x_index][next_state.y_index]:
                continue
            else:
                next_state.cost_to_here = (
                    current_state.cost_to_here
                    + current_state.cost_to_gridstate(next_state)
                )

                next_index = next_state.get_index_2d()
                if next_index in open_dict:
                    if open_dict[next_index].cost_to_here > next_state.cost_to_here:
                        open_dict[next_index].cost_to_here = next_state.cost_to_here
                        open_dict[next_index].parent = current_state
                else:
                    open_dict[next_state.get_index_2d()] = next_state
                    q.put(next_state)

    print(f"Heuristic search finished, iter {it:5d} ")

    return heuristic


def analystic_expand(
    current_state: SE2State,
    goal_state: SE2State,
    grid: GridMap,
    vehicle_cfg=VehicleConfig(),
):
    """
    TODO analystic expand trajectory
    """

    # def quintic_coefficient(start, end, T):
    #     """Calculate quintic polynomial."""
    #     b = np.array([start, end, 0, 0, 0, 0])
    #     A = np.array(
    #         [
    #             [1, 0, 0, 0, 0, 0],
    #             [1, 1 * T, 1 * T**2, 1 * T**3, 1 * T**4, 1 * T**5],
    #             [0, 1, 0, 0, 0, 0],
    #             [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
    #             [0, 0, 2, 0, 0, 0],
    #             [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
    #         ]
    #     )

    #     p = np.dot(np.linalg.inv(A), b)
    #     return p

    # def linear_intep(start, end, T):
    #     A = np.array([[0, 1], [T, 1]])
    #     b = np.array([start, end])
    #     p = np.dot(np.linalg.inv(A), b)
    #     return p

    # def linear_value(p, t):
    #     return p @ np.array([t, 1]).T

    # def quintic_value(p, t):
    #     x = (
    #         p[0]
    #         + p[1] * t
    #         + p[2] * t**2
    #         + p[3] * t**3
    #         + p[4] * t**4
    #         + p[5] * t**5
    #     )
    #     x_dot = (
    #         p[1]
    #         + 2 * p[2] * t
    #         + 3 * p[3] * t**2
    #         + 4 * p[4] * t**3
    #         + 5 * p[5] * t**4
    #     )

    #     return x, x_dot

    # dx = goal_state.x - current_state.x
    # dy = goal_state.y - current_state.y
    # dheading = (goal_state.so2 - current_state.so2).heading

    # distance = np.sqrt(dx * dx + dy * dy)
    # T = int(distance / gridmap_cfg.xy_resolution)
    # Th = int(abs(dheading) / gridmap_cfg.heading_resolution)

    # px = quintic_coefficient(current_state.x, goal_state.x, T)
    # py = quintic_coefficient(current_state.y, goal_state.y, T)
    # ph = quintic_coefficient(current_state.heading, goal_state.heading, T)

    # """
    # TODO : ERROR.
    # """
    # analystic_path = []
    # for t in range(1, int(T)):
    #     next_h, next_hdot = quintic_value(ph, t)
    #     next_x, next_xdot = quintic_value(px, t)
    #     next_y, next_ydot = quintic_value(py, t)

    #     # next_h = linear_value(ph, t)
    #     # dx = next_x - current_state.x
    #     # dy = next_y - current_state.y
    #     # dh = next_h - current_state.heading

    #     # w_vec = np.array([dx, dy, 0])
    #     # v_vec = np.array(
    #     #     [np.cos(current_state.heading), np.sin(current_state.heading), 0]
    #     # )
    #     # wv_linalg = np.linalg.norm(w_vec) * np.linalg.norm(v_vec)
    #     # dh = np.arccos(np.clip(np.dot(w_vec, v_vec) / (wv_linalg), -1.0, 1.0))
    #     # cross = np.cross(v_vec, w_vec)
    #     # dh *= np.sign(cross[2])

    #     # se2 = current_state.se2 + SE2(dx, dy, dh)
    #     # se2state = SE2State.from_se2(se2)

    #     se2state = SE2State(next_x, next_y, next_h)

    #     if collision(grid, se2state):
    #         return []

    #     dh = (se2state.so2 - current_state.so2).heading
    #     # if abs(dh) > 0.1:
    #     #     return []
    #     v = np.sqrt(next_xdot**2 + next_ydot**2)
    #     se2state.v = v * np.cos(next_h)

    #     se2state.delta = np.clip(
    #         np.arctan(dh * vehicle_cfg.wheel_base / se2state.v),
    #         -vehicle_cfg.max_front_wheel_angle,
    #         vehicle_cfg.max_front_wheel_angle,
    #     )
    #     analystic_path += [se2state]
    #     current_state = se2state

    # # analystic_path += [goal_state]

    # return analystic_path
    raise NotImplementedError("analystic_expand function has not be implementated.")


def hybrid_a_star_search(
    start: SE2State,
    goal: SE2State,
    grid_map: GridMap,
    search_cfg=SearchConfig(),
):
    import faulthandler

    faulthandler.enable()

    compute_3d_index(grid_map, start)
    compute_3d_index(grid_map, goal)

    print(f"Start : {start}")
    print(f"Goal : {goal}")

    max_it = search_cfg.max_iteration
    max_heading_index_error = search_cfg.max_heading_index_error
    penalty_change_gear = search_cfg.penalty_change_gear
    heuristic_weight = search_cfg.heuristic_weight

    visited_map = generate_visited_map_3d(grid_map)

    heuristic_map = breadth_first_search(goal, start, grid_map)
    start.cost_to_here = start.cost_to_state(start)
    start.cost_to_goal = heuristic_map[start.x_index][start.y_index]
    q = PriorityQueue()
    close_list = []

    open_dict = dict()
    open_dict[start.get_index_3d()] = start
    q.put((start.cost(), start))

    goal_set = get_next_states(goal, grid_map)

    it = 0

    print("Hybrid a star start seaching")

    while ((not q.empty())) and (it < max_it):
        it += 1
        if it % 10000 == 0:
            print(f"Hybrid seaching , iter : {it:5d}")

        hybrid_a_star_search_step(
            q,
            open_dict,
            close_list,
            heuristic_weight,
            penalty_change_gear,
            grid_map,
            heuristic_map,
            visited_map,
        )
        current_state = close_list[-1]
        # analystic_state_list = analystic_expand(current_state, goal, grid_map)
        # if len(analystic_state_list) > 0:
        #     print(f"Find goal. Iteration {it:5d}")
        #     return back_track_close(close_list) + analystic_state_list

        if current_state in goal_set:
            # current_state.parent = goal
            print(f"Find goal. Iteration {it:5d}")
            return back_track_close(close_list)

    print(f"Search failed, Iteration {it:5d}")
    return back_track_close(close_list)


def bidirection_hybrid_a_star_search(
    start: SE2State,
    goal: SE2State,
    grid_map: GridMap,
    search_cfg=SearchConfig(),
):
    import faulthandler
    import matplotlib.pyplot as plt

    faulthandler.enable()

    compute_3d_index(grid_map, start)
    compute_3d_index(grid_map, goal)

    print(f"Start : {start}")
    print(f"Goal : {goal}")

    """
    Search Configure.
    """

    max_it = search_cfg.max_iteration
    heuristic_weight = search_cfg.heuristic_weight
    penalty_change_gear = search_cfg.penalty_change_gear
    visited_exam_interval = search_cfg.visited_exam_interval
    """
    Primitive init.
    """
    print("Primitive initializing")

    heuristic_map = breadth_first_search(goal, start, grid_map)

    # import utils
    # plt.figure(1, figsize=[8, 8])
    # heatmap = np.array(heuristic_map) * 5
    # utils.plot_heatmap(heatmap)
    # plt.draw()
    # plt.show()

    start.cost_to_here = start.cost_to_state(start)
    start.cost_to_goal = heuristic_map[start.x_index][start.y_index]
    q = PriorityQueue()
    q.put((start.cost(), start))

    open_dict: dict[Tuple, SE2State] = dict()
    open_dict[start.get_index_3d()] = start
    close_list = []
    visited_map = generate_visited_map_3d(grid_map)

    """
    Dual init.
    """

    print("Dual initializing")
    dual_start = deepcopy(goal)
    dual_goal = deepcopy(start)

    dual_visited_map = deepcopy(visited_map)

    dual_heuristic_map = breadth_first_search(dual_goal, dual_start, grid_map)

    # import utils

    # heatmap = np.array(dual_heuristic_map) * 5
    # utils.plot_heatmap(heatmap)
    # plt.draw()
    # plt.show()

    dual_start.cost_to_here = dual_start.cost_to_state(dual_start)
    dual_start.cost_to_goal = dual_heuristic_map[dual_start.x_index][dual_start.y_index]
    dual_q = PriorityQueue()
    dual_q.put((dual_start.cost(), dual_start))

    dual_open_dict: dict[Tuple, SE2State] = dict()
    dual_open_dict[dual_start.get_index_3d()] = dual_start
    dual_close_list = []

    it = 0

    print("Bidirection hybrid a star start seaching.")

    while True:
        it += 1

        if it > max_it:
            raise TimeoutError("Search has over max iteration.")

        if q.empty() and dual_q.empty():
            raise BufferError("All Queue is empty, Search failed.")

        hybrid_a_star_search_step(
            q,
            open_dict,
            close_list,
            heuristic_weight,
            penalty_change_gear,
            grid_map,
            heuristic_map,
            visited_map,
        )
        hybrid_a_star_search_step(
            dual_q,
            dual_open_dict,
            dual_close_list,
            heuristic_weight,
            penalty_change_gear,
            grid_map,
            dual_heuristic_map,
            dual_visited_map,
        )

        if it % visited_exam_interval == 0:
            merge_prim_index, merge_dual_index = visitedmap_check(
                visited_map, dual_visited_map, open_dict, dual_open_dict
            )
            if merge_prim_index is None:
                print(f"Merge. Iteration {it:5d}, can not find goal.")
                continue

            print(f"Find goal. Iteration {it:5d}")
            return back_track_merge(
                open_dict[merge_prim_index], dual_open_dict[merge_dual_index]
            )


def visitedmap_check(
    visited_map: List[List[List[bool]]],
    dual_visited_map: List[List[List[bool]]],
    open_dict: dict,
    dual_open_dict: dict,
    search_cfg: SearchConfig = SearchConfig(),
):
    visited_array_3d = np.array(visited_map) * 1
    dual_visited_array_3d = np.array(dual_visited_map) * 1

    visited_array_2d = np.sum(visited_array_3d, -1)
    visited_array_2d = np.where(visited_array_2d > 0, 1, 0)
    dual_visited_array_2d = np.sum(dual_visited_array_3d, -1)
    dual_visited_array_2d = np.where(dual_visited_array_2d > 0, 1, 0)

    merge_2dmap = visited_array_2d + dual_visited_array_2d

    merge_prim_index = None
    merge_dual_index = None

    if np.any(merge_2dmap == 2):
        merge_2dindex = np.where(merge_2dmap == 2)
        index2d_array = np.array(merge_2dindex)
        _, num = index2d_array.shape

        min_cost = np.inf

        for i in range(num):
            visited_heading_index = np.where(
                visited_array_3d[index2d_array[0, i], index2d_array[1, i]] == 1
            )
            visited_heading_index = np.sort(visited_heading_index).flatten()

            dual_visited_heading_index = np.where(
                dual_visited_array_3d[index2d_array[0, i], index2d_array[1, i]] == 1
            )
            dual_visited_heading_index = np.sort(dual_visited_heading_index).flatten()

            if len(visited_heading_index) < 1 or len(dual_visited_heading_index) < 1:
                continue

            prim_heading_index = None
            min_heading_index_error = int(1e8)

            for pi in visited_heading_index:
                for di in dual_visited_heading_index:
                    heading_index_error = abs(pi - di)
                    if heading_index_error <= search_cfg.max_heading_index_error:
                        if heading_index_error < min_heading_index_error:
                            min_heading_index_error = heading_index_error
                            prim_heading_index = pi
                            dual_heading_index = di

            if prim_heading_index is None:
                continue

            prim_index3d = (
                index2d_array[0, i],
                index2d_array[1, i],
                prim_heading_index,
            )

            dual_index3d = (
                index2d_array[0, i],
                index2d_array[1, i],
                dual_heading_index,
            )

            cost = (
                open_dict[prim_index3d].cost_to_here
                + dual_open_dict[dual_index3d].cost_to_here
                + min_heading_index_error
            )

            if cost < min_cost:
                min_cost = cost
                merge_prim_index = prim_index3d
                merge_dual_index = dual_index3d

    return merge_prim_index, merge_dual_index


def hybrid_a_star_search_step(
    q: PriorityQueue,
    open_dict: dict,
    close_list: list,
    heuristic_weight: float,
    penalty_change_gear: int,
    grid_map: GridMap,
    heuristic_map: List[List[float]],
    visited_map: List[List[bool]],
):
    """
    all parameter is passed by reference.
    """
    if q.empty():
        return

    current_state: SE2State = q.get()[1]

    current_state.visited = True

    visited_map[current_state.x_index][current_state.y_index][
        current_state.heading_index
    ] = True

    close_list += [current_state]

    for next_state in get_next_states(current_state, grid_map):
        if collision(grid_map, next_state):
            continue
        elif visited_map[next_state.x_index][next_state.y_index][
            next_state.heading_index
        ]:
            continue
        else:
            cost_to_here = current_state.cost_to_here + current_state.cost_to_state(
                next_state
            )
            if (
                next_state.direction_index > 3 and current_state.direction_index <= 3
            ) or (
                next_state.direction_index <= 3 and current_state.direction_index > 3
            ):
                """Different direction."""
                cost_to_here = cost_to_here * penalty_change_gear

            else:
                """Same direction."""
                pass

            next_state.cost_to_here = cost_to_here
            next_index = next_state.get_index_3d()
            next_state.parent = current_state

            next_state.cost_to_goal = (
                heuristic_weight * heuristic_map[next_state.x_index][next_state.y_index]
            )

            try:
                if next_index in list(open_dict.keys()):
                    if open_dict[next_index].cost() > next_state.cost():
                        open_dict[next_index].cost_to_goal = next_state.cost_to_goal
                        open_dict[next_index].cost_to_here = next_state.cost_to_here
                        open_dict[next_index].parent = current_state
                else:
                    open_dict[next_state.get_index_3d()] = next_state
                    q.put((next_state.cost(), next_state), block=False)
            except:
                print("Queue has full")


import multiprocessing


def hybrid_a_star_worker(
    stop_event: multiprocessing.Event,
    visited_map_q: multiprocessing.Queue,
    start: SE2State,
    goal: SE2State,
    grid_map: GridMap,
    search_cfg=SearchConfig(),
):
    """
    worker
    """
    import faulthandler
    import os

    faulthandler.enable()

    max_it = search_cfg.max_iteration
    penalty_change_gear = search_cfg.penalty_change_gear
    heuristic_weight = search_cfg.heuristic_weight
    merge_interval = search_cfg.visited_exam_interval
    visited_map = generate_visited_map_3d(grid_map)

    heuristic_map = breadth_first_search(goal, start, grid_map)
    start.cost_to_here = start.cost_to_state(start)
    start.cost_to_goal = heuristic_map[start.x_index][start.y_index]
    q = PriorityQueue()
    close_list = []

    open_dict = dict()
    open_dict[start.get_index_3d()] = start
    q.put((start.cost(), start))

    it = 0

    print(f"PID {os.getpid()} start searching")

    while it < max_it:
        it += 1

        if stop_event.is_set():
            print(f"PID {os.getpid()} find goal.")
            break

        if q.empty() and it % merge_interval == 0:
            print(f"PID {os.getpid()} empty.")

        if visited_map_q.empty() and it % merge_interval == 0:
            print(f"PID {os.getpid()} search, iter {it:5d}")

            visited_map_q
            dict_pass = deepcopy(open_dict)
            visited_pass = deepcopy(visited_map)
            visited_map_q.put((visited_pass, dict_pass))

        hybrid_a_star_search_step(
            q,
            open_dict,
            close_list,
            heuristic_weight,
            penalty_change_gear,
            grid_map,
            heuristic_map,
            visited_map,
        )

    while not visited_map_q.empty():
        visited_map_q.get()  # clear the queue
    visited_map_q.close()


def multiprocess_bidirection_hybrid_a_star_search(
    start: SE2State,
    goal: SE2State,
    grid_map: GridMap,
    search_cfg=SearchConfig(),
):
    import faulthandler
    import time

    faulthandler.enable()

    """
    Search Configure.
    """
    print(f"Start : {start}")
    print(f"Goal : {goal}")

    compute_3d_index(grid_map, start)
    compute_3d_index(grid_map, goal)
    """
    Primitive.
    """
    stop_event = multiprocessing.Event()

    q1 = multiprocessing.Queue(maxsize=2)
    worker1 = multiprocessing.Process(
        target=hybrid_a_star_worker,
        args=(stop_event, q1, start, goal, grid_map, search_cfg),
    )
    worker1.start()

    """
    Dual.
    """

    dual_start = deepcopy(goal)
    dual_goal = deepcopy(start)

    q2 = multiprocessing.Queue(maxsize=2)
    worker2 = multiprocessing.Process(
        target=hybrid_a_star_worker,
        args=(stop_event, q2, dual_start, dual_goal, grid_map, search_cfg),
    )
    worker2.start()

    # print("Multiprocessing bidirection hybrid a star start.")

    maxit = search_cfg.max_iteration / search_cfg.visited_exam_interval

    it = 0
    visited_map, dual_visited_map = None, None

    while True:
        it += 1
        if it > maxit:
            raise TimeoutError("Bidirection Hybrid A Star Search Failed.")

        time.sleep(0.1)

        if not q1.empty():
            visited_map, open_dict = q1.get()

        if not q2.empty():
            dual_visited_map, dual_open_dict = q2.get()

        if visited_map is not None and dual_visited_map is not None:
            merge_prim_index, merge_dual_index = visitedmap_check(
                visited_map, dual_visited_map, open_dict, dual_open_dict
            )
            if merge_prim_index is not None and merge_dual_index is not None:
                break

    q1.close()
    q2.close()

    worker1.terminate()
    worker2.terminate()

    # print("\033[32m=== Search Success. ===\033[0m")

    return back_track_merge(
        open_dict[merge_prim_index], dual_open_dict[merge_dual_index]
    )
