# from ParkingSimulation.statemachine import Vehicle
import numpy as np
import time
import random

from typing import TypeVar, List
from settings import *
from search import (
    multiprocess_bidirection_hybrid_a_star_search,
    bidirection_hybrid_a_star_search,
    upsample_smooth,
    downsample_smooth,
)
from config import ControlConfig
from se2state import SE2State, generate_vehicle_vertexes
import sys

from gridmap import GridMap, collision, generate_gridmap_from_polygon
from coordinate_transform import move_polygon_list, move_polygon

Vehicle = TypeVar("Vehicle")


class State:
    def handle(self, vehicle: Vehicle):
        # vehicle.step()
        pass

    def __repr__(self) -> str:
        return f"State baseclass"


class ParkingFail(State):
    def handle(self, vehicle: Vehicle):
        print(f"Current state is : {vehicle.state}")
        print("\033[31m=== Parking Failed. ===\033[0m")
        sys.exit(1)

    def __repr__(self) -> str:
        return f"ParkingFail"


class ParkingSucess(State):
    def handle(self, vehicle: Vehicle):
        print(f"Current state is : {vehicle.state}")
        print("\033[32m=== Parking Success. ===\033[0m")
        vehicle.env.close()

        plt.figure(1, figsize=[8, 10])
        utils.plot_control(vehicle.reference_trajectory)
        utils.plot_control(vehicle.tracking_trajectory)
        plt.show()

        sys.exit(0)

    def __repr__(self) -> str:
        return f"ParkingSucess"


class Perception(State):
    def handle(self, vehicle: Vehicle):
        vehicle.step()
        if (
            vehicle.current_se2state is not None
            and len(vehicle.global_parking_polygons) > 0
        ):
            vehicle.set_state(Decision())

        print(f"Current state is : {vehicle.state}")

    def __repr__(self) -> str:
        return f"Perception"


class Decision(State):
    def handle(self, vehicle: Vehicle):
        vehicle.step()

        if (
            vehicle.current_se2state is not None
            and len(vehicle.global_parking_polygons) > 0
        ):
            """
            Select the best parking space.
            """

            grid = generate_gridmap_from_polygon(
                vehicle.global_obstacle_polygons,
                vehicle.global_parking_polygons,
                vehicle.current_se2state,
            )

            vehicle.gridmap = grid

            goal_se2statelist = []
            for parking_polygon in vehicle.global_parking_polygons:
                goal_se2statelist += generate_goal_states_from_parking_rectpolygon(
                    parking_polygon
                )

            sorted_goal_se2states = sort_goal_states(
                goal_se2statelist, vehicle.current_se2state
            )
            print(f"\033[32mSelect the best parking space.\033[0m")

            goal_se2state = sorted_goal_se2states[0]
            vehicle.goal_se2state = goal_se2state

            vehicle.set_state(Planing())
        else:
            vehicle.set_state(Perception())

        print(f"Current state is : {vehicle.state}")

    def __repr__(self) -> str:
        return f"Decision"


from optimizer import Optimizer


class Planing(State):
    def __init__(self) -> None:
        super().__init__()
        self.planning_count = 0

    def handle(self, vehicle: Vehicle):
        """
        frontend search. if failed . set planning.
        backend optimization. if failed. set planning.
        count +=1
        if successed, set tracking
        """
        vehicle.step()

        try:
            print("\033[32m===Start frontend search.===\033[0m")
            path = multiprocess_bidirection_hybrid_a_star_search(
                vehicle.current_se2state, vehicle.goal_se2state, vehicle.gridmap
            )
            # path = bidirection_hybrid_a_star_search(
            #     vehicle.current_se2state, vehicle.goal_se2state, vehicle.gridmap
            # )

            ctrl_cfg = ControlConfig()
            interval = int((path[1].t - path[0].t) / ctrl_cfg.dt) - 1
            if interval > 3:
                us_path = upsample_smooth(path, interval)
            else:
                us_path = path

            print("\033[32m===Start backend Optimize.===\033[0m")
            opti = Optimizer()
            opti.initialize(us_path, vehicle.global_obstacle_polygons, vehicle.gridmap)
            opti.solve()

            reference_trajectory = opti.extract_result(
                current_time=vehicle.current_se2state.t
            )

            vehicle.reference_trajectory = reference_trajectory
            vehicle.tracking_trajectory = []

            print(f"Current state is : {vehicle.state}")

            vehicle.set_state(Tracking())
        except Exception as e:
            print(f"\033[31m Catch Exception {e} \033[0m")
            print(f"Replan ...{self.planning_count:4d}")
            """
            1. TODO reduce discrete grid size if a star queue is empty.
            """
            self.planning_count += 1

            if self.planning_count >= 5:
                vehicle.set_state(ParkingFail())

    def __repr__(self) -> str:
        return f"Planing"


class CheckCollision(State):
    def handle(self, vehicle: Vehicle):
        if vehicle.polygon_manager.check_collision(
            vehicle.current_se2state, vehicle.reference_trajectory
        ):
            u = vehicle.controller.emergency_stop(vehicle.current_se2state)
            print(
                f"\033[33m curre trajectory has collision with some obstacle. replaning.\033[0m"
            )
            vehicle.set_state(Decision())
        else:
            target_se2state = get_closest_time_states(
                vehicle.reference_trajectory, vehicle.current_se2state
            )
            u = vehicle.controller.action(
                vehicle.current_se2state,
                target_se2state,
            )
            vehicle.set_state(Tracking())

        vehicle.step(u)

        print(f"Current state is : {vehicle.state}, time is {time.time():10.2f}")

    def __repr__(self) -> str:
        return f"CheckCollision"


class Tracking(State):
    def handle(self, vehicle: Vehicle):
        """
        Check collision every 10 ms.
        """

        target_se2state = get_closest_time_states(
            vehicle.reference_trajectory, vehicle.current_se2state
        )
        u = vehicle.controller.action(
            vehicle.current_se2state,
            target_se2state,
        )

        done = vehicle.step(u)

        if done:
            vehicle.set_state(ParkingSucess())

        if (
            int(vehicle.env.step_count)
            % vehicle.polygon_manager.collision_check_interval
            == 0
        ):
            vehicle.set_state(CheckCollision())

        print(f"Current state is : {vehicle.state}, time is {time.time():10.2f}")

    def __repr__(self) -> str:
        return f"Tracking"


from geometry import PolygonContainer, Polygon, polygon_intersect_polygon


class PolygonManager:
    def __init__(self) -> None:
        self.global_obstacle_polygon_container: PolygonContainer = PolygonContainer()
        self.global_parking_polygon_container: PolygonContainer = PolygonContainer()
        self.collision_check_interval = 100

    def check_collision(
        self, current_se2state: SE2State, reference_trajectory: List[SE2State]
    ):
        index = 0

        for se2state in reference_trajectory:
            if current_se2state.t > se2state.t:
                index += 1

        for i in range(index, len(reference_trajectory)):
            se2state = reference_trajectory[i]
            vehicle_vertices = generate_vehicle_vertexes(se2state)
            vehicle_polygon = Polygon(vehicle_vertices)
            for obstacle_polygon in self.global_obstacle_polygon_container:
                if polygon_intersect_polygon(vehicle_polygon, obstacle_polygon):
                    print(f"\033[31m==Collision happened.==\033[0m")
                    print(f"CheckCollision se2 state is...{se2state}")
                    return True

        return False


from parking_env import ParkingEnvironment
from controller import Controller


class Vehicle:
    def __init__(self) -> None:
        self.state: State = None
        self.polygon_manager: PolygonManager = None

        self.env: ParkingEnvironment = None
        self.path_opti: Optimizer = None
        self.controller: Controller = None
        self.file_name: str = None

    @property
    def current_se2state(self):
        return self.env.se2state

    @current_se2state.setter
    def current_se2state(self, current_se2state: SE2State):
        self.env.se2state = current_se2state

    @property
    def goal_se2state(self):
        return self.env.goal_se2state

    @goal_se2state.setter
    def goal_se2state(self, goal_se2state: SE2State):
        self.env.goal_se2state = goal_se2state

    @property
    def reference_trajectory(self):
        return self.env.reference_trajectory

    @reference_trajectory.setter
    def reference_trajectory(self, reference_trajectory: List[SE2State]):
        self.env.reference_trajectory = reference_trajectory

    @property
    def tracking_trajectory(self):
        return self.env.tracking_trajectory

    @tracking_trajectory.setter
    def tracking_trajectory(self, tracking_trajectory: List[SE2State]):
        self.env.tracking_trajectory = tracking_trajectory

    @property
    def global_obstacle_polygons(self):
        return self.polygon_manager.global_obstacle_polygon_container

    @property
    def global_parking_polygons(self):
        return self.polygon_manager.global_parking_polygon_container

    def initialize(self):
        self.env = ParkingEnvironment(render_mode="rgb_array")
        self.polygon_manager = PolygonManager()
        self.path_opti = Optimizer()
        self.controller = Controller()
        self.set_state(Perception())

    def set_state(self, state: State):
        self.state = state

    def add_global_obstacle_polygon(self, obstacle_polygon_list: List[Polygon]):
        if len(obstacle_polygon_list) == 0:
            return
        for obstacle_polygon in obstacle_polygon_list:
            self.polygon_manager.global_obstacle_polygon_container.__iadd__(
                obstacle_polygon
            )

    def add_global_parking_polygon(self, parking_polygon_list: List[Polygon]):
        if len(parking_polygon_list) == 0:
            return

        for parking_polygon in parking_polygon_list:
            self.polygon_manager.global_parking_polygon_container += parking_polygon

    def action(self):
        self.state.handle(self)

    def step(self, u: np.ndarray = np.array([0, 0], dtype=np.float32)):
        current_se2state, reward, t, done, info = self.env.step(u)

        global_obstacle_polygon = move_polygon_list(
            info["local_obstacle_polygon_list"], current_se2state.se2
        )

        global_parking_polygon = move_polygon_list(
            info["local_parking_polygon_list"], current_se2state.se2
        )

        self.add_global_obstacle_polygon(global_obstacle_polygon)
        self.add_global_parking_polygon(global_parking_polygon)

        return done


def main():
    print("\033[32m=== Simulation Start. ===\033[0m")

    veh = Vehicle()
    veh.initialize()

    TASK_NUM = 1
    veh.file_name = "se2path" + str(TASK_NUM) + ".pickle"
    start_states = [SE2State(4, 15, -3.12), SE2State(15.90, 14.03, 0.38)]

    start_se2state = start_states[TASK_NUM]
    env_opts = {"start_se2state": start_se2state}
    veh.env.reset(options=env_opts)

    """
    random start, random dynamics obstacles.
    """
    start_se2state
    start_perception_index = random.randint(10, 20)
    count = 0

    while True:
        veh.action()
        count += 1

        if count == start_perception_index:
            veh.env.perception()


if __name__ == "__main__":
    main()
