from statemachine import Vehicle
from se2state import SE2State
import random


def main():
    print("\033[32m=== Simulation Start. ===\033[0m")

    veh = Vehicle()
    veh.initialize()

    TASK_NUM = 1
    start_states = [SE2State(4, 15, -3.12), SE2State(15.90, 14.03, 0.11)]

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
