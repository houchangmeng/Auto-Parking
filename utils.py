import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import List, Tuple, TypeVar

SE2State = TypeVar("SE2State")
Polygon = TypeVar("Polygon")
Point = TypeVar("Point")


def plot_line(p1: Tuple, p2: Tuple, linetype="-b", alpha=1.0):
    x = np.hstack((p1[0], p2[0]))
    y = np.hstack((p1[1], p2[1]))
    plt.plot(x, y, linetype, alpha=alpha)


def plot_polygon(polygon: Polygon, linetype="-b", alpha=1.0):
    plot_polygon_vertexes(polygon.vertexes, linetype, alpha)


def plot_polygon_vertexes(vertexes_list: List[Point], linetype="-b", alpha=1.0):
    point_array = np.array(vertexes_list)
    point_array = np.vstack((point_array, point_array[0, :]))
    plt.plot(point_array[:, 0], point_array[:, 1], linetype, alpha)
    plt.draw()


def plot_vehicle(se2state: SE2State):
    from se2state import generate_vehicle_vertexes, generate_wheels_vertexes

    v_b = generate_vehicle_vertexes(se2state)
    plot_polygon_vertexes(v_b, "-g", alpha=0.6)

    plt.plot(se2state.x, se2state.y, "gs")  # rear axle center.
    for v_w in generate_wheels_vertexes(se2state):
        plot_polygon_vertexes(v_w, "-r", alpha=0.2)


def plot_task(
    obstacle_polygon_list: List[Polygon],
    start_se2state: SE2State,
    goal_se2state: SE2State,
):
    for obs_polygon in obstacle_polygon_list:
        plot_polygon(obs_polygon)
        plt.draw()

    plt.axis("equal")

    plot_vehicle(start_se2state)
    plt.text(
        start_se2state.x,
        start_se2state.y,
        f" {start_se2state.heading:5.2f}",
        color="b",
    )
    plt.draw()

    plot_vehicle(goal_se2state)
    plt.text(
        start_se2state.x,
        start_se2state.y,
        f" {start_se2state.heading:5.2f}",
        color="r",
        size=10,
    )
    plt.draw()
    # plt.pause(0.001)


def plot_path(path: List[SE2State], label="path"):
    x = np.array([se2.x for se2 in path])
    y = np.array([se2.y for se2 in path])
    plt.plot(x, y, label=label)
    plt.legend()
    plt.draw()
    # plt.pause(0.001)


def plot_trajectory_animation(path: List[SE2State]):
    for se2 in path:
        plot_vehicle(se2)
        plt.draw()
        plt.pause(0.01)


def plot_control(path: List[SE2State]):
    v = np.array([se2.v for se2 in path])
    a = np.array([se2.a for se2 in path])
    j = np.array([se2.jerk for se2 in path])
    d = np.array([se2.delta for se2 in path])
    x = np.array([se2.x for se2 in path])
    y = np.array([se2.y for se2 in path])
    heading = np.array([se2.heading for se2 in path])

    N = 7
    if path[-1].t > 0:
        T = path[-1].t
    else:
        T = 0.2 * len(path)

    t = np.linspace(0, T, len(path))
    plt.subplot(N, 1, 1)
    plt.plot(t, v, label="v")
    plt.legend()

    plt.subplot(N, 1, 2)
    plt.plot(t, a, label="a")
    plt.legend()

    plt.subplot(N, 1, 3)
    plt.plot(t, j, label="jerk")
    plt.legend()

    plt.subplot(N, 1, 4)
    plt.plot(t, d, label="delta")
    plt.legend()
    plt.subplot(N, 1, 5)
    plt.plot(t, x, label="x")
    plt.legend()
    plt.subplot(N, 1, 6)
    plt.plot(t, y, label="y")
    plt.legend()

    plt.subplot(N, 1, 7)
    plt.plot(t, heading, label="heading")
    plt.legend()

    # plt.subplot(5, 1, 5)
    # plt.plot(t, d, label="curvature")
    # plt.legend()


def plot_heatmap(data: np.ndarray):
    data = np.array(data).T
    plt.gcf()
    im = plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)
    plt.colorbar(im, extend="both", extendrect=True)


def record_gif(file_name="image_list.pickle"):
    """
    Make a gif image for giving image folder
    """

    import imageio
    import time
    import pickle

    # file_name = "image_list.pickle"
    with open(file_name, "rb") as f:
        image_list = pickle.load(f)

    imageio.mimsave(time.asctime() + ".gif", image_list, "GIF", duration=0.02)


def record_video(file_name="image_list.pickle"):
    import pickle
    import cv2

    with open(file_name, "rb") as f:
        image_list = pickle.load(f)

    fps = 50  # 视频帧率
    size = (800, 1600)  # 需要转为视频的图片的尺寸

    video = cv2.VideoWriter(
        r"F:\Writing\PHD\AAAI2022\composed_vertical.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )  # 创建视频流对象-格式一

    for img_path in image_list:
        image = cv2.imread(
            img_path
        )  # 注意image的尺寸与size的尺寸是相反的，比如size是(800, 1600)，那么image的尺寸就应该是（1600，800）
        video.write(image)

    video.release()
    cv2.destroyAllWindows()
