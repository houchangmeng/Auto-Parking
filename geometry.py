import numpy as np
from typing import Tuple, TypeVar, List, Tuple
from copy import deepcopy

"""
Geometry functions.
"""

Point = TypeVar("Point", Tuple, List)

Line = TypeVar("Line", List[Tuple], List[List])

Circle = TypeVar("Circle", Tuple, List)  # x, y ,radius


class Polygon:
    """
    Convex Polygon, anticlockwise vertexes/lines.
    """

    def __init__(self, vertexes: List[Point]):
        # anticlockwise_vertexes_sort(vertexes)
        self.vertexes: List[Point] = anticlockwise_vertexes_sort(vertexes)
        self.lines: List[Line] = vertexes_to_lines(vertexes)
        self.norms: List[Point] = lines_to_norm(self.lines)
        self.center: Point = tuple(np.mean(np.array(vertexes), axis=1))

    @property
    def ndarray(self):
        return np.array(self.vertexes).T

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Polygon):
            if len(self.vertexes) != len(other.vertexes):
                return False
            array1 = np.array(self.vertexes)
            array2 = np.array(other.vertexes)

            if np.sum(abs(array1 - array2)) > 1e-3:
                return False

            return True

        else:
            raise TypeError("Unsupported operand type for ==")

    def __repr__(self) -> str:
        return "polygon vertexes: " + str(self.vertexes)


class PolygonContainer:
    def __init__(self) -> None:
        self.polygon_list = []
        self.N = 0
        self.iter_index = 0

    def __len__(self):
        return len(self.polygon_list)

    def __getitem__(self, index):
        if index > self.N:
            raise IndexError("Index out range.")

        return self.polygon_list[index]

    def __add__(self, other: Polygon) -> bool:
        if isinstance(other, Polygon):
            if other in self.polygon_list:
                pass
            else:
                self.polygon_list += [other]

            return PolygonContainer(self.polygon_list)
        else:
            raise TypeError("Unsupported operand type for +")

    def __iadd__(self, other: Polygon) -> bool:
        if isinstance(other, Polygon):
            if other in self.polygon_list:
                pass
            else:
                self.polygon_list += [other]
                self.N += 1

            return self
        else:
            raise TypeError("Unsupported operand type for +=")

    def __sub__(self, other):
        if isinstance(other, Polygon):
            if other in self.polygon_list:
                self.polygon_list.remove(other)
            else:
                pass
            return PolygonContainer(self.polygon_list)
        else:
            raise TypeError("Unsupported operand type for -")

    def __isub__(self, other):
        if isinstance(other, Polygon):
            if other in self.polygon_list:
                self.polygon_list.remove(other)
                self.N -= 1
            else:
                pass

            return self
        else:
            raise TypeError("Unsupported operand type for -=")

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        last_index = self.iter_index
        if last_index >= len(self.polygon_list):
            raise StopIteration
        else:
            self.iter_index += 1
            return self.polygon_list[last_index]


def ndarray_to_vertexlist(vertexs_array: np.ndarray):
    """
    vertexs_array: 2 * n, n * 2
    """

    nx, n = vertexs_array.shape
    if nx != 2 and n == 2:
        tmp = nx
        nx = n
        n = tmp
        vertexs_array = vertexs_array.T
    elif nx == 2 and n != 2:
        pass
    else:
        raise ValueError("Check numpy array shape!")

    vertexlist = []
    for i in range(n):
        vertexlist += [(vertexs_array[0, i], vertexs_array[1, i])]

    return vertexlist


def move_vertexes_array(
    vertexs_array: np.ndarray, rot_angle: float, offset: np.ndarray
):
    """
    ### move vertexs, coord is fixed, change points.
    rot_angle [rad].
    """
    nv, n = vertexs_array.shape
    no, n = offset.shape
    if nv != 2 or no != 2:
        raise ValueError("Check numpy array shape! 2 * n")

    rot = np.array(
        [
            [np.cos(rot_angle), -np.sin(rot_angle)],
            [np.sin(rot_angle), np.cos(rot_angle)],
        ]
    )
    offset = np.array(offset).reshape((2, 1))
    return rot @ vertexs_array + offset


def change_vertexes_array_coord(
    vertexs_array: np.ndarray, rot_angle: float, offset: np.ndarray
):
    """
    ### change vertexs coord, point is fixed, change coord.
    ---
    rot_angle [rad]. rotate current coord to target coord

    offset [m]. trans current coord to target coord
    """
    nv, n = vertexs_array.shape
    no, n = offset.shape
    if nv != 2 or no != 2:
        raise ValueError("Check numpy array shape! 2 * n")

    rot = np.array(
        [
            [np.cos(rot_angle), np.sin(rot_angle)],
            [-np.sin(rot_angle), np.cos(rot_angle)],
        ]
    )

    return rot @ (vertexs_array - offset)


def to_left(line: Line, point: Point):
    """
    ### 2D To left test.

    l: line [(x1, y1), (x2, y2)]
    p: point (x1, y1)
    """
    vec1 = np.array(line[1]) - np.array(line[0])
    vec2 = np.array(point) - np.array(line[0])
    return np.cross(vec1, vec2) > 0


# def anticlockwise_vertexes_sort(vertexes: List[Point]):
#     """
#     ### anticlockwise sort.
#     """

#     vertexes_array = np.array(vertexes).T  # 2 * N
#     center_x, center_y = np.mean(vertexes_array, axis=1)

#     n = len(vertexes)

#     for i in range(n):
#         for j in range(n - i - 1):
#             line = [(center_x, center_y), (vertexes[j][0], vertexes[j][1])]
#             point = (vertexes[j + 1][0], vertexes[j + 1][1])
#             if not to_left(line, point):
#                 temp = vertexes[j]
#                 vertexes[j] = vertexes[j + 1]
#                 vertexes[j + 1] = temp

#     sorted_vertexes = vertexes
#     return sorted_vertexes


def get_bottom_point(vertexes: List[Point]):
    min_index = 0
    n = len(vertexes)
    for i in range(n):
        if vertexes[i][1] < vertexes[min_index][1] or (
            vertexes[i][1] == vertexes[min_index][1]
            and vertexes[i][0] < vertexes[min_index][0]
        ):
            min_index = i
    return min_index


def pointset_to_convex_hull(vertexes_list: List[Point]):
    N = len(vertexes_list)
    sorted_vertexes = anticlockwise_vertexes_sort(vertexes_list)

    if N < 3:
        raise ValueError("point too small.")
    if N == 3:
        return sorted_vertexes

    from scipy.spatial import ConvexHull

    hull = ConvexHull(np.array(sorted_vertexes))
    hull_array = hull.points[hull.vertices, :].T
    return ndarray_to_vertexlist(hull_array)


def anticlockwise_vertexes_sort(vertexes: List[Point]):
    """
    ### anticlockwise sort.
    """

    vertexes_array = np.array(vertexes).T
    center_x, center_y = np.mean(vertexes_array, axis=1)
    point_with_angle = []
    n = len(vertexes)
    for i in range(n):
        atan2 = np.arctan2(vertexes[i][1] - center_y, vertexes[i][0] - center_x)
        if atan2 < 0:
            atan2 += 2 * np.pi
        point_with_angle += [(vertexes[i], atan2)]

    for i in range(n):
        for j in range(n - i - 1):
            if point_with_angle[j][1] > point_with_angle[j + 1][1]:
                temp = point_with_angle[j]
                point_with_angle[j] = point_with_angle[j + 1]
                point_with_angle[j + 1] = temp

    sorted_vertexes = [vertex for vertex, _ in point_with_angle]

    return sorted_vertexes


def line_intersect_line(l1: Line, l2: Line):
    """
    ### Line interset line test.

    point: (x, y)
    l1: [point, point]
    l2: [point, point]
    """
    if to_left(l2, l1[0]) ^ to_left(l2, l1[1]):  # 异或， 一个在左边，一个在右边
        if to_left(l1, l2[0]) ^ to_left(l1, l2[1]):
            return True

    return False


def point_in_circle(point: Point, circle: Circle):
    """
    ### Point in circle test.

    circle: (x, y, r)
    point: (x, y)

    """
    if np.hypot(point[0] - circle[0], point[1] - circle[1]) < circle[2]:
        return True
    return False


def line_intersect_circle(line: Line, circle):
    """
    ### Line intersect circle test.
    circle: (x, y, r)
    line: [p1, p2]
    """
    if point_in_circle(line[0], circle) or point_in_circle(line[1], circle):
        return True

    oa = np.array([circle[0] - line[0][0], circle[1] - line[0][1]])
    ob = np.array([circle[0] - line[1][0], circle[1] - line[1][1]])
    ao = -oa
    bo = -ob
    ab = np.array([line[0][0] - line[1][0], line[0][1] - line[1][1]])
    ba = -ab
    d = abs(np.cross(ab, ob) / np.linalg.norm(ab))

    if d <= circle[2]:
        if np.dot(ao, ab) > 0 and np.dot(bo, ba) > 0:
            return True


def vertexes_to_lines(vertexes: List[Point]):
    """
    ### From anticlockwise vertexes get anticlockwise lines.
    """
    lines = []
    newvertexes = deepcopy(vertexes)
    newvertexes.append(newvertexes[0])
    for i in range(len(newvertexes) - 1):
        lines.append((newvertexes[i], newvertexes[i + 1]))

    return lines


def lines_to_norm(lines: List[Line]):
    """
    ### Return every norm vector without normlize.
    """
    norms = []

    for line in lines:
        vec = np.array(line[0]) - np.array(line[1])
        norms.append((vec[1], -vec[0]))  # (y,-x) in the left

    return norms


def vertexes_to_norm(vertexes: List[Point]):
    lines = vertexes_to_lines(vertexes)
    return lines_to_norm(lines)


def get_polygon_area(polygon: Polygon):
    def getS(a, b, c):
        """
        Get triangle area.
        """
        return abs(
            ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * 0.5
        )

    total_area = 0
    for i in range(1, len(polygon.vertexes) - 1):
        total_area += getS(
            polygon.vertexes[0], polygon.vertexes[i], polygon.vertexes[i + 1]
        )

    return total_area


def polygon_intersect_polygon(polygon1: Polygon, polygon2: Polygon):
    def dotProduct(nor, points: list):
        res = []

        for p in points:
            res.append(nor[0] * p[0] + nor[1] * p[1])
        return (min(res), max(res))

    sep_axis = polygon1.norms + polygon2.norms

    for sep in sep_axis:
        res1 = dotProduct(sep, polygon1.vertexes)
        res2 = dotProduct(sep, polygon2.vertexes)

        if res1[1] < res2[0] or res1[0] > res2[1]:
            return False
        else:
            continue
    return True


def polygon_intersect_line(polygon: Polygon, line: Line):
    """
    ### Line intersect this polygon ?
    line: [p1(x,y), p2]
    """

    for l in polygon.lines:
        if line_intersect_line(line, l):
            return True

    return False


def point_in_polygon(polygon: Polygon, point: Point):
    """
    ### Point in polygon ?

    Point: (x, y)
    """

    for l in polygon.lines:
        if not to_left(l, point):
            return False

    return True


def polygon_in_polygon(lhs_polygon: Polygon, rhs_polygon: Polygon):
    """
    ### Polygon in polygon ?

    """
    for vertex in lhs_polygon.vertexes:
        if not point_in_polygon(rhs_polygon, vertex):
            return False

    return True


def get_polygon_halfspaces(polygon: Polygon):
    """
    Return A, b, the polygon can represent A@[x,y] <= b
    [x,y] in polygon.
    """

    N = len(polygon.lines)
    A_ret = np.zeros((N, 2))
    b_ret = np.zeros((N, 1))
    for i in range(N):
        v1, v2 = polygon.lines[i][1], polygon.lines[i][0]
        ab = np.zeros((2, 1))

        if abs(v1[0] - v2[0]) < 1e-10:
            if v2[1] < v1[1]:
                Atmp = np.array([1, 0])
                btmp = v1[0]
            else:
                Atmp = np.array([-1, 0])
                btmp = -v1[0]
        elif abs(v1[1] - v2[1]) < 1e-10:
            if v1[0] < v2[0]:
                Atmp = np.array([0, 1])
                btmp = v1[1]
            else:
                Atmp = np.array([0, -1])
                btmp = -v1[1]
        else:
            temp1 = np.array([[v1[0], 1], [v2[0], 1]])
            temp2 = np.array([[v1[1]], [v2[1]]])
            ab = np.linalg.inv(temp1) @ temp2

            a = ab[0, 0]
            b = ab[1, 0]
            if v1[0] < v2[0]:
                Atmp = np.array([-a, 1])
                btmp = b
            else:
                Atmp = np.array([a, -1])
                btmp = -b

        A_ret[i, :] = Atmp
        b_ret[i, :] = btmp

    return A_ret, b_ret


def test_scale_halfspaces():
    import random

    shape_points = [(random.randint(0, 10), random.randint(0, 10)) for i in range(3)]

    polygon = Polygon(shape_points)

    A, b = get_polygon_halfspaces(polygon)


def test_halfspaces():
    import random
    import matplotlib.pyplot as plt
    import utils

    shape_points = [(random.randint(0, 10), random.randint(0, 10)) for i in range(3)]

    polygon = Polygon(shape_points)

    A, b = get_polygon_halfspaces(polygon)

    for _ in range(100):
        point = (random.randint(0, 10), random.randint(0, 10))

        point_array = np.array([point])
        flag1 = point_in_polygon(polygon, point)
        flag2 = np.all(A @ point_array.T < b)

        plt.cla()
        utils.plot_polygon(polygon)
        plt.plot(point[0], point[1], "ro")
        plt.draw()
        plt.pause(0.1)
        if flag1 == flag2:
            print(f"\033[032m[test halfspaces pass, {flag1}]\033[0m")
        else:
            print(f"\033[031m[test halfspaces fail, {flag1}]\033[0m")


def test_single_area():
    import random
    import matplotlib.pyplot as plt
    import utils

    shape_point = [(9.75, 3.0), (7.25, 3.0), (7.25, 9.0), (9.75, 9.0)]

    vehicle_point = (9.4555, 5.60)

    obstacle_polygon = Polygon(shape_point)
    area = get_polygon_area(obstacle_polygon)
    utils.plot_polygon(obstacle_polygon)
    plt.plot(vehicle_point[0], vehicle_point[1], "ro")
    plt.draw()
    plt.pause(0.1)
    total_area = 0
    for l in obstacle_polygon.lines:
        a, b, c = l[0], l[1], vehicle_point
        total_area += (
            np.fabs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * 0.5
        )
    print(f"\033[032m[in polygon , {area,total_area}]\033[0m")
    plt.show()


def test_area():
    import random
    import matplotlib.pyplot as plt
    import utils

    for i in range(100):
        shape_points = [
            (random.randint(0, 10), random.randint(0, 10)) for i in range(4)
        ]
        convexhull_points = pointset_to_convex_hull(shape_points)
        convex_polygon = Polygon(convexhull_points)
        utils.plot_polygon(convex_polygon)

        plt.draw()
        area = get_polygon_area(convex_polygon)

        point = (random.randint(0, 10), random.randint(0, 10))
        plt.plot(point[0], point[1], "ro")
        plt.draw()
        plt.pause(0.5)

        total_area = 0
        for l in convex_polygon.lines:
            a, b, c = l[0], l[1], point
            # total_area += (
            #     np.fabs(
            #         c[0] * a[1]+ a[0] * b[1]+ b[0] * c[1]- c[0] * b[1]- a[0] * c[1]- b[0] * a[1]
            #     )
            #     * 0.5
            # )
            total_area += (
                np.fabs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                * 0.5
            )
        if point_in_polygon(convex_polygon, point):
            if abs(total_area - area) < 1e-3:
                print(f"\033[032m[in polygon , test pass, {area==total_area}]\033[0m")
            else:
                print(f"\033[031m[in polygon , test fail, {area,total_area}]\033[0m")
        else:
            if abs(total_area - area) < -1e-3:
                print(f"\033[031m[out polygon , test fail, {area,total_area}]\033[0m")
            else:
                print(f"\033[032m[out polygon , test pass, {total_area>=area}]\033[0m")

        plt.pause(0.1)
        plt.cla()


def test_convex_hull():
    import random
    import matplotlib.pyplot as plt
    import utils

    for i in range(50):
        shape_points = [
            (random.randint(0, 10), random.randint(0, 10)) for i in range(7)
        ]
        plt.cla()
        utils.plot_polygon(Polygon(shape_points))
        plt.pause(0.1)
        plt.draw()
        convexhull_points = pointset_to_convex_hull(shape_points)
        utils.plot_polygon(Polygon(convexhull_points))
        plt.pause(0.5)
        plt.draw()


def test_polygon_eq():
    import random
    import matplotlib.pyplot as plt
    import utils

    for i in range(50):
        shape_points = [
            (random.randint(0, 10), random.randint(0, 10)) for i in range(7)
        ]

        p1 = Polygon(shape_points)
        # shape_points = [
        #     (random.randint(0, 10), random.randint(0, 10)) for i in range(7)
        # ]
        p2 = Polygon(shape_points)
        if p1 == p2:
            print("p1 == p2")
        print(f"\033[032m[polygon_eq, test pass, {p1!=p2}]\033[0m")


def test_polygon_eq_list():
    import random
    import matplotlib.pyplot as plt
    import utils

    for i in range(5):
        shape_points = [
            (random.randint(0, 10), random.randint(0, 10)) for i in range(7)
        ]

        plist = [Polygon(shape_points)]

        p = Polygon(shape_points)
        if p in plist:
            print(f"\033[032m[polygon_eq, test pass, {p}]\033[0m")


def test_polygon_container():
    import random
    import matplotlib.pyplot as plt
    import utils

    polygon_container = PolygonContainer()
    for i in range(5):
        shape_points = [
            (random.randint(0, 10), random.randint(0, 10)) for i in range(7)
        ]

        polygon_container += Polygon(shape_points)

    print(len(polygon_container))

    for polygon in polygon_container:
        print(polygon)

    print(len(polygon_container))


if __name__ == "__main__":
    test_polygon_container()
