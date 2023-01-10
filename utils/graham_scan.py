import numpy as np
import matplotlib.pyplot as plt


# Get the Start Point in the left buttom
def get_left_bottom_point(x, y, n):
    k = 0
    for i in range(n):
        if y[i] < y[k] or (y[i] == y[k] and x[i] < x[k]):
            k = i
    return k


# Get the arc from p0 to p1
def get_arc(p1_y, p1_x, p0_y, p0_x):
    if p1_x - p0_x == 0:
        if p1_y - p0_y == 0:
            return -1
        else:
            return np.pi/2

    tan = (p1_y - p0_y)/(p1_x - p0_x)
    arc = np.arctan(tan)
    if arc >= 0:
        return arc
    else:
        return np.pi + arc


# Sort points by their tan values
def sort_points_tan(x, y, n, p_start):
    ps_arctan = np.zeros(x.shape)
    for i in range(n):
        ps_arctan[i] = get_arc(y[i], x[i], y[p_start], x[p_start])

    ps_sorted = np.argsort(ps_arctan)
    return ps_sorted


# for two 2d vectors U=(Ux,Uy) V=(Vx,Vy)
# the crossproduct is U x V = Ux*Vy - Uy*Vx
def cross_product(p1_x, p1_y, p2_x, p2_y, p0_x, p0_y):
    return (p1_x - p0_x) * (p2_y - p0_y) - (p2_x - p0_x) * (p1_y - p0_y)


# Graham Scan
def graham_scan(x, y, n):
    """ This function returns a python list """
    x = np.asarray(x)
    y = np.asarray(y)

    p_start = get_left_bottom_point(x, y, n)
    ps_sorted = sort_points_tan(x, y, n, p_start)

    ps_result = []
    ps_result.append(ps_sorted[0])
    ps_result.append(ps_sorted[1])
    ps_result.append(ps_sorted[2])

    top = 2
    for i in range(3, n):
        while (top >= 1 and cross_product(x[ps_sorted[i]], y[ps_sorted[i]], x[ps_result[top]], y[ps_result[top]] \
                    , x[ps_result[top - 1]], y[ps_result[top - 1]]) >= 0):
            ps_result.pop()
            top -= 1
        ps_result.append(ps_sorted[i])
        top += 1

    return ps_result


def test():
    n = 4
    points = np.random.rand(n, 2)

    for point in points:
        plt.scatter(point[0], point[1], marker='o', c='y')
 
    x = points[:, 0]
    y = points[:, 1]
    result = graham_scan(x, y, n)
 
    length = len(result)
    for i in range(0, length-1):
        plt.plot([x[result[i]], x[result[i+1]]], [y[result[i]], y[result[i+1]]], c='r')
    plt.plot([x[result[0]], x[result[length-1]]], [y[result[0]], y[result[length-1]]], c='r')
 
    plt.show()


# if __name__ == "__main__":
#     test()
