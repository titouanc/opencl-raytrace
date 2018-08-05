import logging
from functools import partial
from multiprocessing import Pool

import numpy as np

logger = logging.getLogger("linalg")
DTYPE = np.float64

def line2seg2d(O, D, P0, P1):  # NOQA
    """
    Find the intersection between a semi-line and a line segment in 2D
    Return k (>=0) such that O+kD lies between P0 and P1, or NaN if
    such a point does not exist
    O: the origin of the semi-line
    D: the direction of the semi-line from its origin
    P1, P2: The two points delimiting the segment
    """
    logger.debug("Line %s + k%s -> segment [%s,%s]", O, D, P0, P1)
    a = np.matrix([D, P0 - P1]).T
    try:
        a = a.I
    except np.linalg.LinAlgError:
        return np.nan
    b = np.matrix(P0 - O)
    res = a * b.T
    k, l = res[0, 0], res[1, 0]
    return k if (0 <= l <= 1) and (k >= 0) else np.nan


def line2tri3d(O, D, P0, P1, P2):  # NOQA
    # -kD + l(p1-p0) + m(p2-p0) = O - p0
    logger.debug("Line %s + k%s -> triangle <%s,%s,%s>", O, D, P0, P1, P2)
    a = np.matrix([-D, P1 - P0, P2 - P0]).T
    try:
        a = a.I
    except np.linalg.LinAlgError:
        return np.nan
    b = np.matrix(O - P0)
    res = a * b.T
    k, l, m = res[0, 0], res[1, 0], res[2, 0]
    logger.info("k=%s,l=%s,m=%s", k, l, m)
    in_triangle = (0 <= l <= 1) and (0 <= m <= 1) and (0 <= (l + m) <= 1)
    return k if k >= 0 and in_triangle else np.nan


def subtriangle_points(pos):
    x, y = pos[:2] % 1
    a = y > x
    b = y > 1 - x
    offset = np.array([
        [not a, a],
        [b, b],
    ], dtype=np.int)
    return pos[:2].astype(np.int) + offset


def first_entry_point(origin, direction, surface):
    h, w = surface.shape
    p00 = np.array([0, 0], dtype=DTYPE)
    p01 = np.array([0, h], dtype=DTYPE)
    p10 = np.array([w, 0], dtype=DTYPE)
    p11 = np.array([w, h], dtype=DTYPE)

    o = np.array(origin[:2], dtype=DTYPE)
    d = np.array(direction[:2], dtype=DTYPE)

    # Sides of the surface
    sides = {
        'S': (p00, p10),
        'W': (p00, p01),
        'N': (p01, p11),
        'E': (p10, p11),
    }

    # Determine which side of the surface is touched first
    closest = None
    for side, points in sides.items():
        k = line2seg2d(o, d, *points)
        if not np.isnan(k):
            logger.debug("Intercept side %s (%sx direction)", side, k)
            if closest is None or k < closest:
                closest = k

    # The ray doesnt pass over the surface, nothing else to do here
    if closest is None:
        return None

    # Point of intersection
    return origin + closest * direction


def raycast(origin, direction, surface):
    """
    Cast a ray from given origin in given direction, and find the first
    intersection with the volume constructed over the given surface, if any.
    return the 3D point of intersection, or None if such a point does not exist
    origin: a 3D point (np.array shape=(3,))
    direction: a 3D point (np.array shape=(3,))
    surface: a 2D array (np.array shape=(h,w))
    """

    # Normalize direction
    # direction = direction / np.sqrt(
    #     direction[0]**2 + direction[1]**2 + direction[2]**2
    # )

    # Find first entry point
    h, w = surface.shape
    p = first_entry_point(origin, direction, surface)
    if p is None:
        return np.nan

    while 0 <= p[0] < w - 1 and 0 <= p[1] < h - 1:
        print(p)

        # Find vertices (x,y) coords of the current triangle
        p0 = .5 + np.floor(p[:2])
        p1, p2 = subtriangle_points(p)

        # Find their heights
        x, y = np.floor(p[:2]).astype(np.int)
        local_h = surface[y:y + 2, x:x + 2]

        if local_h.min() < p[2] < local_h.max():
            hp0 = surface[y:y+2,x:x+2].mean()  # NOQA
            hp1, hp2 = surface[p1[1], p1[0]], surface[p2[1], p2[0]]

            # Find if we intercept the triangle
            k = line2tri3d(
                origin, direction,
                np.array(list(p0) + [hp0]),
                np.array(list(p1) + [hp1]),
                np.array(list(p2) + [hp2]),
            )
        else:
            k = np.nan

        # If not, go to next triangle in this direction
        if np.isnan(k):
            for seg in [(p0, p1), (p1, p2), (p2, p0)]:
                nk = line2seg2d(p[:2], direction[:2], *seg)
                if not np.isnan(nk):
                    if nk > 1e-10:
                        p += nk * direction
                        break
                    else:
                        p += direction * 0.01
        else:
            p = origin + k * direction
            return p[2]
    return np.nan


def _raycast(args=tuple(), kwargs={}):
    return raycast(*args, **kwargs)


def render(origin, direction, surface, aperture=2.5, width=160, height=90):
    if width > height:
        a_h = aperture
        a_v = aperture * height / float(width)
    else:
        a_h = aperture * width / float(height)
        a_v = aperture
    horizontal = np.sin(np.linspace(-a_h / 2, a_h / 2, width))
    vertical = np.sin(np.linspace(a_v / 2, -a_v / 2, height))

    rays = [
        (origin, direction + np.array([x, 0, y]), surface)
        for i, y in enumerate(vertical)
        for j, x in enumerate(horizontal)
    ]
    pixels = [_raycast(r) for r in rays]
    # pixels = Pool(8).map(_raycast, rays)
    return np.array(pixels, dtype=DTYPE).reshape(height, width)


if __name__ == "__main__":
    from time import time
    from sys import argv

    n = 100
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    Z = ((X - n / 2)**2 + (Y - n / 2)**2) / 2

    origin = np.array([n / 2, -2 * n, 0])
    direction = np.array([0, 1, .5])

    t0 = time()
    res = render(origin, direction, Z, aperture=1.2, width=80, height=120)
    t1 = time()
    print("Rendering time: %ss" % (t1 - t0))

    if len(argv) > 1 and argv[1] == 'plot':
        import matplotlib.pyplot as plt
        plt.imshow(res)
        plt.show()
