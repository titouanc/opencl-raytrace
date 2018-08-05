import logging
import numpy as np
import pyopencl as cl
import pyopencl.array as clarr
from raytrace import line2seg2d, line2tri3d

__GPU__ = True

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

if __GPU__:
    logger.info("=== USING GPU BACKEND ===")

    ctx = cl.create_some_context()
    Q = cl.CommandQueue(ctx)
    prog = cl.Program(ctx, open("k_linalg.cl").read()).build()

    def k(kernel):
        def inner(*args):
            args = np.array(args, dtype=np.float32).flatten()
            args_g = clarr.to_device(Q, args)
            res_g = clarr.zeros(Q, (1,), dtype=np.float32)
            kernel(Q, (1,), (1,), res_g.data, args_g.data).wait()
            return res_g.get()[0]
        return inner

    line2seg2d = k(prog.k_line2seg2d)  # NOQA
    line2tri3d = k(prog.k_line2tri3d)  # NOQA

# I'm lazy, m'kay
a = lambda *args: np.array(list(args), dtype=np.float32)  # NOQA

up = a(0, 1)
right = a(1, 0)


def test_horizontal_segment():
    origin = a(0, 0)
    p0 = a(-1, 1)
    segment = (p0, p0 + 2 * right)

    # Segment and line are perpendicular
    assert line2seg2d(origin, up, *segment) == 1
    # Same, but the line starts lower
    assert line2seg2d(origin - up, up, *segment) == 2
    # Line reaches segment at the right hand side point
    assert line2seg2d(origin, up + right, *segment) == 1
    # Line direction goes not to the segment
    assert np.isnan(line2seg2d(origin, -up, *segment))
    # Segment and line are parallel
    assert np.isnan(line2seg2d(origin, right, *segment))
    # The line pass on the right of the segment
    assert np.isnan(line2seg2d(origin, up + 2 * right, *segment))


def test_vertical_segment():
    origin = a(0, 0)
    p0 = a(1, 1)
    segment = (p0, p0 - 2 * up)

    # Segment and line are perpendicular
    assert line2seg2d(origin, right, *segment) == 1
    # Line reaches upper point of the segment
    assert line2seg2d(origin, up + right, *segment) == 1
    # Segment and line are parallel
    assert np.isnan(line2seg2d(origin, up, *segment))
    # The line pass above the segment
    assert np.isnan(line2seg2d(origin, 2 * up + right, *segment))


def test_diagonal_segment():
    origin = a(0, 0)
    p0 = a(0, 2)
    segment = (p0, p0 - 2 * up + 2 * right)

    # Segment and line are perpendicular
    assert line2seg2d(origin, up + right, *segment) == 1
    # Segment touch line at upper-left point
    assert line2seg2d(origin, up, *segment) == 2
    # Segment touch line at lower-right point
    assert line2seg2d(origin, right, *segment) == 2
    # Segment and line are parallel
    assert np.isnan(line2seg2d(origin, up - right, *segment))
    # Line pass on the left of the segment
    assert np.isnan(line2seg2d(origin, 2 * up - right, *segment))
    # Line pass below the segment
    assert np.isnan(line2seg2d(origin, 2 * right - up, *segment))


def test_vertical_triangle():
    p0 = np.array([0, 1, 0])
    p1 = np.array([1, 1, 0])
    p2 = np.array([1, 1, 1])
    o = np.array([.5, .5, 0])
    d = np.array([0, 1, 1])

    assert line2tri3d(o, d, p0, p1, p2) == .5

    d = np.array([0, 1, 3])
    assert np.isnan(line2tri3d(o, d, p0, p1, p2))
