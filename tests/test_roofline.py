import numpy as np

import allo
from allo.ir.types import int32, float32
from utils.roofline import Roofline
from utils.xlogging import get_logger


logger = get_logger()

def get_schedule():
    def conv2D_lb_wb_schedule(A: int32[10, 10]) -> int32[8, 8]:
        B: int32[8, 8] = 0
        for y, x in allo.grid(8, 8):
            v: int32 = 0
            for r, c in allo.reduction(3, 3):
                v += A[y + r, x + c]
            B[y, x] = v
        return B

    s = allo.customize(conv2D_lb_wb_schedule)
    # xo, xi = s.split("x", 4)
    # s.split("x", 4)
    # s.reorder("x.outer", "y", "x.inner")
    # LB = s.reuse_at(s.A, "y")
    # WB = s.reuse_at(LB, "x.inner")
    # s.partition(LB, dim=2)
    # s.partition(WB)
    s.reshape(s.B, (8, 2, 4))
    # s.pipeline("y")
    mod = s.build()
    # s.build(target="vivado_hls", mode="csyn", project="conv.prj")()

    np_A = np.random.randint(0, 10, size=(10, 10)).astype(np.int32)
    np_C = np.zeros((8, 2, 4), dtype="int")

    for y in range(0, 8):
        for xo in range(0, 2):
            for xi in range(0, 4):
                for r in range(0, 3):
                    for c in range(0, 3):
                        np_C[y][xo][xi] += np_A[y + r][xi + xo * 4 + c]

    np_B = mod(np_A)

    assert np.array_equal(np_B, np_C), 'Algorithm fails test.'

    return s


def test_simple():
    """ Tests __init__ of Roofline. """

    s = get_schedule()
    _ = Roofline(s)

def test_get_opcount_per_kernel():
    """ Tests if get_opcount_per_kernel returns an integer. """

    s = get_schedule()
    roofline = Roofline(s)
    got = roofline.get_opcount_per_kernel()
    assert isinstance(got, int)

