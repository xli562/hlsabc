from pathlib import Path
import numpy as np
import pytest

import allo
from allo.ir.types import int32, float32
from utils.roofline import Roofline
from utils.xlogging import logger



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
    """ Tests if get_opcount_per_kernel returns an integer.
    
    :raises warning: incorrect estimate for operation count
    """

    exp = 576
    s = get_schedule()
    roofline = Roofline(s)
    got = roofline.get_opcount_per_kernel()
    assert isinstance(got, int), f'Expects type int, got {got}.'
    if got != exp:
        warning_str = 'Incorrect estimate for operation count. '
        warning_str += f'Expected {exp} operations, got {got} operations.'
        logger.warning(warning_str)

def test_get_bytes_per_kernel():
    """ Tests if get_bytes_per_kernel returns an integer.
    
    :raises warning: incorrect estimate for external memory access
    """
    exp = 2560
    s = get_schedule()
    roofline = Roofline(s)
    got = roofline.get_bytes_per_kernel()
    assert isinstance(got, int), f'Expects type int, got {got}.'
    if got != exp:
        warning_str = 'Incorrect estimate for external memory access. '
        warning_str += f'Expected {exp} bytes, got {got} bytes.'
        logger.warning(warning_str)

def test_get_hls_sources():
    """ Tests if get_hls_sources parses the project directory correctly. """
    
    exp = set([Path('tests/resource/example.prj/kernel.cpp'),
              Path('tests/resource/example.prj/host.cpp')])
    s = get_schedule()
    roofline = Roofline(s)
    got = set(roofline._get_hls_sources('tests/resource/example.prj'))
    assert got == exp, f'Expects {exp}, got {got}.'

def test_get_hls_rpts():
    """ Tests if get_hls_rpts parses the project directory correctly. """
    exp = [Path('tests/resource/example.prj/out.prj/solution1/syn/report/conv2D_lb_wb_schedule_csynth.rpt')]
    s = get_schedule()
    roofline = Roofline(s)
    got = roofline._get_hls_rpts('tests/resource/example.prj')
    assert got == exp, f'Expects {exp}, got {got}.'

def test_get_clk_period():
    """ Tests if get_clk_period can correctly read the clock period from
    the json report. """
    
    exp = 3.33
    s = get_schedule()
    roofline = Roofline(s)
    prj_path = Path('tests/resource/example.prj')
    got = roofline._get_clk_period(prj_path)
    assert np.isclose(got, exp, rtol=1e-3), f'Expects {exp}, got {got}.'

def test__get_once_latency_per_kernel():
    """ Tests if _get_once_latency_per_kernel returns a float.
    
    :raises warning: incorrect throughput estimate.
    """
    exp = 1794
    s = get_schedule()
    roofline = Roofline(s)
    prj_path = Path('tests/resource/example.prj')
    file_paths = roofline._get_hls_sources(prj_path)
    file_paths.extend(roofline._get_hls_rpts(prj_path))
    got = roofline._get_once_latency_per_kernel(file_paths, '')
    assert isinstance(got[0], float), f'Expects type int, got {got}.'
    assert isinstance(got[1], str), f'Expects type str, got {got}.'
    if int(got[0]) != exp:
        warning_str = 'Incorrect throughput estimate. '
        warning_str += f'Expected {exp} [s / kernel], got {got} [s / kernel].'
        logger.warning(warning_str)

def test_get_latency_range():
    """ Tests if get_latency_range can correctly read the interval range from
    the json report. """
    
    exp = (1795, 1795)
    s = get_schedule()
    roofline = Roofline(s)
    prj_path = Path('tests/resource/example.prj')
    got = roofline._get_latency_range(prj_path)
    assert got == exp, f'Expects {exp}, got {got}.'

def test_get_latency_per_kernel():
    exp = 1794
    s = get_schedule()
    roofline = Roofline(s)
    prj_path = Path('tests/resource/example.prj')
    file_paths = roofline._get_hls_sources(prj_path)
    file_paths.extend(roofline._get_hls_rpts(prj_path))
    lat_range = roofline._get_latency_range(prj_path)
    got = roofline.get_latency_per_kernel(file_paths, lat_range, tries=1)
    assert isinstance(got, float), f'Expects type float, got {got}.'
    if int(got) != exp:
        warning_str = 'Incorrect throughput estimate. '
        warning_str += f'Expected {exp} [s / kernel], got {got} [s / kernel].'
        logger.warning(warning_str)

def test_get_coords():
    exp = np.asarray([0.3210702341137124, 0.225])
    s = get_schedule()
    roofline = Roofline(s)
    got = np.asarray(roofline.get_coords(tries=1))
    # assert np.all(np.isclose(got, exp, rtol=1e-3))
