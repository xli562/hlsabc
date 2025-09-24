from pathlib import Path
import numpy as np
import pytest

import allo
from allo.ir.types import int32, float32
from utils.estimator import Estimator
from utils.llm import LLM, GFLASH, GLITE
from utils.xlogging import logger


example_prj_path = Path('tests/resource/example.prj')

def get_schedule():
    """ Returns an Allo `Schedule`.
    
    :return: an Allo `Schedule`, used for testing.
    """
    def conv2D_lb_wb_schedule(A: int32[10, 10]) -> int32[8, 8]:
        B: int32[8, 8] = 0
        for y, x in allo.grid(8, 8):
            v: int32 = 0
            for r, c in allo.reduction(3, 3):
                v += A[y + r, x + c]
            B[y, x] = v
        return B

    s = allo.customize(conv2D_lb_wb_schedule)
    s.reshape(s.B, (8, 2, 4))
    mod = s.build()

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
    """ Tests __init__ of estimator. """

    s = get_schedule()
    _ = Estimator(s)

@pytest.mark.slow
def test_get_opcount_per_kernel():
    """ Tests if get_opcount_per_kernel returns an integer.
    
    :raises warning: incorrect estimate for operation count
    """
    exp = 576
    s = get_schedule()
    estimator = Estimator(s)
    estimator.estm_model = LLM(GFLASH)
    estimator.critic_model = LLM(GFLASH)
    estimator.extract_model = LLM(GLITE)
    got = estimator.get_opcount_per_kernel()
    assert isinstance(got, int), f'Expects type int, got {got}.'
    if got != exp:
        warning_str = 'Incorrect estimate for operation count. '
        warning_str += f'Expected {exp} operations, got {got} operations.'
        logger.warning(warning_str)

@pytest.mark.slow
def test_get_bytes_per_kernel():
    """ Tests if get_bytes_per_kernel returns an integer.
    
    :raises warning: incorrect estimate for external memory access
    """
    exp = 2816
    s = get_schedule()
    estimator = Estimator(s)
    estimator.estm_model = LLM(GFLASH)
    estimator.critic_model = LLM(GFLASH)
    estimator.extract_model = LLM(GLITE)
    got = estimator.get_bytes_per_kernel()
    assert isinstance(got, int), f'Expects type int, got {got}.'
    if got != exp:
        warning_str = 'Incorrect estimate for external memory access. '
        warning_str += f'Expected {exp} bytes, got {got} bytes.'
        logger.warning(warning_str)

def test__get_hls_sources():
    """ Tests if get_hls_sources parses the project directory correctly. """
    
    exp = set([example_prj_path / 'kernel.cpp',
               example_prj_path / 'host.cpp'])
    s = get_schedule()
    estimator = Estimator(s)
    got = set(estimator._get_hls_sources(example_prj_path))
    assert got == exp, f'Expects {exp}, got {got}.'

def test__get_hls_rpts():
    """ Tests if get_hls_rpts parses the project directory correctly. """
    exp = [example_prj_path / 'out.prj/solution1/syn/report/conv2D_lb_wb_schedule_csynth.rpt']
    s = get_schedule()
    estimator = Estimator(s)
    got = estimator._get_hls_rpts(example_prj_path)
    assert got == exp, f'Expects {exp}, got {got}.'

def test__get_clk_period():
    """ Tests if get_clk_period can correctly read the clock period from
    the json report. """
    
    exp = 3.33e-9
    s = get_schedule()
    estimator = Estimator(s)
    got = estimator._get_clk_period(example_prj_path)
    assert np.isclose(got, exp, rtol=1e-3), f'Expects {exp}, got {got}.'

@pytest.mark.slow
def test_get_seconds_per_kernel():
    exp = 5.974e-6
    s = get_schedule()
    estimator = Estimator(s)
    estimator.estm_model = LLM(GFLASH)
    estimator.critic_model = LLM(GFLASH)
    estimator.extract_model = LLM(GLITE)
    got = estimator.get_seconds_per_kernel(example_prj_path)
    assert isinstance(got, float), f'Expects type float, got {got}.'
    if not np.isclose(got, exp, rtol=1e-2):
        warning_str = 'Incorrect throughput estimate. '
        warning_str += f'Expected {exp} [s / kernel], got {got} [s / kernel].'
        logger.warning(warning_str)

@pytest.mark.slow
def test_get_coords():
    exp = np.asarray([9.64174877e+07, 2.04545455e-01])
    s = get_schedule()
    estimator = Estimator(s)
    estimator.estm_model = LLM(GFLASH)
    estimator.critic_model = LLM(GFLASH)
    estimator.extract_model = LLM(GLITE)
    got = np.asarray(estimator.get_coords(example_prj_path))
    assert len(got) == 2 \
           and isinstance(got[0], float) \
           and isinstance(got[1], float), \
            f'Expects [float,float], got {got}.'
    if not np.allclose(got, exp, rtol=1e-1):
        warning_str = 'Incorrect coordinate estimate. '
        warning_str += f'Expected {exp}, got {got}.'
        logger.warning(warning_str)
