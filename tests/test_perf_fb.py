import numpy as np
from utils.perf_fb import PerfFB
from utils.xlogging import logger


def test_simple():
    perf_fb = PerfFB()
    assert perf_fb is not None

def test_run_c():
    """ Tests running cordic """

    perf_fb = PerfFB('./input')
    perf_fb.set_hls_incl_dir('/opt/xilinx/Vivado/2019.2/include')

    got = perf_fb._run_c([[np.pi/3, 20]])
    logger.debug(got)
    assert isinstance(got, list)
    assert isinstance(got[0], list)
    assert isinstance(got[0][0], float)

def test_rmse():
    """ Tests RMSE computation """

    perf_fb = PerfFB()
    def stub(a):
        return [[1,2],[-3,4]]
    
    gt = {'1,2':'0.75,2',
          '-1,-3':'-3.1,8'}
    
    got = perf_fb._rmse(gt, stub)
    logger.debug(got)

    assert np.all(np.isclose(got, [0.33489057,0.5], rtol=1e-5))

def test_utilization():
    """ Tests parsing HLS report for utilization feedback """

    perf_fb = PerfFB('./tests/resource')
    exp = {'BRAM_18K':(65,280),
           'DSP48E':(2,220),
           'FF':(17658,106400),
           'LUT':(52125,53200),
           'URAM':(0,0)}
    got = perf_fb.utilization()
    logger.debug(got)
    assert got == exp

def test_latency():
    perf_fb = PerfFB('./tests/resource')

    exp = [120.0, 120.0]
    got = perf_fb._latency()
    assert got == exp

def test_suggestion():
    perf_fb = PerfFB('./tests/resource')

    logger.debug(f'\n{perf_fb.suggestion()}')
