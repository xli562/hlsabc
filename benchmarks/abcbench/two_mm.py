# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T
from utils.xlogging import logger, log_data
from utils.estimator import Estimator


def two_mm_np(A, B, C, D, alpha, beta):
    out_AB = np.dot(A, B)
    out_ABC = np.dot(out_AB, C)
    output = out_ABC * beta + D * alpha
    return output


def mm1[
    T: (float32, int32), P: int32, Q: int32, R: int32
](A: "T[P, Q]", B: "T[Q, R]", out_AB: "T[P, R]"):
    for i0, j0 in allo.grid(P, R, name="mm1"):
        for k0 in allo.reduction(Q):
            out_AB[i0, j0] += A[i0, k0] * B[k0, j0]


def mm2[
    T: (float32, int32), P: int32, R: int32, S: int32
](out_AB: "T[P, R]", C: "T[R, S]", out_ABC: "T[P, S]"):
    for i1, j1 in allo.grid(P, S, name="mm2"):
        for k1 in allo.reduction(R):
            out_ABC[i1, j1] += out_AB[i1, k1] * C[k1, j1]


def ele_add[
    T: (float32, int32), P: int32, S: int32
](out_ABC: "T[P, S]", D: "T[P, S]", output: "T[P, S]"):
    for i2, j2 in allo.grid(P, S):
        output[i2, j2] = out_ABC[i2, j2] * beta + D[i2, j2] * alpha


def kernel_2mm[
    T: (float32, int32), P: int32, R: int32, Q: int32, S: int32
](A: "T[P, Q]", B: "T[Q, R]", C: "T[R, S]", D: "T[P, S]") -> "T[P, S]":
    out_AB: T[P, R] = 0
    out_ABC: T[P, S] = 0
    output: T[P, S]
    mm1[T, P, Q, R](A, B, out_AB)
    mm2[T, P, R, S](out_AB, C, out_ABC)
    ele_add[T, P, S](out_ABC, D, output)
    return output


def two_mm(concrete_type, p, r, q, s, variant:int):
    assert variant in range(0, 2**11), f'Got {variant}, expected within [0, 2**11).'
    alpha = 0.1
    beta = 0.5

    sch0 = allo.customize(mm1, instantiate=[concrete_type, p, q, r])
    if (variant >> 0) & 1:
        sch0.reorder("k0", "j0")
    if (variant >> 1) & 1:
        sch0.buffer_at(sch0.out_AB, axis="i0")
    if (variant >> 2) & 1:
        sch0.pipeline("k0")
    if (variant >> 3) & 1:
        i0 = sch0.get_loops("mm1")["mm1"]["i0"]
        sch0.dataflow(i0)

    sch1 = allo.customize(mm2, instantiate=[concrete_type, p, r, s])
    if (variant >> 4) & 1:
        sch1.reorder("k1", "j1")
    if (variant >> 5) & 1:
        sch1.buffer_at(sch1.out_ABC, axis="i1")
    if (variant >> 6) & 1:
        sch1.pipeline("k1")
    if (variant >> 7) & 1:
        i1 = sch1.get_loops("mm2")["mm2"]["i1"]
        sch1.dataflow(i1)

    sch2 = allo.customize(ele_add, instantiate=[concrete_type, p, s])
    if (variant >> 8) & 1:
        sch2.pipeline("j2")

    sch = allo.customize(kernel_2mm, instantiate=[concrete_type, p, r, q, s])
    sch.compose(sch0)
    sch.compose(sch1)
    sch.compose(sch2)

    # sch.to(sch.out_AB, "mm2")
    # sch.to(sch.out_ABC, "ele_add")

    if (variant >> 9) & 1:
        sch.dataflow("mm1")
    if (variant >> 10) & 1:
        sch.dataflow("mm2")

    return sch


def test_two_mm(variant:int):
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    P = psize["two_mm"][test_psize]["P"]
    Q = psize["two_mm"][test_psize]["Q"]
    R = psize["two_mm"][test_psize]["R"]
    S = psize["two_mm"][test_psize]["S"]
    dtype = np.float32
    A = np.random.rand(P, Q).astype(dtype)
    B = np.random.rand(Q, R).astype(dtype)
    C = np.random.rand(R, S).astype(dtype)
    D = np.random.rand(P, S).astype(dtype)
    logger.debug(variant)
    sch = two_mm(float32, P, R, Q, S, variant)
    mod = sch.build()
    output = mod(A, B, C, D)
    output_ref = two_mm_np(A, B, C, D, 0.1, 0.5)
    np.testing.assert_allclose(output, output_ref, rtol=1e-5, atol=1e-5)
    # Run roofline estimation
    rl = Estimator(sch)
    coords = rl.get_coords()
    log_data(f'{bin(variant):<5} {coords[0]:<35} {coords[1]:<35}')
    return -1 if coords == (0,0) else 0


if __name__ == "__main__":
    pytest.main([__file__])
