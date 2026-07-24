import torch
from mmasim_kernels import MMAKernel


def probe_exponent_for_zero_rne_fp16(
    kernel: MMAKernel, F: int, i: int = 0, j: int = 1
) -> int:
    assert kernel.a_type == torch.float16
    a = torch.zeros([kernel.k], dtype=kernel.a_type)
    b = torch.zeros([kernel.k], dtype=kernel.b_type)
    c = torch.tensor(0.0, dtype=kernel.c_type)
    e_lsb = -26
    a[i], b[i] = 2.0**-11, 2.0**-14
    a[j], b[j] = 0.25, 2.0**-24
    while kernel.dpa(a, b, c).item() != 0.0:
        e_lsb -= 1
        a[j] *= 0.5
    return e_lsb + 1 + F


def probe_exponent_for_zero_rz_fp32(
    kernel: MMAKernel, F: int, i: int = 0, j: int = 1
) -> int:
    assert kernel.a_type in [torch.float32, torch.bfloat16]
    a = torch.zeros([kernel.k], dtype=kernel.a_type)
    b = torch.zeros([kernel.k], dtype=kernel.b_type)
    c = torch.tensor(0.0, dtype=kernel.c_type)
    e_lsb = -126 - 24
    a[i], b[i] = 2.0**-23, 2.0**-126
    a[j], b[j] = -(2.0**-24), 2.0**-126
    while kernel.dpa(a, b, c).item() == 0.0:
        e_lsb -= 1
        a[j] *= 0.5
    return e_lsb + 1 + F
