import torch
from mmasim_kernels import MMAKernel


def probe_fdpa_c_rounding(
    kernel: MMAKernel, F: int, i: int = 0, j: int = 1
) -> dict[float, float]:
    a = torch.zeros([kernel.k], dtype=kernel.a_type)
    b = torch.zeros([kernel.k], dtype=kernel.b_type)
    c = torch.tensor(0.0, dtype=kernel.c_type)
    Ua, Ub = 2.0**7, 2.0**7
    a[i], b[i] = Ua, Ub
    a[j], b[j] = Ua, -Ub
    ulp = 2.0**-F * Ua * Ub
    res = {}
    for x in [0.25, 0.75, 0.5, 1.5]:
        c[None] = x * ulp
        res[x] = kernel.dpa(a, b, c).item() / ulp
        c[None] = -x * ulp
        res[-x] = kernel.dpa(a, b, c).item() / ulp
    return res


def probe_fdpa_ab_rounding(
    kernel: MMAKernel, F: int, i: int = 0, j: int = 1, k: int = 2
) -> dict[float, float]:
    a = torch.zeros([kernel.k], dtype=kernel.a_type)
    b = torch.zeros([kernel.k], dtype=kernel.b_type)
    c = torch.tensor(0.0, dtype=kernel.c_type)
    Ua, Ub = 2.0**7, 2.0**7
    a[i], b[i] = Ua, Ub
    a[j], b[j] = Ua, -Ub
    hF = F // 2
    va = Ua * 2.0 ** (-hF)
    vb = Ub * 2.0 ** (hF - F)
    res = {}
    for x in [0.25, 0.75, 0.5, 1.5]:
        a[k], b[k] = x * va, vb
        res[x] = kernel.dpa(a, b, c).item() / (va * vb)
        a[k], b[k] = -x * va, vb
        res[-x] = kernel.dpa(a, b, c).item() / (va * vb)
    return res
