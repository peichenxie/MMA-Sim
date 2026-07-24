import torch
from mmasim_kernels import MMAKernel


def probe_fdpa_c_precision(kernel: MMAKernel, i: int = 0, j: int = 1) -> int:
    a = torch.zeros([kernel.k], dtype=kernel.a_type)
    b = torch.zeros([kernel.k], dtype=kernel.b_type)
    c = torch.tensor(0.0, dtype=kernel.c_type)
    Ua, Ub = 2.0**7, 2.0**7
    a[i], b[i] = Ua, Ub
    a[j], b[j] = Ua, -Ub
    c[None] = 0.5 * Ua * Ub
    F = 1
    while c.item() > 0:
        if kernel.dpa(a, b, c).item() != c.item():
            return F - 1
        c *= 0.5
        F += 1
    return -1


def probe_fdpa_ab_precision(
    kernel: MMAKernel, i: int = 0, j: int = 1, k: int = 2
) -> int:
    a = torch.zeros([kernel.k], dtype=kernel.a_type)
    b = torch.zeros([kernel.k], dtype=kernel.b_type)
    c = torch.tensor(0.0, dtype=kernel.c_type)
    Ua, Ub = 2.0**7, 2.0**7
    a[i], b[i] = Ua, Ub
    a[j], b[j] = Ua, -Ub
    a[k], b[k] = 0.5 * Ua, Ub
    F = 1
    while a[k].item() > 0:
        if kernel.dpa(a, b, c).item() != a[k].item() * b[k].item():
            return F - 1
        if a[k].item() < b[k].item():
            b[k] *= 0.5
        else:
            a[k] *= 0.5
        F += 1
    return -1
