import torch
from mmasim_kernels import MMAKernel


def is_fdpa_ab_overflowing(kernel: MMAKernel, i: int = 0, j: int = 1) -> bool:
    a = torch.zeros([kernel.k], dtype=kernel.a_type)
    b = torch.zeros([kernel.k], dtype=kernel.b_type)
    c = torch.tensor(0.0, dtype=kernel.c_type)
    Ua = torch.finfo(kernel.a_type).max
    Ub = torch.finfo(kernel.b_type).max
    a[i], b[i] = Ua, Ub
    a[j], b[j] = -Ua, Ub
    return kernel.dpa(a, b, c).item() != 0


def is_fdpa_ab_underflowing(kernel: MMAKernel, i: int = 0, j: int = 1) -> bool:
    a = torch.zeros([kernel.k], dtype=kernel.a_type)
    b = torch.zeros([kernel.k], dtype=kernel.b_type)
    c = torch.tensor(0.0, dtype=kernel.c_type)
    eps = torch.finfo(kernel.a_type).smallest_normal
    a[i], b[i] = eps, 0.5
    a[j], b[j] = eps, 0.5
    return kernel.dpa(a, b, c).item() == 0.0
