import torch
from mmasim_kernels import MMAKernel


def is_fdpa_ab_renormalized(
    kernel: MMAKernel, F: int, i: int = 0, j: int = 1, k: int = 2
) -> bool:
    a = torch.zeros([kernel.k], dtype=kernel.a_type)
    b = torch.zeros([kernel.k], dtype=kernel.b_type)
    c = torch.tensor(0.0, dtype=kernel.c_type)
    Ua, Ub = 2.0**7, 2.0**7
    a[i], b[i] = 1.5 * Ua, Ub
    a[j], b[j] = 1.5 * Ua, -Ub
    hF = F // 2
    va = Ua * 2.0 ** (-hF)
    vb = Ub * 2.0 ** (hF - F)
    a[k], b[k] = va, vb
    return kernel.dpa(a, b, c).item() != va * vb
