import torch
from mmasim_kernels import MMAKernel


def probe_fdpa_output_precision(kernel: MMAKernel, i: int = 0, j: int = 1) -> int:
    a = torch.zeros([kernel.k], dtype=kernel.a_type)
    b = torch.zeros([kernel.k], dtype=kernel.b_type)
    c = torch.tensor(0.0, dtype=kernel.c_type)
    Ua, Ub = 2.0**7, 2.0**7
    U = Ua * Ub
    a[i], b[i] = Ua, Ub
    eps = 0.5 * U
    a[j], b[j] = 0.5 * Ua, Ub
    p = 1
    while kernel.dpa(a, b, c).item() == U + eps:
        p += 1
        eps *= 0.5
        if a[j].item() < b[j].item():
            b[j] *= 0.5
        else:
            a[j] *= 0.5
    return p - 1


def probe_output_rounding(
    kernel: MMAKernel, output_prec: int, i: int = 0, j: int = 1
) -> dict[float, float]:
    a = torch.zeros([kernel.k], dtype=kernel.a_type)
    b = torch.zeros([kernel.k], dtype=kernel.b_type)
    c = torch.tensor(0.0, dtype=kernel.c_type)
    Ua, Ub = 2.0**7, 2.0**7
    h_prec = output_prec // 2
    va = Ua * 2.0 ** (-h_prec)
    vb = Ub * 2.0 ** (h_prec - output_prec)
    res = {}
    a[i], b[i] = Ua, Ub
    for x in [0.25, 0.75, 0.5, 1.5]:
        a[j], b[j] = x * va, vb
        res[x] = (kernel.dpa(a, b, c).item() - Ua * Ub) / (va * vb)
    a[i], b[i] = -Ua, Ub
    for x in [-0.25, -0.75, -0.5, -1.5]:
        a[j], b[j] = x * va, vb
        res[x] = (kernel.dpa(a, b, c).item() + Ua * Ub) / (va * vb)
    return res


def probe_output_rne_precision(
    kernel: MMAKernel, output_prec: int, i: int = 0, j: int = 1, k: int = 2
) -> int:
    a = torch.zeros([kernel.k], dtype=kernel.a_type)
    b = torch.zeros([kernel.k], dtype=kernel.b_type)
    c = torch.tensor(0.0, dtype=kernel.c_type)
    Ua, Ub = 2.0**7, 2.0**7
    h_prec = output_prec // 2
    va = Ua * 2.0 ** (-h_prec)
    vb = Ub * 2.0 ** (h_prec - output_prec)
    a[i], b[i] = Ua, Ub
    a[j], b[j] = 0.5 * va, vb
    a[k], b[k] = 0.5 * va, 0.5 * vb
    rne_prec = output_prec + 2
    while kernel.dpa(a, b, c).item() == Ua * Ub + va * vb:
        rne_prec += 1
        if a[k].item() < b[k].item():
            b[k] *= 0.5
        else:
            a[k] *= 0.5
    return rne_prec - 1
