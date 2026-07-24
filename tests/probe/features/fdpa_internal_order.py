import torch
from mmasim_kernels import MMAKernel


def probe_internal_summation_order(kernel: MMAKernel, F: int) -> torch.Tensor:
    Ua, Ub = 2.0**7, 2.0**7
    hF = F // 2
    va = Ua * 2.0 ** (-hF - 1)
    vb = Ub * 2.0 ** (hF - F)
    k = kernel.k
    a = torch.full([k], va, dtype=kernel.a_type)
    b = torch.full([k], vb, dtype=kernel.b_type)
    c = torch.tensor(va * vb, dtype=kernel.c_type)
    res = torch.zeros([k + 1, k + 1], dtype=torch.int32)
    # aibi = U, ajbj = -U
    for i in range(k):
        a[i], b[i] = Ua, Ub
        for j in range(i):
            a[j], b[j] = Ua, -Ub
            res[i, j] = int(kernel.dpa(a, b, c).item() / (va * vb))
            a[j], b[j] = va, vb
        a[i], b[i] = va, vb
    # c = U, ajbj = -U
    c[None] = Ua * Ub
    for j in range(k):
        a[j], b[j] = Ua, -Ub
        res[k, j] = int(kernel.dpa(a, b, c).item() / (va * vb))
        a[j], b[j] = va, vb
    return res
